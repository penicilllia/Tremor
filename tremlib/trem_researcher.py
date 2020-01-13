import numpy as np
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from typing import Tuple, Callable

from calibration_info import CalibrationInfo, MM_IN_INCH
from file_helper import FilesHelper, OS_SLASH
from aruco_helper import ArucoHelper

# TODO:
# 1) Протестировать калибровку камеры
# 2) Протестировать распознавание меток
# 3) Проверить работу out_frame и draw_axes для get_xyz


class TremResearcher(ArucoHelper):
    # константа для указания, что анализ производится в трехмерном пространстве
    AXES_COUNT = 3

    def __init__(self, marks_num: int = 5, marks_size_mm: float = 10):
        self.calibration_info = None  # калибровочная информация
        self.marks_num = marks_num
        self.real_sqr_size = marks_size_mm  # реальный размер метки в мм

        # коллецкция id используемых меток
        self.all_ids = list(range(marks_num))

    @property
    def calibrated(self) -> bool:
        return self.calibration_info is not None

    def get_xyz(self, frame: np.ndarray,
                out_frame: bool = False,
                draw_axes: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        # Есть потенциал для использования углов меток в пространстве,
        # на данном этапе углы не задействованы.
        if not self.calibrated:
            raise Exception("Calibration have not been executed!")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray,
                                              self.calibration_info.aruco_dict,
                                              parameters=self.parameters)

        frame_markers = None
        if out_frame:
            # Функция выделяет на картинке найденные квадраты
            frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners,
                                                      ids)

        mtx = self.calibration_info.mtx_for_img(frame)
        dist = self.calibration_info.dist
        rvecs, tvecs, _ = aruco.\
            estimatePoseSingleMarkers(corners, self.real_sqr_size, mtx, dist)

        if tvecs is not None:
            if draw_axes:
                frame_markers = aruco.drawAxis(
                    frame_markers if out_frame else frame,
                    mtx, dist, rvecs, tvecs, length=15)

            coords = np.ndarray((self.marks_num, self.AXES_COUNT),
                                dtype=np.float)
            for _id, xyz in zip(ids[:, 0], tvecs):
                coords[_id] = xyz
            return coords.flatten(), frame_markers
        else:
            return np.asarray(
                [np.NaN] * (self.marks_num * self.AXES_COUNT)), None

    def set_calibration_info(self, calib_info: CalibrationInfo):
        self.calibration_info = calib_info

    def get_report(self, video_path: str,
                   sec_to_cut: Tuple[float, float] = (1, 1.5),
                   out_video: bool = False,
                   out_video_dir: str = f'.{OS_SLASH}Размеченные видео',
                   draw_axes: bool = False) -> pd.DataFrame:

        video_reader = CaptureIterator(video_path)
        frame_count = len(video_reader)

        if out_video:
            out = self.create_out_video(video_path, out_video_dir)

        report = np.ndarray(
            (len(video_reader), len(self.all_ids) * self.AXES_COUNT))
        time_col = np.ndarray(shape=(frame_count,))

        for frame_num, cur_time, frame in enumerate(CaptureIterator(video_path)):
            time_col[frame_num] = cur_time
            points, marked_frame = self.get_xyz(frame, out_frame=out_video,
                                                draw_axes=draw_axes)
            report[frame_num] = points

            if out_video:
                out.write(frame if all(np.isnan(report[-1])) else marked_frame)

        L_time_edge, R_time_edge = sec_to_cut[0], time_col[-1] - sec_to_cut[1]

        if R_time_edge <= L_time_edge:
            raise Exception('There are no so much time to cut!\n'
                            f'Total seconds: {time_col[-1]}'
                            f'Seconds to cut from in beginning: {sec_to_cut[0]}'
                            f'Seconds to cut from ending: {sec_to_cut[1]}')

        bool_time_cutter = np.logical_and(time_col > L_time_edge,
                                          time_col < R_time_edge)
        report = report[bool_time_cutter, :]
        time_col = time_col[bool_time_cutter]

        # Обрезаем записи без значений (NaN) в начале и в конце массива
        # т.к. они не будут участвовать в расчетах и не несут полезной информации
        # report.
        report_isnan = np.isnan(report)
        nan_strs = np.all(report_isnan, axis=1)
        no_nan_strs = np.all(np.logical_not(report_isnan), axis=1)
        try:
            L, R = np.where(no_nan_strs, True), len(nan_strs) - np.where(
                nan_strs[::-1], False)
        except Exception as ex:
            print(video_path, " - ", ex)
            return None, None

        report = report[L:R]
        time_col = time_col[L:R]

        axes_names = ['X', 'Y', 'Z']
        df_report = pd.DataFrame(index=time_col,
                                 data=report,
                                 columns=pd.MultiIndex.from_product(
                                     [self.all_ids, axes_names]))
        df_report.index.name = 'Время, мс'
        df_report.columns.names = ['№ метки', 'Ось']

        return df_report

    def add_derivative_cols(self, df_report: pd.DataFrame) -> pd.DataFrame:
        def series_derivative(col, min_span_size):
            spans = self.get_clear_spans(col, min_span_size=30)
            dx = np.empty(len(col), dtype=np.float64)

            for s in spans:
                dx[s[0]:s[1]] = np.gradient(col.iloc[s[0]:s[1]],
                                            col.index[s[0]:s[1]])
            return pd.Series(dx, index=col.index)

        for _id in self.all_ids:
            df_report[(_id, 'dX')] = series_derivative(df_report[(_id, 'X')],
                                                       min_span_size=30)
            df_report[(_id, 'dY')] = series_derivative(df_report[(_id, 'Y')],
                                                       min_span_size=30)

        df_report = df_report.reindex(sorted(df_report.columns), axis=1)
        return df_report

    def draw_separate(self, df_report: pd.DataFrame,
                      file_name: str,
                      path_to_save: str = f'.{OS_SLASH}graphs'):
        if df_report.empty:
            return

        fig, plots = plt.subplots(1, len(self.all_ids),
                                  figsize=(len(self.all_ids) * 6, 6))

        for i, _id in enumerate(self.all_ids[::-1]):
            plots[i].plot(df_report[(_id, 'X')], df_report[(_id, 'Y')])
            plots[i].title.set_text('Метка #%d' % (_id))
            plots[i].invert_yaxis()
            plots[i].xaxis.tick_top()
            plots[i].set_ylabel('mm')
            plots[i].set_xlabel('mm')

        FilesHelper.mk_dir_if_not_exists(path_to_save)
        plt.savefig(f'%s{OS_SLASH}%s_раздельно.jpg' % (path_to_save, file_name),
                    dpi=250)

        plt.close(fig)
        plt.cla()
        plt.clf()

    def draw_together(self, raw_report: pd.DataFrame,
                      file_name: str,
                      path_to_save: str = f'.{OS_SLASH}graphs'):
        if raw_report.empty:
            return

        plt.gca().invert_yaxis()
        plt.gca().xaxis.tick_top()
        for i, _id in enumerate(self.all_ids):
            plt.plot(raw_report[(_id, 'X')], raw_report[(_id, 'Y')])
        plt.legend(["Метка #%d" % _id for _id in self.all_ids])

        FilesHelper.mk_dir_if_not_exists(path_to_save)
        plt.savefig(f'%s{OS_SLASH}%s_вместе.jpg' % (path_to_save, file_name),
                    dpi=400)
        plt.cla()
        plt.clf()

    def get_errors_finder(self, frames_per_step: int,
                          step_mm: float,
                          text_for_log: str) -> Callable:
        def get_errors(check_img_paths: np.ndarray):
            print(f"{text_for_log}. Started.\n")
            is_folder = len(check_img_paths) != 0 and OS_SLASH in \
                        check_img_paths[0]
            if is_folder:
                print(f'(Folder: {check_img_paths[0].split(OS_SLASH)[-2]})')

                path_to_save, _ = check_img_paths[0].rsplit(OS_SLASH, 1)
                path_to_save += f'{OS_SLASH}Marked'
                FilesHelper.mk_dir_if_not_exists(path_to_save)

            positions = np.ndarray(
                shape=(len(check_img_paths), self.AXES_COUNT))

            for img_num, image_path in enumerate(check_img_paths):
                frame = cv2.imread(image_path)
                coord, marked_frame = self.get_xyz(frame, out_frame=True,
                                                   draw_axes=True)
                if is_folder:
                    _, file_name = image_path.rsplit(OS_SLASH, 1)
                    ret = cv2.imwrite(f'{path_to_save}{OS_SLASH}{file_name}',
                                      marked_frame)
                positions[img_num] = coord
            positions = np.asarray(positions)
            positions -= positions[0]

            length = np.asarray(
                [np.linalg.norm(position) for position in positions])
            r = length - np.asarray(
                [i * step_mm for i in range(len(length) // frames_per_step) for
                 j in range(frames_per_step)])

            print(f"{text_for_log}. Finished.\n")
            if len(check_img_paths) != 0 and OS_SLASH in check_img_paths[0]:
                print(f'(Folder: {check_img_paths[0].split(OS_SLASH)[-2]})')

            return r

        return get_errors

    # region Private
    @staticmethod
    def create_out_video(orig_file_path, path_to_save):
        cap = cv2.VideoCapture(orig_file_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Открываем поток записи в видеофайл
        FilesHelper.mk_dir_if_not_exists(path_to_save)

        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        fn, ext = orig_file_path.rsplit(OS_SLASH, 1)[-1].rsplit('.', 1)
        out_video_name = f"{path_to_save}{OS_SLASH}{fn} (Размеченный).{ext}"

        out_video = cv2.VideoWriter(f"{path_to_save}{OS_SLASH}{out_video_name}",
                                    fourcc, cap.get(cv2.CAP_PROP_FPS),
                                    (width, height))
        cap.release()
        return out_video



    @staticmethod
    def get_clear_spans(arr, min_span_size=30):
        spans = []
        isSpan = False
        start = None
        i = 0
        while i < len(arr):
            if np.isnan(arr.iloc[i]) and isSpan:
                if i - start >= min_span_size:  # Если последовательность достаточно длинная
                    spans.append((start, i))
                start = None
                isSpan = False
            elif not np.isnan(arr.iloc[i]) and not isSpan:
                isSpan = True
                start = i
            i += 1
        if start is not None and i - start >= min_span_size:
            spans.append((start, i))
        return spans

    # endregion


class CaptureIterator:
    def __init__(self, fname: str):
        self.cap = cv2.VideoCapture(fname)
        if not self.cap.isOpened():
            raise FileNotFoundError()

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[float, np.ndarray]:
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return self.cap.get(cv2.CAP_PROP_POS_MSEC), frame
            else:
                self.cap.release()
        raise StopIteration

    def __len__(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
