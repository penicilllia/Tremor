import numpy as np
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
from fractions import Fraction
from typing import Tuple
from PIL import Image
import io

from calibration_info import CalibrationInfo, MM_IN_INCH

class ArucoHelper:
    """
    Класс помощник, в котором собраны статические методы по работе с
    метками и досками ArUco, а так же функционал калибровки камеры по
    доскам ArUco.
    """

    '''Параметры поиска метки на фотографии'''
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,  # флаги для поиска
        100,  # кол-во итераций алгоритма поиска
        0.0001)  # точность поиска координат метки

    '''параметры поиска метки на фотографии'''
    parameters = aruco.DetectorParameters_create()

    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    @staticmethod
    def gen_board(aruco_dict: cv2.aruco_Dictionary,
                  screen_resolution: Tuple[int, int] = (2560, 1440)) -> Image:
        """
        Генерирует картинку с доской ArUcoBoard для показа на экране.
        :param aruco_dict: набор меток ArUco.
        :param screen_resolution: разрешение экрана в пикселях
        :return: изображение с сгенерированной доской для данного разрешения.
        """
        squares_by_x, squares_by_y = \
            CalibrationInfo.get_squares_for_screen_resolution(screen_resolution)

        board = aruco.CharucoBoard_create(squaresX=squares_by_x,
                                          squaresY=squares_by_y,
                                          squareLength=2,
                                          markerLength=1,
                                          dictionary=aruco_dict)
        img = board.draw(screen_resolution)
        return Image.fromarray(img)

    @staticmethod
    def gen_marks(aruco_dict: cv2.aruco_Dictionary,
                  markers_num: int = 5,
                  real_size_mm: float = 10,
                  print_dpi: int = 300,
                  page_size=(11.69, 8.27)) -> Image.Image:
        """
        Генерирует картинку с маркерами для дальнейшей печати.
        :param aruco_dict: набор меток ArUco.
        :param markers_num: количество меток.
        :param real_size_mm: размер метки в милиметрах.
        :param print_dpi: DPI печатаемой картинки.
        :param page_size: размеры картинки в дюймах, по умолчанию размер А4.
        :return: изображение подсписанных меток для печати.
        """
        fig = plt.figure(figsize=page_size, dpi=print_dpi)

        x_max = int(page_size[0] * print_dpi)
        y_max = int(page_size[1] * print_dpi)

        marker_size_px = int(real_size_mm / MM_IN_INCH * print_dpi)

        space_between_markers = int(7 / 20 * print_dpi)
        page_pad = int(print_dpi / 3)
        page_unit = marker_size_px + space_between_markers
        col_size = (y_max - page_pad * 2) // page_unit
        row_size = (x_max - page_pad * 2) // page_unit

        if col_size * row_size < markers_num:
            raise Exception(f"Cannot draw {markers_num} "
                            f"markers with size: {real_size_mm}mm\n"
                            f"Max marker count: {col_size * row_size}.")

        for x in range(row_size):
            for y in range(col_size):
                cur_marker_num = x * col_size + y
                if cur_marker_num + 1 > markers_num:
                    break

                ax = fig.add_subplot(col_size, row_size, cur_marker_num + 1)
                img = aruco.drawMarker(dictionary=aruco_dict,
                                       id=cur_marker_num,
                                       sidePixels=marker_size_px)
                x_offset = page_pad + page_unit * x
                y_offset = y_max - page_unit * (y + 1)
                plt.figtext(x_offset / x_max,
                            (y_offset - space_between_markers * 0.60) / y_max,
                            f'Метка {cur_marker_num}',
                            fontdict={'family': 'serif',
                                      'color': 'black',
                                      'weight': 'normal',
                                      'size': 6})

                plt.figimage(img,
                             cmap=mpl.cm.gray,
                             xo=x_offset,
                             yo=y_offset)
                plt.grid(True)
                ax.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        im = Image.open(buf)
        return im


    @staticmethod
    def get_images_resolution(images_paths: np.ndarray) -> Tuple[int, int]:
        """
        Получить общее для всех фотографий разрешение. Если в массиве есть путь
        фотографии, с отличным от предыдущих разрешений, выбрасывается
        исключение.
        :param images_paths: массив путей к фотографиям.
        :return: кортеж вида (width px, height px).
        """
        resolution = None
        if images_paths is None or images_paths.size == 0:
            raise Exception("Calibration images are not given.")

        for image_path in images_paths:
            try:
                img = cv2.imread(image_path)
                if 0 in img.shape:
                    raise Exception("Cannot use empty image")
            except Exception as ex:
                print(f"Cannot load file {image_path}")
                raise ex

            h, w, _ = img.shape

            if resolution is None:
                resolution = (w, h)
            elif resolution != (w, h):
                raise Exception(f"Callibration image {image_path}\n"
                                f" has different resolution to other images.")

        return resolution

    @staticmethod
    def calibrate_by_monitor_photo(
            aruco_dict: cv2.aruco_Dictionary,
            images_paths: np.ndarray,
            screen_resolution: Tuple[int, int] = (2560, 1440),
            screen_inches: float = 27) -> CalibrationInfo:
        """
        Находит параметры калибровки по данным фотографиям монитора
        с изображением доски ArUcoBoard.
        :param aruco_dict: набор меток из которых сделан
         фотографируемый ArUcoBoard.
        :param images_paths: массив путей к фотографиям.
        :param screen_resolution: разрешение фотографируемого монитора.
        :param screen_inches: диагональ фотографируемого монитора в дюймах.
        :return: калобровочная информация в объекте CalibrationInfo.
        """
        images_resolution = ArucoHelper.get_images_resolution(images_paths)
        ratio = Fraction(*images_resolution)
        calibration_info = CalibrationInfo(aruco_dict, ratio, screen_resolution,
                                           screen_inches)
        ret, mtx, dist, _, _ = \
            ArucoHelper.calibrate_camera(aruco_dict, calibration_info,
                                            images_paths)

        # множитель разрешения калибровочной картинки
        f = images_resolution[0] / ratio.numerator
        mtx_norm = mtx
        # нормализуем калибровочную матрицу,
        # т.к. полученная матрица адаптирована
        # для конкретного разрешения
        mtx_norm /= np.asarray([
            [f, 1, f],
            [f, f, 1],
            [1, 1, 1]
        ])

        calibration_info.set_calibration_mtx_dist(mtx_norm, dist)
        return calibration_info

    @staticmethod
    def read_chessboards(aruco_dict: cv2.aruco_Dictionary,
                         calibration_info: CalibrationInfo,
                         images: np.ndarray):
        """
        Charuco base pose estimation.
        """
        print("POSE ESTIMATION STARTS:")
        all_corners, all_ids = [], []

        criteria = ArucoHelper.criteria

        for img_num, im in enumerate(images):
            print("%03d/%03d BOARDS RECOGNIZED" % (img_num, len(images)))
            gray = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

            if len(corners) == 0:
                continue
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner, winSize=(3, 3),
                                 zeroZone=(-1, -1), criteria=criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, calibration_info.calib_board)
            if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3:
                all_corners.append(res2[1])
                all_ids.append(res2[2])

        print("read_chessboards DONE")
        imsize = gray.shape
        return all_corners, all_ids, imsize

        # private

    @staticmethod
    def calibrate_camera(aruco_dict: cv2.aruco_Dictionary,
                         calibration_info: CalibrationInfo,
                         images: np.ndarray):
        """
        Calibrates the camera using the detected corners.
        """
        allCorners, allIds, imsize = ArucoHelper.read_chessboards(
            aruco_dict, calibration_info, images)

        print("CAMERA CALIBRATION ...", end=' ')

        cameraMatrixInit = np.array([[1000., 0., imsize[0] / 2.],
                                     [0., 1000., imsize[1] / 2.],
                                     [0., 0., 1.]], dtype=np.float64)

        dist_coeffs_init = np.zeros((8, 1), dtype=np.float64)  # (5, 1)
        flags = cv2.CALIB_RATIONAL_MODEL
        (ret, camera_matrix, distortion_coefficients0,
         rotation_vectors, translation_vectors,
         stdDeviationsIntrinsics, stdDeviationsExtrinsics,
         perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
            charucoCorners=allCorners,
            charucoIds=allIds,
            board=calibration_info.calib_board,
            imageSize=imsize,
            cameraMatrix=cameraMatrixInit,
            distCoeffs=dist_coeffs_init,
            flags=flags,
            criteria=(
                cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
        print("DONE")

        return ret, camera_matrix, distortion_coefficients0, \
               rotation_vectors, translation_vectors