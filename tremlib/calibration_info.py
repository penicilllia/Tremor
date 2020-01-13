import numpy as np
import cv2
from cv2 import aruco
from fractions import Fraction
from typing import Tuple
import pickle

from file_helper import OS_SLASH, FilesHelper

MM_IN_INCH = 25.4


class CalibrationInfo:
    """
    Класс для хранения информации о калибровке камеры
    """

    SUPPORTED_RATIOS_TO_SQUARES_COUNT = {
        (4, 3): (11, 8),
        (16, 9): (12, 7),
        (16, 10): (11, 7),
        (21, 9): (14, 6)
    }

    @staticmethod
    def get_squares_for_screen_resolution(screen_resolution: Tuple[int, int]):
        """
        Получение количества квадратов по горизонтали и вертикали для построения
        доски ArUco по данному разрешению экрана. Если разрешение экрана не
        поддерживается, возвращется исключение.
        :param screen_resolution: Разрешение экрана в виде
         кортежа (width px, heght px)
        :return:
        """
        dict_ratio_squares = {
            Fraction(*key): value for key, value in
            CalibrationInfo.SUPPORTED_RATIOS_TO_SQUARES_COUNT.items()}

        fract = Fraction(screen_resolution[0], screen_resolution[1])

        if fract in dict_ratio_squares.keys():
            return dict_ratio_squares[fract]
        else:
            s = ""
            for x in CalibrationInfo.SUPPORTED_RATIOS_TO_SQUARES_COUN.keys():
                s += f"{x[0]}:{x[1]}, "
            raise Exception(f"Only {s[:-2]} screen supported in this version.")

    def __init__(self, aruco_dict: cv2.aruco_Dictionary,
                 calib_photos_ratio: Fraction,
                 screen_resolution: Tuple[int, int],
                 screen_inches: float,
                 mtx: np.ndarray = None,
                 dist: np.ndarray = None):
        """
        :param aruco_dict: набор меток ArUco
        :param calib_photos_ratio: соотношение сторон калибровочных фотографий
        :param screen_resolution: разрешение фотографируемого экрана/монитора
        :param screen_inches: размер фотографируемого экрана/монитора в дюймах
        :param mtx: нормализованная матрица калибровки
        :param dist: вектор искажений
        """

        # Check supporting screen resolution
        squares_by_x, squares_by_y = \
            CalibrationInfo.get_squares_for_screen_resolution(screen_resolution)

        self.aruco_dict = aruco_dict
        self.calib_photos_ratio = calib_photos_ratio

        # Берем минимельное соотношение стороный изображения к количеству меток
        # по данной стороне. Минимум берется, т.к. при не полном заполнении
        # метками изображение дополняется белыми полями.
        marker_length_px = min(screen_resolution[0] / squares_by_y,
                               screen_resolution[1] / squares_by_x)

        dpi = (screen_resolution[0]**2 + screen_resolution[1]**2) ** 0.5
        dpi /= screen_inches
        # пикселей в метке / DPI * милиметров в дюйме
        read_board_marker_size = marker_length_px / dpi * MM_IN_INCH

        self.calib_board = aruco.CharucoBoard_create(
            squaresX=squares_by_x,
            squaresY=squares_by_y,
            squareLength=read_board_marker_size*2,
            markerLength=read_board_marker_size,
            dictionary=aruco_dict)

        self.set_calibration_mtx_dist(mtx, dist)

    def set_calibration_mtx_dist(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist

    def mtx_for_img(self, img: np.ndarray) -> np.ndarray:
        '''
        Создает калибровачную матрицу, адаптированную для
        изображения с данным разрешением
        :param img: изображение (ndarray)
        :return: калибровачная матрица 3x3 (ndarray)
        '''
        h, w, _ = img.shape
        img_ratio = Fraction(h, w)

        if img_ratio != self.calib_photos_ratio:
            raise Exception(f"""Cannot generate calibration matrix 
            for images with ratio {img_ratio.numerator} : {img_ratio.denominator}.
            Ratio of calibration set is {self.calib_photos_ratio.numerator} : 
            {self.calib_photos_ratio.denominator}.""")
        else:
            f = h / img_ratio.numerator
            return self.mtx * np.asarray([
                [f, 1, f],
                [f, f, 1],
                [1, 1, 1]
            ])

    def save(self, file_path: str):
        if OS_SLASH in file_path:
            path, file_name = file_path.rsplit(OS_SLASH, maxsplit=1)
            FilesHelper.mk_dir_if_not_exists(path)

        with open(file_path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_calibration_info(file_path: str) -> 'CalibrationInfo':
        try:
            with open(file_path, 'rb') as f:
                calibration_info = pickle.load(f)
        except Exception as ex:
            print(ex)
            raise ex

        return calibration_info
