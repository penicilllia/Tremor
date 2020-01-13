# This Python file uses the following encoding: utf-8

import sys
import os
sys.path.append(os.path.abspath('../tremlib'))

import cv2
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PIL.ImageQt import ImageQt
from PIL.Image import Image
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaPlayer, QMediaPlaylist, QMediaContent

from tremlib.trem_researcher import TremResearcher
from file_helper import FilesHelper, OS_SLASH
from calibration_info import CalibrationInfo

import widgets_ext
from mainwindow import Ui_MainWindow

ArucDict_dict = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}
ArucDict_dict = {key: cv2.aruco.getPredefinedDictionary(value)
                 for key, value in ArucDict_dict.items()}


class MainWindow(QMainWindow):
    markers_img: Image = None
    board_img: Image = None

    calibration_photo_files: np.ndarray = None
    calibration_info = property()
    _calib_info: CalibrationInfo = None

    video_files: np.ndarray = None
    player = None
    playlist = None

    video_position = pyqtSignal(int)

    def __init__(self, parent=None, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi('./mainwindow.ui', self)
        self.setup_ui(self)
        self.connect_events()

    def setup_ui(self, window):
        self.gen_markers_arucoDict.addItems(ArucDict_dict.keys())
        self.gen_markers_arucoDict.setCurrentIndex(0)

        self.gen_board_arucoDict.addItems(ArucDict_dict.keys())
        self.gen_board_arucoDict.setCurrentIndex(0)

        self.calibrate_aruco_dict.addItems(ArucDict_dict.keys())
        self.calibrate_aruco_dict.setCurrentIndex(0)

        self.gen_marker_Gview = widgets_ext.MyPicBox(self.gen_marker_Gview)
        self.gen_board_Gview = widgets_ext.MyPicBox(self.gen_board_Gview)
        self.calibrate_Gview = widgets_ext.MyPicBox(self.calibrate_Gview)

        self.calibrate_photo_list.setSelectionMode(
            QAbstractItemView.SingleSelection)

        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player.setVideoOutput(self.video)
        self.player.mediaStatusChanged.connect(self.print_status)
        # self.playlist = QMediaPlaylist(self.player)
        # self.player.setPlaylist(self.playlist)

    def connect_events(self):
        self.gen_markers_run_btn.clicked.connect(self.gen_markers_and_show)
        self.gen_markers_save_btn.clicked.connect(self.gen_markers_save)

        self.gen_board_run_btn.clicked.connect(self.gen_board_and_show)
        self.gen_board_save_btn.clicked.connect(self.gen_board_save)

        self.calibrate_load_photo.clicked.connect(self.load_photo_dir)
        self.calibrate_photo_list.itemSelectionChanged.connect(
            self.on_calibration_photo_selected
        )

        self.calibrate_start_btn.clicked.connect(self.calibrate_start)
        self.calibrate_load_settings.clicked.connect(self.load_settings)
        self.calibrate_save_settings.clicked.connect(self.save_settings)

        self.video_tracking_load_video_bnt.clicked.connect(
            self.load_videos_dir
        )
        self.video_tracking_list.itemSelectionChanged.connect(
            self.on_video_file_selected
        )

        self.play_btn.clicked.connect(self.player.play)
        self.pause_btn.clicked.connect(self.player.pause)
        self.stop_btn.clicked.connect(self.player.stop)

        # self.video_position.connect(self.player.position)
        # self.video_position.connect(self.video_slider.setValue)



    @calibration_info.getter
    def calibration_info(self):
        return self._calib_info

    @calibration_info.setter
    def calibration_info(self, value: CalibrationInfo):
        self._calib_info = value
        if value is None:
            return

        self.calibration_status.setText("Да")
        r = self._calib_info.calib_photos_ratio
        r1, r2 = r.numerator, r.denominator
        self.calibration_ratio.setText(f"{r1}:{r2}")
        d = self._calib_info.aruco_dict

        found = False
        for name, dict in ArucDict_dict.items():
            if dict == d:
                self.calibration_dictionary.setText(name)
                found = True
                break

        if not found and d is None:
            msg = "В калибровочной информации отсутствует словрь меток!"
            mbox = QMessageBox()
            mbox.setWindowTitle('Ошибка!')
            mbox.setText(msg)
            mbox.exec()
            self.calibration_info = None

    def gen_markers_and_show(self):
        key = self.gen_markers_arucoDict.currentText()
        aruco_dict = ArucDict_dict[key]

        markers_num = self.gen_markers_num.value()
        markers_size = self.gen_markers_size.value()
        print_dpi = self.gen_markers_dpi.value()

        try:
            img = TremResearcher.gen_marks(
                aruco_dict=aruco_dict,
                markers_num=markers_num,
                real_size_mm=markers_size,
                print_dpi=print_dpi
            )
            self.markers_img = img
        except Exception as ex:
            mbox = QMessageBox()
            mbox.setWindowTitle('Ошибка!')
            mbox.setText(str(ex))
            mbox.exec()
            return

        MainWindow.show_img_on_view(img, self.gen_marker_Gview)

    def gen_markers_save(self):
        if self.markers_img is None:
            mbox = QMessageBox()
            mbox.setWindowTitle('Ошибка!')
            mbox.setText("Маркеры еще не сгенирированы.")
            mbox.exec()
            return

        file_name, _ = QFileDialog.getSaveFileName(self, 'Сохранить',
                                                   filter='png(*.png)')
        if file_name is not None and file_name != "":
            self.markers_img.save(file_name)

    def gen_board_and_show(self):
        key = self.gen_board_arucoDict.currentText()
        aruco_dict = ArucDict_dict[key]
        resolution = self.gen_board_scr_w.value(), self.gen_board_scr_h.value()

        try:
            img = TremResearcher.gen_board(
                aruco_dict=aruco_dict,
                screen_resolution=resolution
            )
            self.board_img = img
        except Exception as ex:
            mbox = QMessageBox()
            mbox.setWindowTitle('Ошибка!')
            mbox.setText(str(ex))
            mbox.exec()
            return

        MainWindow.show_img_on_view(img, self.gen_board_Gview)

    def gen_board_save(self):
        if self.board_img is None:
            mbox = QMessageBox()
            mbox.setWindowTitle('Ошибка!')
            mbox.setText("Маркеры еще не сгенирированы.")
            mbox.exec()
            return

        file_name, _ = QFileDialog.getSaveFileName(self, 'Сохранить',
                                                   filter='png(*.png)')
        if file_name is not None and file_name != "":
            self.board_img.save(file_name)

    def load_photo_dir(self):
        dialog = QFileDialog()
        dialog.setWindowTitle('Выберете папку с калибровочными фотографиями')
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setFileMode(QFileDialog.Directory)

        if dialog.exec_() == QFileDialog.Accepted:
            photo_dir: str = dialog.selectedFiles()[0]
            files = FilesHelper.photo_files(photo_dir)
            self.calibration_photo_files = files
            self.calibrate_photo_list.clear()
            self.calibrate_photo_list.addItems(
                [f.rsplit(OS_SLASH, 1)[1] for f in files])

    def on_calibration_photo_selected(self):
        inx: QModelIndex = self.calibrate_photo_list.currentRow()
        file = self.calibration_photo_files[inx]
        pix = QPixmap(file)
        self.calibrate_Gview.setPixmap(pix)

    def calibrate_start(self):
        key = self.calibrate_aruco_dict.currentText()
        aruco_dict = ArucDict_dict[key]
        resolution = self.calibrate_screen_width.value(),\
                     self.calibrate_screem_height.value()
        screen_inches = self.calibrate_screen_inches.value()
        photos = self.calibration_photo_files
        try:
            calib = TremResearcher.calibrate_by_monitor_photo(aruco_dict,
                                                      photos,
                                                      resolution,
                                                      screen_inches)
            self.calibration_info = calib
        except Exception as ex:
            mbox = QMessageBox()
            mbox.setWindowTitle('Ошибка!')
            mbox.setText(str(ex))
            mbox.exec()
        else:
            mbox = QMessageBox()
            mbox.setWindowTitle('Сообщение')
            mbox.setText('Калибровка завершена.')
            mbox.exec()

    def load_settings(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Открыть',
                                                   filter='pkl(*.pkl)')
        if file_name is not None and file_name != "":
            calib = CalibrationInfo.load_calibration_info(file_name)
            self.calibration_info = calib

    def save_settings(self):
        file_name, _ = QFileDialog.getSaveFileName(self, 'Сохранить',
                                                   filter='pkl(*.pkl)')
        if file_name is not None and file_name != "":
            self.calibration_info.save(file_name)

    @staticmethod
    def show_img_on_view(img: Image, view: widgets_ext.MyPicBox):
        p1 = ImageQt(img)
        p2 = QImage(p1)
        p3 = QPixmap(p2)
        view.setPixmap(QPixmap(p3))

    def load_videos_dir(self):
        dialog = QFileDialog()
        dialog.setWindowTitle('Выберете папку с видеофайлами')
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setFileMode(QFileDialog.Directory)

        if dialog.exec_() == QFileDialog.Accepted:
            photo_dir: str = dialog.selectedFiles()[0]
            files = FilesHelper.video_files(photo_dir)
            self.video_files = files
            self.video_tracking_list.clear()
            self.video_tracking_list.addItems(
                [f.rsplit(OS_SLASH, 1)[1] for f in files])

    def on_video_file_selected(self):
        inx: QModelIndex = self.video_tracking_list.currentRow()
        file = self.video_files[inx]

        self.player.setMedia(QMediaContent(QUrl("https://www.youtube.com/watch?v=7cQ5n9j5Guo")))
        #self.player.setMedia(QMediaContent(QUrl.fromLocalFile(file)))

    def print_status(self):
        print("Status changed to:", self.player.mediaStatus())
        self.video_slider.setMaximum(self.player.duration())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()

    window.show()
    sys.exit(app.exec_())
