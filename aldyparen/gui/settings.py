import os

from PyQt5.QtCore import QSettings


class AldyparenSettings:
    def __init__(self, app: 'AldyparenApp'):
        self.app = app
        self.qsettings = QSettings("aldyparen", "aldyparen-py")

        def _load_int(name):
            val = self.qsettings.value(name)
            return int(val) if val is not None else None

        self.work_dir = self.qsettings.value("work_dir") or os.getcwd()

        # Settings not settable in UI.
        self.work_view_width = _load_int("work_view_width") or 480
        self.work_view_height = _load_int("work_view_height") or 270
        self.movie_view_width = _load_int("movie_view_width") or 240
        self.movie_view_height = _load_int("movie_view_height") or 135

        # Settings settable in UI.
        self.app.main_window.spin_box_image_resolution_1.setValue(_load_int("image_width") or 3840)
        self.app.main_window.spin_box_image_resolution_2.setValue(_load_int("image_height") or 2160)
        self.app.main_window.spin_box_video_resolution_1.setValue(_load_int("video_width") or 2560)
        self.app.main_window.spin_box_video_resolution_2.setValue(_load_int("video_height") or 1440)
        self.app.main_window.spin_box_fps.setValue(_load_int("video_fps") or 16)
        self.app.main_window.spin_box_downsampling.setValue(_load_int("downsample_factor") or 1)
        self.app.main_window.spin_box_animation_length.setValue(_load_int("animation_length") or 16)

    def get_downsample_factor(self):
        return self.app.main_window.spin_box_downsampling.value()

    def get_video_fps(self):
        return self.app.main_window.spin_box_fps.value()

    def save(self):
        self.qsettings.setValue("work_dir", self.work_dir)
        self.qsettings.setValue("work_view_width", self.work_view_width)
        self.qsettings.setValue("work_view_height", self.work_view_height)
        self.qsettings.setValue("movie_view_width", self.movie_view_width)
        self.qsettings.setValue("movie_view_height", self.movie_view_height)
        self.qsettings.setValue("image_width", self.app.main_window.spin_box_image_resolution_1.value())
        self.qsettings.setValue("image_height", self.app.main_window.spin_box_image_resolution_2.value())
        self.qsettings.setValue("video_width", self.app.main_window.spin_box_video_resolution_1.value())
        self.qsettings.setValue("video_height", self.app.main_window.spin_box_video_resolution_2.value())
        self.qsettings.setValue("video_fps", self.app.main_window.spin_box_fps.value())
        self.qsettings.setValue("downsample_factor", self.app.main_window.spin_box_downsampling.value())
        self.qsettings.setValue("animation_length", self.app.main_window.spin_box_animation_length.value())
