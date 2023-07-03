import numpy as np
from PyQt5 import QtWidgets, QtGui, uic, QtCore
from PyQt5.QtCore import QPointF, QCoreApplication
from PyQt5.QtWidgets import QMessageBox, QGraphicsSceneWheelEvent, QGraphicsSceneMouseEvent, QApplication, QComboBox, \
    QPlainTextEdit, QLabel, QSpinBox, QScrollBar
from typing import Union

from .. import ColorPalette
from ..painters import ALL_PAINTERS


def show_alert(text, title=""):
    alert = QMessageBox()
    alert.setWindowTitle(title)
    alert.setText(text)
    alert.exec()


class WorkFrameScene(QtWidgets.QGraphicsScene):
    def __init__(self, parent, app: 'AldyparenApp'):
        super().__init__(parent)
        self.app = app
        self.is_dragging = False
        self.drag_start_x = 0.0
        self.drag_start_y = 0.0
        self.frame_width_pxl = 0
        self.cursor_math_pos = None  # type: Union[None, np.complex128]

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        self.calculate_cursor_math_pos(event.scenePos())
        if self.is_dragging:
            if self.cursor_math_pos is None:
                self.is_dragging = False
            else:
                cur_pos = event.scenePos()
                x = cur_pos.x()
                y = cur_pos.y()
                self.apply_drag(x - self.drag_start_x, y - self.drag_start_y)
                self.drag_start_x = x
                self.drag_start_y = y

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        self.is_dragging = True
        self.drag_start_x = event.scenePos().x()
        self.drag_start_y = event.scenePos().y()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        self.is_dragging = False

    def wheelEvent(self, event: QGraphicsSceneWheelEvent):
        modifiers = QApplication.keyboardModifiers()
        delta = -event.delta() / 120
        if bool(modifiers & QtCore.Qt.ShiftModifier):
            delta *= 20

        self.calculate_cursor_math_pos(event.scenePos())
        if self.cursor_math_pos is None:
            return
        if bool(modifiers & QtCore.Qt.ControlModifier):
            # 2 degrees minimal increment (for standard mouse).
            angle = delta * (np.pi / 90)
            self.app.update_work_frame_transform(
                self.app.work_frame.transform.rotate_at_point(self.cursor_math_pos, angle))
        else:
            self.app.update_work_frame_transform(
                self.app.work_frame.transform.scale_at_point(self.cursor_math_pos, 1.05 ** delta))

    def apply_drag(self, dx_pxl, dy_pxl):
        upsp = self.units_per_screen_pixel()
        self.app.update_work_frame_transform(
            self.app.work_frame.transform.translate(dx_pxl * upsp, -dy_pxl * upsp))

    def units_per_screen_pixel(self):
        return self.app.work_frame.transform.scale / self.frame_width_pxl

    def calculate_cursor_math_pos(self, pos: QPointF):
        x = pos.x()
        y = pos.y()
        if x <= 0 or x >= self.width() or y <= 0 or y >= self.height():
            self.cursor_math_pos = None
        else:
            upsp = self.units_per_screen_pixel()
            x = x - 0.5 * self.width()
            y = -(y - 0.5 * self.height())
            self.cursor_math_pos = self.app.work_frame.transform.center + \
                                   np.complex128(x + 1j * y) * upsp
        return self.cursor_math_pos


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, app: 'AldyparenApp'):
        super(MainWindow, self).__init__()
        self.app = app
        uic.loadUi('layout/main.xml', self)

        self.setMouseTracking(True)

        # Initialize painter list.
        combo = self.combo_painter_type  # type: QComboBox
        for painter_class in ALL_PAINTERS:
            combo.addItem(painter_class.__name__)
        combo.activated.connect(lambda idx: app.select_painter_type(idx))

        # Initialize palette list.
        combo = self.combo_palette_type  # type: QComboBox
        combo.addItem("Grayscale")
        combo.addItem("Random")
        combo.addItem("Gradient")
        combo.addItem("Gradient+Black")

        self.edit_painter_config.textChanged.connect(
            lambda: self.on_config_text_changed())

        # Buttons.
        self.button_reset_transform.clicked.connect(
            lambda: self.app.reset_transform())
        self.button_reset_painter_config.clicked.connect(
            lambda: self.confirm_then("Reset painter config?", self.app.reset_config))
        self.button_generate_palette.clicked.connect(self.on_generate_palette_click)
        self.button_force_update.clicked.connect(self.on_force_update_clicked)
        self.button_export_image.clicked.connect(self.app.export_image)
        self.button_make_animation.clicked.connect(self.make_animation)

        # Menu items.
        self.menu_new_project.triggered.connect(lambda: show_alert(""))
        self.menu_open_project.triggered.connect(lambda: show_alert(""))
        self.menu_save_project.triggered.connect(lambda: show_alert(""))
        self.menu_save_project_as.triggered.connect(lambda: show_alert(""))
        self.menu_render_photo.triggered.connect(lambda: show_alert(""))
        self.menu_render_video.triggered.connect(lambda: show_alert(""))
        self.menu_settings.triggered.connect(lambda: show_alert(""))
        self.menu_exit.triggered.connect(self.on_exit)
        self.menu_video_clear.triggered.connect(self.clear_movie)
        self.menu_video_append.triggered.connect(self.app.append_movie_frame)
        self.menu_video_replace.triggered.connect(self.app.replace_movie_frame)
        self.menu_video_remove_last_frame.triggered.connect(lambda: self.app.remove_last_frames(1))
        self.menu_video_remove_last_10_frames.triggered.connect(lambda: self.app.remove_last_frames(10))
        self.menu_video_make_animation.triggered.connect(self.make_animation)
        self.menu_video_remove_selected_frame.triggered.connect(self.app.remove_selected_frame)
        self.menu_video_selected_frame_to_work_area.triggered.connect(self.app.clone_selected_frame)

        self.scroll_bar_movie.sliderMoved.connect(self.on_movie_scroll)
        self.scroll_bar_movie.valueChanged.connect(self.on_movie_scroll)

        self.scene_movie = QtWidgets.QGraphicsScene(self)
        self.view_movie.setScene(self.scene_movie)
        self.scene_work_frame = WorkFrameScene(self, app)
        self.view_work_frame.setScene(self.scene_work_frame)

    def set_image(self, view, scene, image):
        image = QtGui.QImage(
            image, image.shape[1], image.shape[0], image.shape[1] * 3, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap(image)
        if not (pix.width() == view.width() and pix.height() == view.height()):
            scale = min(view.width() / pix.width(),
                        view.height() / pix.height())
            pix = pix.scaled(int(pix.width() * scale),
                             int(pix.height() * scale))
        scene.clear()
        scene.addPixmap(pix)
        if hasattr(scene, "frame_width_pxl"):
            scene.frame_width_pxl = pix.width()

    def set_movie_frame(self, image):
        self.set_image(self.view_movie, self.scene_movie, image)

    def set_work_frame(self, image):
        self.set_image(self.view_work_frame, self.scene_work_frame, image)

    def set_mono_color(self, color):
        pic = np.zeros((100, 100, 3), dtype=np.ubyte)
        for i in range(100):
            for j in range(100):
                pic[i, j, :] = color
        self.set_movie_frame(pic)

    def set_painter_config(self, text):
        edit = self.edit_painter_config  # type: QPlainTextEdit
        edit.setPlainText(text)

    def on_config_text_changed(self):
        edit = self.edit_painter_config  # type: QPlainTextEdit
        self.app.set_painter_config(edit.toPlainText())

    def set_label_painter_status(self, status):
        label = self.label_config_status  # type: QLabel
        label.setText(status)
        if status == "OK":
            label.setStyleSheet("color: green;")
        else:
            label.setStyleSheet("color: red;")

    def confirm(self, text) -> bool:
        qm = QMessageBox
        return qm.question(self, '', text, qm.Yes | qm.No) == qm.Yes

    def confirm_then(self, text, action):
        if self.confirm(text):
            action()

    def show_status(self, status):
        self.statusbar.showMessage(status)

    def on_generate_palette_click(self):
        try:
            palette = self.generate_palette()
        except ValueError as err:
            show_alert(str(err))
            return
        self.app.update_work_frame_palette(palette)

    def generate_palette(self) -> ColorPalette:
        combo = self.combo_palette_type  # type: QComboBox
        palette_type = combo.itemText(combo.currentIndex())
        spin_box = self.spin_box_palette_size  # type: QSpinBox
        size = spin_box.value()
        c1 = self.edit_color1.text()
        c2 = self.edit_color2.text()
        if palette_type == 'Grayscale':
            return ColorPalette.grayscale(size=size)
        elif palette_type == 'Random':
            return ColorPalette.random(size=size)
        if palette_type == 'Gradient':
            return ColorPalette.gradient(c1, c2, size=size)
        elif palette_type == 'Gradient+Black':
            return ColorPalette.gradient_plus_one(c1, c2, 'black', size=size)
        else:
            raise ValueError(f"Unrecognized palette type: {palette_type}")

    def on_force_update_clicked(self):
        downsample_factor = self.spin_box_downsampling.value()
        self.app.reset_work_frame(downsample_factor=downsample_factor)

    def make_animation(self):
        if len(self.app.frames) == 0:
            show_alert("Movie is empty")
            return
        if self.app.selected_frame_idx != len(self.app.frames) - 1:
            if not self.confirm("Are you sure you want insert animation in the middle of movie?"):
                return
        length = self.spin_box_animation_length.value()
        try:
            self.app.make_animation(length)
        except ValueError as err:
            show_alert(str(err))
            return
        self.on_movie_updated()

    def clear_movie(self):
        if self.confirm("Permanently delete all frames?"):
            self.app.frames = []
            self.on_movie_updated()

    def on_movie_scroll(self):
        new_idx = self.scroll_bar_movie.value()
        if new_idx != self.app.selected_frame_idx:
            self.app.selected_frame_idx = new_idx
            self.on_movie_updated()

    def on_movie_updated(self):
        mov_len = len(self.app.frames)
        cur_idx = self.app.selected_frame_idx
        sb = self.scroll_bar_movie  # type: QScrollBar
        if mov_len == 0:
            self.scene_movie.clear()
            self.label_frame_info.setText("Movie is empty")
            sb.setEnabled(False)
        else:
            assert 0 <= cur_idx < mov_len
            frame = self.app.frames[cur_idx]
            if hasattr(frame, "cached_movie_preview"):
                image = frame.cached_movie_preview
            else:
                image = self.app.movie_frame_renderer.render(self.app.frames[cur_idx])
                object.__setattr__(frame, "cached_movie_preview", image)
            self.set_movie_frame(image)
            self.label_frame_info.setText("Frame %d of %d" % (cur_idx + 1, mov_len))
            sb.setEnabled(True)
            sb.setMaximum(mov_len - 1)
            sb.setValue(cur_idx)

    def closeEvent(self, event):
        event.ignore()
        self.on_exit()

    def on_exit(self):
        # TODO: check for unsaved changes and active tasks.
        if self.confirm("Exit?"):
            QCoreApplication.exit(0)
