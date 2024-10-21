import math
import os
from typing import Union, TYPE_CHECKING

import numpy as np
from PyQt5 import QtWidgets, QtGui, uic, QtCore
from PyQt5.QtCore import QPointF, QCoreApplication, QUrl, QThreadPool, QSize
from PyQt5.QtGui import QDesktopServices, QColor, QIcon
from PyQt5.QtWidgets import QMessageBox, QGraphicsSceneWheelEvent, QGraphicsSceneMouseEvent, QApplication, QComboBox, \
    QPlainTextEdit, QLabel, QSpinBox, QScrollBar, QFileDialog, QColorDialog

from .async_runners import render_movie_preview_async, render_video_async
from ..graphics import ColorPalette, Transform
from ..painters import ALL_PAINTERS

if TYPE_CHECKING:
    from .app import AldyparenApp


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
        self.cursor_rel_screen_pos = None

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        self.calculate_cursor_pos(event.scenePos())
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
            delta *= 25

        self.calculate_cursor_pos(event.scenePos())
        if self.cursor_math_pos is None:
            return
        if bool(modifiers & QtCore.Qt.ControlModifier):
            # 2 degrees minimal increment (for standard mouse).
            angle = delta * (np.pi / 90)
            tr = self.app.work_frame.transform.rotate_and_scale_at(self.cursor_rel_screen_pos, angle=angle)

        else:
            tr = self.app.work_frame.transform.rotate_and_scale_at(self.cursor_rel_screen_pos,
                                                                   scale_factor=10 ** (0.02 * delta))
        self.app.update_work_frame_transform(tr)

    def apply_drag(self, dx_pxl, dy_pxl):
        delta = np.complex128(dx_pxl - 1j * dy_pxl) / self.frame_width_pxl
        self.app.update_work_frame_transform(
            self.app.work_frame.transform.translate(delta))

    def calculate_cursor_pos(self, pos: QPointF):
        x = pos.x()
        y = pos.y()
        if x <= 0 or x >= self.width() or y <= 0 or y >= self.height():
            self.cursor_rel_screen_pos = None
            self.cursor_math_pos = None
        else:
            x = x - 0.5 * self.width()
            y = -(y - 0.5 * self.height())
            self.cursor_rel_screen_pos = np.complex128(x + 1j * y) / self.frame_width_pxl
            self.cursor_math_pos = self.app.work_frame.transform.map_screen_to_math(self.cursor_rel_screen_pos)


class PalettePreviewScene(QtWidgets.QGraphicsScene):
    def __init__(self, parent, app: 'AldyparenApp'):
        super().__init__(parent)
        self.app = app

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        num_colors = self.app.work_frame.palette.colors.shape[0]
        color_idx = int(np.floor((event.scenePos().x() / self.width()) * num_colors))
        cur_clr = self.app.work_frame.palette.colors[color_idx]
        dialog = QColorDialog()
        dialog.setCurrentColor(QColor(cur_clr[0], cur_clr[1], cur_clr[2], 255))
        dialog.exec()
        if dialog.result() == 1:
            self.app.set_palette_color(color_idx, dialog.selectedColor())


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, app: 'AldyparenApp'):
        super(MainWindow, self).__init__()
        self.ui_handlers_locked = True
        self.app = app
        uic.loadUi('layout/main.xml', self)

        app_icon = QIcon()
        app_icon.addFile('layout/icon16.png', QSize(16, 16))
        app_icon.addFile('layout/icon64.png', QSize(64, 64))
        self.setWindowIcon(app_icon)

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

        # Edits
        self.edit_painter_config.textChanged.connect(self.on_config_text_changed)
        self.edit_center_x.textChanged.connect(self.on_transform_text_edited)
        self.edit_center_y.textChanged.connect(self.on_transform_text_edited)
        self.edit_scale_log10.textChanged.connect(self.on_transform_text_edited)
        self.edit_rotation_deg.textChanged.connect(self.on_transform_text_edited)

        # Buttons.
        self.button_reset_transform.clicked.connect(
            lambda: self.app.reset_transform())
        self.button_reset_painter_config.clicked.connect(
            lambda: self.confirm_then("Reset painter config?", self.app.reset_painter_config))
        self.button_generate_palette.clicked.connect(self.on_generate_palette_click)
        self.button_reset_work_frame.clicked.connect(self.app.reset_work_frame)
        self.button_reset_video_preview.clicked.connect(self.app.reset_video_preview)
        self.button_render_photo.clicked.connect(self.render_image)
        self.button_render_video.clicked.connect(self.render_video)
        self.button_make_animation.clicked.connect(self.make_animation)

        # Menu items.
        self.menu_new_project.triggered.connect(self.new_project)
        self.menu_open_project.triggered.connect(self.open_project)
        self.menu_save_project.triggered.connect(self.save_project)
        self.menu_save_project_as.triggered.connect(self.save_project_as)
        self.menu_render_photo.triggered.connect(self.render_image)
        self.menu_render_video.triggered.connect(self.render_video)
        self.menu_settings.triggered.connect(lambda: show_alert("Not implemented"))
        self.menu_exit.triggered.connect(self.on_exit)
        self.menu_video_clear.triggered.connect(self.clear_movie)
        self.menu_video_append.triggered.connect(self.app.append_movie_frame)
        self.menu_video_replace.triggered.connect(self.app.replace_movie_frame)
        self.menu_video_remove_last_frame.triggered.connect(lambda: self.app.remove_last_frames(1))
        self.menu_video_remove_last_10_frames.triggered.connect(lambda: self.app.remove_last_frames(10))
        self.menu_video_make_animation.triggered.connect(self.make_animation)
        self.menu_video_remove_selected_frame.triggered.connect(self.app.remove_selected_frame)
        self.menu_video_selected_frame_to_work_area.triggered.connect(self.app.clone_selected_frame)
        self.menu_mandelbrot.triggered.connect(lambda: self.app.load_preset("mandelbrot"))
        self.menu_mandelbrot_hp.triggered.connect(lambda: self.app.load_preset("mandelbrot_hp"))
        self.menu_burning_ship.triggered.connect(lambda: self.app.load_preset("burning_ship"))
        self.menu_burning_ship_hp.triggered.connect(lambda: self.app.load_preset("burning_ship_hp"))
        self.menu_docs.triggered.connect(self.open_docs)

        self.scroll_bar_movie.sliderMoved.connect(self.on_movie_scroll)
        self.scroll_bar_movie.valueChanged.connect(self.on_movie_scroll)

        self.scene_movie = QtWidgets.QGraphicsScene(self)
        self.view_movie.setScene(self.scene_movie)
        self.scene_work_frame = WorkFrameScene(self, app)
        self.view_work_frame.setScene(self.scene_work_frame)
        self.scene_palette_preview = PalettePreviewScene(self, app)
        self.view_palette_preview.setScene(self.scene_palette_preview)

        self.transform_text_is_invalid = True
        self.ui_handlers_locked = False

    def set_image(self, view, scene, image):
        if type(image) is str:
            scene.clear()
            scene.addText(image)
            return
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
        if self.app.is_loading_project or self.ui_handlers_locked:
            return
        edit = self.edit_painter_config  # type: QPlainTextEdit
        self.ui_handlers_locked = True
        self.app.set_painter_config(edit.toPlainText())
        self.ui_handlers_locked = False

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
        self.show_palette_preview(palette)
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
            self.label_frame_info.setText(self.app.get_selected_frame_info())
            sb.setEnabled(True)
            sb.setMaximum(mov_len - 1)
            sb.setValue(cur_idx)
            self.app.shown_movie_frame_is_invalid = True
        self.update_title()

    def closeEvent(self, event):
        event.ignore()
        self.on_exit()

    def on_exit(self):
        if self.app.photo_rendering_tasks_count > 0 or self.app.video_rendering_tasks_count > 0:
            if not self.confirm("There are unfinished tasks. Exiting now may cancel them. Exit anyway?"):
                return
        if self.app.have_unsaved_changes:
            if not self.confirm("There are unsaved changes. Exit anyway?"):
                return

        self.app.is_exiting = True
        self.app.work_frame_renderer.halt()
        QThreadPool.globalInstance().clear()  # Cancels not yet started tasks.
        self.app.settings.save()
        if not QThreadPool.globalInstance().waitForDone(msecs=500):
            show_alert("Please wait for active tasks to be finished or stopped.")
        QThreadPool.globalInstance().waitForDone()

        QCoreApplication.exit(0)

    def render_image(self):
        width = self.spin_box_image_resolution_1.value()
        height = self.spin_box_image_resolution_2.value()
        self.app.render_image(width, height)

    def render_video(self):
        frames_count = len(self.app.frames)
        if frames_count == 0:
            show_alert("Movie is empty, can't render.")
            return
        if self.app.photo_rendering_tasks_count > 0:
            show_alert("Movie is empty, can't render.")
            return
        width = self.spin_box_video_resolution_1.value()
        height = self.spin_box_video_resolution_2.value()
        fps = self.app.settings.get_video_fps()
        filters = "MP4 video files (*.mp4);;All files (*.*)"
        file_name = QFileDialog.getSaveFileName(self, 'Choose video location', self.app.settings.work_dir, filters)[0]
        if len(file_name) == 0:
            return
        dur_sec = math.ceil(len(self.app.frames) / fps)
        prompt = "\n".join([
            "Confirm video render.",
            f"Resolution: {width}x{height}",
            f"Frame rate: {fps} FPS",
            f"Location: {file_name}",
            f"Duration: {len(self.app.frames)} frames, {dur_sec} seconds."
        ])
        if not self.confirm(prompt):
            return
        render_video_async(self.app, width, height, fps, file_name)
        self.app.settings.work_dir = os.path.dirname(file_name)

    def open_docs(self):
        url = QUrl("https://github.com/fedimser/aldyparen-py")
        QDesktopServices.openUrl(url)

    def new_project(self):
        if self.app.have_unsaved_changes:
            if not self.confirm("There are unsaved changes. Create new project anyway?"):
                return
        self.app.new_project()

    def open_project(self):
        filters = "JSON files (*.json);;All files (*.*)"
        file_name = QFileDialog.getOpenFileName(self, 'Open Aldyparen project', self.app.settings.work_dir, filters)[0]
        if len(file_name) == 0:
            return
        self.app.load_project(file_name)
        self.app.settings.work_dir = os.path.dirname(file_name)
        self.update_title()

    def save_project(self):
        if self.app.opened_file_name is not None:
            self.app.save_project()
            self.update_title()
        else:
            self.save_project_as()

    def save_project_as(self):
        filters = "JSON files (*.json);;All files (*.*)"
        file_name = QFileDialog.getSaveFileName(self, 'Save Aldyparen project', self.app.settings.work_dir, filters)[0]
        if len(file_name) == 0:
            show_alert("File not selected - nothing was saved.")
            return
        self.app.opened_file_name = file_name
        self.app.save_project()
        self.app.settings.work_dir = os.path.dirname(file_name)
        self.update_title()

    def update_title(self):
        self.setWindowTitle(self.app.get_window_title())

    def show_palette_preview(self, palette: ColorPalette):
        colors_num = palette.colors.shape[0]
        self.label_current_pelette_colors_num.setText("%d colors" % colors_num)

        w = self.view_palette_preview.width()
        h = self.view_palette_preview.height()
        row = np.zeros((1, w, 3), dtype=np.uint8)
        for i in range(w):
            j = (i * colors_num) // w
            row[0, i, :] = palette.colors[j, :]
        image = np.tile(row, (h, 1, 1))
        self.set_image(self.view_palette_preview, self.scene_palette_preview, image)

    def update_transform_text(self):
        if self.ui_handlers_locked:
            return
        self.ui_handlers_locked = True
        tr = self.app.work_frame.transform
        self.edit_center_x.setText(str(tr.center_x))
        self.edit_center_y.setText(str(tr.center_y))
        self.edit_scale_log10.setText(f'{tr.scale_log10:.5f}')
        self.edit_rotation_deg.setText(str(tr.rotation_deg()))
        self.transform_text_is_invalid = False
        self.ui_handlers_locked = False

    def on_transform_text_edited(self):
        if self.ui_handlers_locked:
            return

        try:
            scale_log10 = float(self.edit_scale_log10.text())
            rotation_deg = float(self.edit_rotation_deg.text())
            new_transform = Transform.create(center_x=self.edit_center_x.text(),
                                             center_y=self.edit_center_y.text(),
                                             scale_log10=scale_log10, rotation_deg=rotation_deg)
        except Exception as e:
            self.transform_text_is_invalid = True
            return
        self.ui_handlers_locked = True
        self.app.update_work_frame_transform(new_transform)
        self.ui_handlers_locked = False
