import json
import os
import time
from pathlib import Path

import matplotlib.image as mpimg
import pytest
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QFileDialog

from aldyparen.gui.app import AldyparenApp
from aldyparen.test_util import _assert_picture


def test_app_runs_and_closes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    project_file = os.path.abspath("examples/example_project_1.json")
    app = AldyparenApp()
    monkeypatch.chdir(tmp_path)
    expected_config = {
        "gen_function": "1*z**4+1*z**3+1*z**2+1.0*z+1*sin(z)+c",
        "radius": 100,
        "max_iter": 500,
    }
    deadline = time.monotonic() + 30
    result = {}

    def close_with_error(message: str):
        result["error"] = message
        app.main_window.close()

    def select_project():
        dialogs = [
            widget
            for widget in QApplication.topLevelWidgets()
            if isinstance(widget, QFileDialog) and widget.isVisible()
        ]
        if not dialogs:
            if time.monotonic() < deadline:
                QTimer.singleShot(10, select_project)
            else:
                close_with_error("Open Project dialog did not appear")
            return
        dialog = dialogs[0]
        dialog.selectFile(project_file)
        QTimer.singleShot(10, dialog.accept)

    def project_is_rendered():
        if app.opened_file_name is not None:
            result["config"] = app.main_window.edit_painter_config.toPlainText()
            app.main_window.spin_box_image_resolution_1.setValue(120)
            app.main_window.spin_box_image_resolution_2.setValue(160)
            app.main_window.button_render_photo.click()
            QTimer.singleShot(10, image_is_rendered)
        elif time.monotonic() < deadline:
            QTimer.singleShot(10, project_is_rendered)
        else:
            close_with_error("Project did not render")

    def image_is_rendered():
        rendered_files = list((tmp_path / "images").glob("*.bmp"))
        if rendered_files and app.photo_rendering_tasks_count == 0:
            result["picture"] = mpimg.imread(rendered_files[0])
            app.main_window.close()
        elif app.error_messages_to_show:
            close_with_error(app.error_messages_to_show[0])
        elif time.monotonic() < deadline:
            QTimer.singleShot(10, image_is_rendered)
        else:
            close_with_error("Image did not render")

    QTimer.singleShot(0, app.main_window.open_project)
    QTimer.singleShot(0, select_project)
    QTimer.singleShot(10, project_is_rendered)
    app.run()

    assert "error" not in result, result.get("error")
    assert json.loads(result["config"]) == expected_config
    assert result["picture"].shape[:2] == (160, 120)
    _assert_picture(result["picture"], "gui_render_image")
    assert app.is_exiting
