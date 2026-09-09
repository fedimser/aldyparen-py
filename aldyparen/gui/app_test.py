from PyQt5.QtCore import QTimer

from aldyparen.gui.app import AldyparenApp


def test_app_runs_and_closes(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    app = AldyparenApp()
    QTimer.singleShot(0, app.main_window.close)
    app.run()

    assert app.is_exiting
