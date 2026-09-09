from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QThreadPool
from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget


def global_thread_pool() -> QThreadPool:
    """Return Qt's application-wide thread pool.

    PyQt's type information marks ``globalInstance`` as optional, although Qt
    provides the singleton while an application is running.
    """
    thread_pool = QThreadPool.globalInstance()
    assert thread_pool is not None
    return thread_pool


class SafeFileIconProvider(QtWidgets.QFileIconProvider):
    """Provide generic icons without consulting the desktop icon provider."""

    def __init__(self):
        super().__init__()
        style = QApplication.style()
        assert style is not None
        self.directory_icon = style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DirIcon)
        self.file_icon = style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon)

    def icon(self, file_info: object):
        if isinstance(file_info, QtCore.QFileInfo) and file_info.isDir():
            return self.directory_icon
        if file_info == self.Folder:
            return self.directory_icon
        return self.file_icon


def select_file(
    parent: QWidget,
    title: str,
    directory: str,
    filters: str,
    accept_mode: QFileDialog.AcceptMode,
    default_suffix: str = "",
) -> str:
    """Selects file using dialog."""
    dialog = QFileDialog(parent)
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)

    # Workaround for running from within VSCode.
    file_system_model = dialog.findChild(QtWidgets.QFileSystemModel)
    if file_system_model is not None:
        file_system_model.setIconProvider(SafeFileIconProvider())

    dialog.setWindowTitle(title)
    dialog.setNameFilter(filters)
    dialog.setAcceptMode(accept_mode)
    dialog.setDefaultSuffix(default_suffix)
    if accept_mode == QFileDialog.AcceptOpen:
        dialog.setFileMode(QFileDialog.ExistingFile)
    else:
        dialog.setFileMode(QFileDialog.AnyFile)
    dialog.setDirectory(directory)

    if dialog.exec() != QtWidgets.QDialog.Accepted:
        return ""
    selected_files = dialog.selectedFiles()
    return selected_files[0] if selected_files else ""
