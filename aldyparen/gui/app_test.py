import os
import pytest
from aldyparen.gui.app import AldyparenApp


@pytest.mark.parametrize("project_name", ["mandelbrot_zoom.json", "example_project_1.json"])
def test_loads_example_project(project_name):
    app = AldyparenApp()
    path = os.path.join("examples", project_name)
    app.load_project(path)
    assert len(app.frames) > 0
