# aldyparen-py

Aldyparen (Algebraic Dynamic Parametric Renderer) is a program for rendering certain types of fractals as photos or videos.

## Development notes

This is a Python clone (rewritten from scratch) of [Aldyparen](https://github.com/fedimser/Aldyparen), which I have written in C# back in 2017. It's built using PyQt5. UI layout designed using [Qt Designer](https://doc.qt.io/qt-6/qtdesigner-manual.html).

To install reequirements, and run program from source, run (it's recommended to use virtualenv):

```
pip3 install -r requirements.txt
python3 run_gui.py
```

To run test and validate style before submit, run:

```
pip3 install -r requirements.txt
pip3 install pycodestyle pytest
pycodestyle --max-line-length=120 ./aldyparen && python3 -m pytest .
```