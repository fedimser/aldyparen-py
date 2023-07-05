# aldyparen-py

Aldyparen (Algebraic Dynamic Parametric Renderer) is a program for rendering certain types of fractals
as high-resolution images or videos.

## Features

* "Work frame" (right-hand side):
  * Allows to explore a fractal (mouse to pan, wheel to zoom, Ctrl+wheel to rotate).
  * Optimized to be interactive (with threads).
  * Can be exported (rendered) to high-precision image (in BMP format).
  * If frame rendering is expensive, increase "downsampling" factor. 
    * Then picture will be first
      rendered at low precision, and then "refined". This allows real-time interaction
      (pan/zoom/rotation) even if the picture is expensive to render.
* "Movie" (left-hand side):
  * Can append frames from work frame.
  * Can add smooth animation, where frames are continuously transformed 
    between key frames.
    Animation is added between currently selected movie frame (on the left)
    and work frame (on the right), if they are compatible.
    * For example, it can transform function `z^2` to `z^3` through `z^2.5`.
  * Can be exported (rendered) to a video file in MP4 format.
* "Painters" - abstract framework allowing exploring and rendering different things:
  * `MandelbroidPainter` - generalization of [Mandelbrot set]() for arbitrary function.
     * Basically replaced `z^2+c` with arbitrary `f(z,c)`.
  * `MadnelbrotHighPrecisionPainter` - high-precision Mandelbrot set renderer. Uses arbitrary-precision arithmetic.
    * Will render Mandelbrot set correctly at very high zoom, where standard complex128 arithmetic 
      fails because of insufficient precision.
    * Doesn't support rotation. 
    * Is not well optimized. Long arithmetic implemented from scratch in Python and
      sped up with Numba.
    * I originally intended this for rendering video of deep zooms,
      but there is much better specialized software for that.
  * `JuliaPainter` - displays [Julia set]. 
    * Can be used to show Newton fractal (pass `func = z - P(z)/P'(z)`).
  * `SierpinskiCarpetPainter` - renders [Sierpinski carpet](https://en.wikipedia.org/wiki/Sierpi%C5%84ski_carpet), 
    as an example of non-algebraic fractal.
* Configurable color palette.
  * Painters are supposed to return numbers of colors (0,1,2...). Then they are mapped
    to RGB colors using palette. If palette is smaller than number of colors, it's 
    repeated from beginning.

## UI screenshot
<img src="examples/screenshot.jpg" width="500"/>


## TODOs

* Allow specifying Numba target (cpu/parallel/cuda).
* Add jupyter example for usage as library (incl.difference between standard/high precision).
* Add instructions for building from source.
* Build as standalone app, see https://build-system.fman.io/pyqt5-tutorial
* Check that built app is cross-platform (Windows/Linux).
* Generate few movies, save them as example JSONs, and publish rendered videos.
* Add tests for the app (loading from file, rendering to file).
* Selecting MadnelbrotHighPrecisionPainter with rotation should not crash.

## Development notes

This application is written in Python using PyQt5. 
UI layout designed using [Qt Designer](https://doc.qt.io/qt-6/qtdesigner-manual.html).
I used [Numba](https://numba.pydata.org/) for optimizing numerical calculations.

This is a Python clone (rewritten from scratch) of [Aldyparen](https://github.com/fedimser/Aldyparen),
which I have written in C# back in 2017. 
This app has all the functionality of the old Aldyparen, plus some extra features (e.g. new "painters").

To install reequirements, and run program from source, run (it's recommended to use virtualenv):

```
pip3 install -r requirements.txt
python3 run_gui.py
```

To run tests and validate style before submit, run:

```
pip3 install -r requirements.txt
pip3 install pycodestyle pytest
pycodestyle --max-line-length=120 ./aldyparen && python3 -m pytest .
```