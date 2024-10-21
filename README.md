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
    * `MandelbroidPainter` - generalization of [Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set) for
      arbitrary function.
        * Basically replaced `z^2+c` with arbitrary `f(z,c)`.
        * It has parameters: `gen_function` (a function of two arguments `z` and `c`, for example `"z**2+c""`),
            `radius` and `max_iter`.
        * To paint point `c`, we start we `z=0` and repeat `z := gen_function(z,c)` until either `|z| > radius` or
            we did `max_iter` iterations. Number of iterations made determines the color of the point `c`.
        * I call such generalized Mandelbrot fractal a "Mandelbroid fractal".
    * `MadnelbroidHighPrecisionPainter` - paints Mandelbroid fractal, but with high-precision.
        * Will render Mandelbroid set correctly at very high zoom, where standard 64-bit floating point arithmetic
          fails because of insufficient precision. To illustrate why we need high-precision, compare these two pictures: 
          [1](https://photos.app.goo.gl/T1M72irowzJn4Nqd6), [2](https://photos.app.goo.gl/ZtiAdVYQJ4W1MfzU7).
          They both show the same region of the [Burning Ship fractal](https://en.wikipedia.org/wiki/Burning_Ship_fractal),
          but the first one uses standard 64-bit floating arithmetic, while the second uses high-precision arithmetic.
        * Parameters:
            * `gen_function`,`radius` and `max_iter` - same as for `MandelbroidPainter`.
        * Currently `gen_function` supports addition, subtraction, multiplication andthe following functions:
            * `sqr` - square, `sqr(z) = z * z`.
            * `abscw` - component-wise modulus, `abscw(z) = |Re(z)| + i*|Im(z)|`.
        * Panning with mouse will not work at very high zoom, but you can specify center with arbitrary precision in a
          text edit in the "Transform" tab, and it will work correctly.
        * It's not well optimized. Long arithmetic implemented from scratch in Python and
          sped up with Numba.
        * I originally intended this for rendering video of deep zooms,
          but there is much better specialized software for that. 
    * `MadnelbrotHighPrecisionPainter` - specialized version of `MadnelbroidHighPrecisionPainter`.
        * Renders only Mandelbrot set with fixed `radius=2`, but does it more efficiently.
        * Parameters: `max_iter`.
    * `JuliaPainter` - displays [Julia set](https://en.wikipedia.org/wiki/Julia_set).
        * Can be used to show [Newton fractal](https://en.wikipedia.org/wiki/Newton_fractal)
          (pass `func = z - P(z)/P'(z)`).
        * Parameters: `func`, `iters`, `tolerance`, `max_colors`.
    * `SierpinskiCarpetPainter` - renders [Sierpinski carpet](https://en.wikipedia.org/wiki/Sierpi%C5%84ski_carpet),
      as an example of non-algebraic fractal.
        * Parameters: `depth`.
* Configurable color palette.
    * Painters are supposed to return numbers of colors (0,1,2...). Then they are mapped
      to RGB colors using palette. If palette is smaller than number of colors, it's
      repeated from beginning.
    * There are some preset palettes (grayscale, gradient, etc.).
    * You can change individual colors by clicking on them in the "Palette" tab.

## UI screenshots

<img src="examples/screenshot1.jpg" width="500"/>
<img src="examples/screenshot2.png" width="500"/>

## Examples

* Example of using this as Python library - [link](examples/example.ipynb).
* Example project - [link](examples/example_project_1.json) (rendered video - [link](https://www.youtube.com/watch?v=fsI0lQ-PMnI)).
* Example high-resolution renders - [link](https://photos.app.goo.gl/TRyUn9QRy7kJ1sYP8).

## How to install and run

Requirements:
* OS: Windows/Linux/MacOS.
* Python 3.10 or higher.
* Git.

Run from command line:
```
git clone https://github.com/fedimser/aldyparen-py.git
cd aldyparen-py
pip3 install -r requirements.txt
python3 run_gui.py
```

## Development notes

This application is written in Python using PyQt5.
UI layout designed using [Qt Designer](https://doc.qt.io/qt-6/qtdesigner-manual.html).
I used [Numba](https://numba.pydata.org/) for optimizing numerical calculations.

This is a Python clone (rewritten from scratch) of [Aldyparen](https://github.com/fedimser/Aldyparen),
which I have written in C# back in 2017.
This app has all the functionality of the old Aldyparen, plus some extra features (e.g. new "painters").

To start development, create virtualenv and install dependencies:
```
pip3 install -r requirements.txt
pip3 install -r requirements-dev.txt
```

To run tests and validate style before commit, run:
```
pycodestyle --max-line-length=120 ./aldyparen && python3 -m pytest .
```
