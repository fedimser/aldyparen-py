from aldyparen.video import VideoRenderer

VideoRenderer(1024, 768, fps=10).render_movie_from_file('examples/mandelbrot_zoom.json', 'videos/mandelbrot_zoom.mp4')
