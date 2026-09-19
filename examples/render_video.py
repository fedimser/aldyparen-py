from aldyparen.video import VideoRenderer

input_path = "./examples/aldyparen_011.json"
output_path = "./aldyparen-video-projects/aldyparen_011.mp4"

# HD 1280x720
# 1440p 2560x1440
renderer = VideoRenderer(256, 144, fps=24, max_memory_bytes=500_000_000, skip_existing_parts=True)
renderer.render_movie_from_file(input_path, output_path)
