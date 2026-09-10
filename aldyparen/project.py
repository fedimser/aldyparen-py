import json
from dataclasses import dataclass
from datetime import datetime

from .graphics import Frame
from .version import VERSION


@dataclass
class AldyparenProject:
    """List of frames and all additional info that is saved to a file."""

    frames: list[Frame]
    work_frame: Frame
    selected_frame_idx: int
    description: str

    @staticmethod
    def create(
        frames: list[Frame],
        *,
        work_frame: Frame | None = None,
        selected_frame_idx: int = 0,
        description: str = "",
    ):
        if work_frame is None:
            work_frame = frames[0] if len(frames) > 0 else Frame.default()
        if len(frames) == 0:
            selected_frame_idx = -1

        return AldyparenProject(frames, work_frame, selected_frame_idx, description)

    @staticmethod
    def load(file_name: str):
        with open(file_name, "r", encoding="utf-8") as f:
            data = json.load(f)

        frames = []
        prev = None
        for frame_json in data["frames"]:
            frame = Frame.deserialize(frame_json, prev=prev)
            frames.append(frame)
            prev = frame

        return AldyparenProject(
            frames,
            Frame.deserialize(data["work_frame"]),
            data["selected_frame_idx"],
            data.get("description", ""),
        )

    def save(self, file_name: str):
        """Saves project to a JSON file."""
        frames_json = []
        prev = None
        for frame in self.frames:
            frames_json.append(frame.serialize(prev=prev))
            prev = frame
        data = {
            "saved_timestamp": datetime.now().isoformat(),
            "version": VERSION,
            "work_frame": self.work_frame.serialize(),
            "frames": frames_json,
            "selected_frame_idx": self.selected_frame_idx,
            "description": self.description,
        }
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(data, f)
