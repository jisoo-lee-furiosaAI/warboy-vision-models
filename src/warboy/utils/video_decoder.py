from typing import Callable

import cv2

from ..yolo.preprocess import YoloPreProcessor
from .queue import PipeLineQueue, StopSig


class VideoDecoder:
    def __init__(
        self,
        video_path: str,
        stream_mux: PipeLineQueue,
        frame_mux: PipeLineQueue,
        preprocess_function: Callable = YoloPreProcessor(),
        recursive: bool = False,
        skip_frames: int = 1,
    ):
        self.video_path = video_path
        self.reader = None
        self.recursive = recursive
        self.preprocessor = preprocess_function
        self.stream_mux = stream_mux
        self.frame_mux = frame_mux
        self.skip_frames = skip_frames

    def run(self):
        img_idx = 0
        if "v4l2src" in self.video_path:
            self.reader = cv2.VideoCapture(self.video_path, cv2.CAP_GSTREAMER)
            if not self.reader.isOpened():
                import re

                print("GStreamer pipeline failed, trying direct access...")

                match = re.search(r"device=/dev/video(\d+)", self.video_path)
                if match:
                    device_number = int(match.group(1))
                    print(f"Fallback to device {device_number}")
                    self.reader = cv2.VideoCapture(device_number)
                else:
                    print("Could not extract device number, using default 0")
                    self.reader = cv2.VideoCapture(0)
        else:
            self.reader = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
        while True:
            try:
                hasFrame, frame = self.reader.read()
                if not hasFrame:
                    if self.recursive:
                        self.reader.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        self.reader.release()
                        break
                if img_idx % self.skip_frames == 0:
                    input_, context = self.preprocessor(frame)
                    self.stream_mux.put((input_, img_idx))
                    self.frame_mux.put((frame, context, img_idx))
                img_idx += 1

            except Exception as e:
                print(e, self.video_path)
                break

        self.stream_mux.put(StopSig)
        self.frame_mux.put(StopSig)
        print(f"End Video!! -> {self.video_path}")
        return
