from typing import Callable

import cv2
import numpy as np

from ..yolo.preprocess import YoloPreProcessor
from .queue import PipeLineQueue, StopSig


class VideoDecoder:
    def __init__(
        self,
        video_path: str,
        stream_mux: PipeLineQueue,
        frame_mux: PipeLineQueue,
        batch_size: int,
        preprocess_function: Callable = YoloPreProcessor(),
        recursive: bool = False,
    ):
        self.video_path = video_path
        self.reader = None
        self.recursive = recursive
        self.preprocessor = preprocess_function
        self.stream_mux = stream_mux
        self.frame_mux = frame_mux
        self.batch_size = batch_size

    def run(self):
        img_idx = 0
        self.reader = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
        batch_frames, batch_inputs, batch_contexts, batch_idxs = [], [], [], []

        try:
            while True:
                hasFrame, frame = self.reader.read()
                if not hasFrame:
                    if len(batch_inputs) >= self.batch_size:
                        self._flush(
                            batch_frames, batch_inputs, batch_contexts, batch_idxs
                        )

                    if self.recursive:
                        self.reader.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                input_, context = self.preprocessor(frame)

                batch_frames.append(frame)
                batch_inputs.append(input_)
                batch_contexts.append(context)
                batch_idxs.append(img_idx)
                img_idx += 1

                if len(batch_inputs) >= self.batch_size:
                    self._flush(batch_frames, batch_inputs, batch_contexts, batch_idxs)

        except Exception as e:
            print(f"[VideoDecoder Error] {e} :: {self.video_path}")

        finally:
            self.reader.release()
            self.stream_mux.put(StopSig)
            self.frame_mux.put(StopSig)
            print(f"[VideoDecoder] End of video: {self.video_path}")

    def _flush(self, frames, inputs, ctxs, idxs):
        if not inputs:
            return
        batched_array = np.stack(inputs, axis=0)
        self.stream_mux.put(batched_array)
        self.frame_mux.put((frames[:], ctxs[:], idxs[:]))
        frames.clear(), inputs.clear(), ctxs.clear(), idxs.clear()
