from typing import Callable

import av
import cv2
import numpy as np
import torch

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
    ):
        self.video_path = video_path
        self.reader = None
        self.recursive = recursive
        self.preprocessor = preprocess_function
        self.stream_mux = stream_mux
        self.frame_mux = frame_mux
        self.batch_size = 8

    def run(self):
        img_idx = 0
        self.reader = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)

        batch_frames, batch_inputs, batch_contexts, batch_idxs = [], [], [], []

        while True:
            try:
                hasFrame, frame = self.reader.read()
                if not hasFrame:
                    if len(batch_inputs) == self.batch_size and not self.recursive:
                        self._flush(
                            batch_frames, batch_inputs, batch_contexts, batch_idxs
                        )

                    if self.recursive:
                        self.reader.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        self.reader.release()
                        break

                input_, context = self.preprocessor(frame)

                if isinstance(input_, np.ndarray):  # numpy → tensor
                    input_ = torch.from_numpy(input_)
                if (
                    input_.ndim == 4 and input_.shape[0] == 1
                ):  # (1,3,640,640) → (3,640,640)
                    input_ = input_.squeeze(0)

                batch_frames.append(frame)
                batch_inputs.append(input_)
                batch_contexts.append(context)
                batch_idxs.append(img_idx)
                img_idx += 1

                if len(batch_inputs) >= self.batch_size:
                    self._flush(batch_frames, batch_inputs, batch_contexts, batch_idxs)

            except Exception as e:
                print(f"[VideoDecoder] {e}", self.video_path)
                break

        self.stream_mux.put(StopSig)
        self.frame_mux.put(StopSig)
        print(f"End Video!! -> {self.video_path}")
        return

    def _flush(self, frames, inputs, ctxs, idxs):
        # tensor를 numpy로 변환하여 multiprocessing 문제 해결
        if not self.recursive:
            # recursive=False일 때는 numpy로 변환해서 전달
            batched_numpy = torch.stack(inputs, dim=0).cpu().numpy()
            self.stream_mux.put(batched_numpy)  # model input
        else:
            # recursive=True일 때는 기존 방식 유지
            batched = torch.stack(inputs, dim=0)  # (B,3,640,640)
            batched = batched.clone().detach().cpu().contiguous().share_memory_()
            self.stream_mux.put(batched)  # model input

        self.frame_mux.put((frames.copy(), ctxs.copy(), idxs.copy()))  # postprocess
        frames.clear()
        inputs.clear()
        ctxs.clear()
        idxs.clear()
