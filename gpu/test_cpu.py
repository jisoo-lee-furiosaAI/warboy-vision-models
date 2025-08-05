import asyncio
import os
import time
from glob import glob

import cv2
import numpy as np
from furiosa.server.model import FuriosaRTModel, FuriosaRTModelConfig
from tqdm import tqdm

from src.warboy.cfg import get_model_params_from_cfg
from src.warboy.yolo.postprocess import ObjDetPostprocess
from src.warboy.yolo.preprocess import letterbox

cfg = get_model_params_from_cfg(
    "/home/jisoo/warboy-vision-models/datasets/cfg/yolov8m.yaml"
)


class YoloPreProcessorCPU:
    def __init__(self, new_shape=(640, 640), tensor_type="uint8"):
        self.new_shape = new_shape
        self.tensor_type = tensor_type

    def __call__(self, frame):
        # --- 1) letterbox resize
        img, ratio, (padw, padh) = letterbox(frame, self.new_shape)

        # --- 2) BGR→RGB, HWC→CHW
        img = img.transpose(2, 0, 1)[::-1]

        if self.tensor_type == "uint8":
            input_ = np.ascontiguousarray(np.expand_dims(img, 0), dtype=np.uint8)
        else:
            input_ = (
                np.ascontiguousarray(np.expand_dims(img, 0), dtype=np.float32) / 255.0
            )

        preproc_params = {
            "ratio": ratio,
            "pad": (padw, padh),
            "org_input_shape": frame.shape[:2],
        }
        return frame, input_, preproc_params


class WarboyRunnerCPU:
    def __init__(self, model, input_path, output_path, runner_info):
        self.model = FuriosaRTModel(
            FuriosaRTModelConfig(
                name=cfg["model_name"],
                model=model,
                batch_size=1,
                worker_num=8,
                npu_device="warboy(1)*2",
            )
        )
        self.preprocessor = YoloPreProcessorCPU()
        self.postprocessor = ObjDetPostprocess(
            model_name=cfg["model_name"],
            model_cfg=runner_info,
            class_names=cfg["class_names"],
            use_traking=False,
        )
        self.input_path = input_path
        self.result_path = output_path
        os.makedirs(self.result_path, exist_ok=True)

    async def load(self):
        await self.model.load()

    async def unload(self):
        await self.model.unload()

    async def process(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: cannot open video {video_path}")
            return

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_bgr, input_, context = self.preprocessor(frame)
            if frame_bgr is None or input_ is None:
                continue

            # Run inference
            output = await self.model.predict(input_)

            # Postprocess (detection + annotation)
            out = await asyncio.to_thread(
                self.postprocessor, output, context, frame_bgr
            )

            out_path = os.path.join(
                self.result_path, f"{os.path.basename(video_path)}.jpg"
            )
            cv2.imwrite(out_path, out)

            frame_count += 1

        cap.release()

    async def run(self):
        if os.path.isdir(self.input_path):
            video_list = glob(os.path.join(self.input_path, "*.mp4"))
        else:
            video_list = [self.input_path]

        for video in tqdm(video_list, desc="Processing Videos"):
            await self.process(video)

        return f"Processed {len(video_list)} video(s)"


async def inference(runner):
    await runner.load()
    t1 = time.time()
    results = await runner.run()
    t2 = time.time()
    await runner.unload()
    return results, t2 - t1


if __name__ == "__main__":
    input_video = "/home/jisoo/warboy-vision-models/datasets/video/high/v_04601.mp4"
    runner = WarboyRunnerCPU(
        model=cfg["onnx_i8_path"],
        input_path=input_video,
        output_path="/home/jisoo/warboy-vision-models/datasets/test_cpu",
        runner_info={
            "iou_thres": cfg["iou_thres"],
            "conf_thres": cfg["conf_thres"],
            "anchors": [None],
        },
    )

    results, infer_time = asyncio.run(inference(runner))
    print(f"Inference done: {results}, Time taken: {infer_time:.2f}s")
