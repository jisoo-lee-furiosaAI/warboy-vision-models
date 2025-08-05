import asyncio
import os
import time
from glob import glob

import cv2
import numpy as np
import PyNvCodec as nvc
from furiosa.server.model import FuriosaRTModel, FuriosaRTModelConfig
from tqdm import tqdm

from src.warboy.cfg import get_model_params_from_cfg
from src.warboy.yolo.postprocess import ObjDetPostprocess

cfg = get_model_params_from_cfg(
    "/home/jisoo/warboy-vision-models/datasets/cfg/yolov8m.yaml"
)

from typing import Tuple

import cv2
import numpy as np


def letterbox(
    img: np.ndarray, new_shape: Tuple[int, int], color=(114, 114, 114), scaleup=True
):
    # GPU에 프레임 업로드
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(img)

    h, w = img.shape[:2]
    ratio = min(new_shape[0] / h, new_shape[1] / w)
    if not scaleup:
        ratio = min(ratio, 1.0)
    new_unpad = (int(round(w * ratio)), int(round(h * ratio)))

    # GPU Resize
    gpu_resized = cv2.cuda.resize(
        gpu_img,
        new_unpad,
        interpolation=cv2.INTER_LINEAR if ratio > 1 else cv2.INTER_AREA,
    )

    # Padding 계산
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    # GPU Padding
    gpu_padded = cv2.cuda.copyMakeBorder(
        gpu_resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )

    # 결과를 다시 CPU로 다운로드
    result = gpu_padded.download()

    return result, ratio, (dw, dh)


class YoloPreProcessor:
    def __init__(self, new_shape=(640, 640), tensor_type="uint8", gpu_id=0):
        self.new_shape = new_shape
        self.tensor_type = tensor_type
        self.gpu_id = gpu_id
        self.cc_ctx = nvc.ColorspaceConversionContext(
            nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG
        )

    def __call__(self, surface):
        # --- 1) NV12 → YUV420
        cvt_nv12_to_yuv = nvc.PySurfaceConverter(
            surface.Width(),
            surface.Height(),
            nvc.PixelFormat.NV12,
            nvc.PixelFormat.YUV420,
            self.gpu_id,
        )
        yuv_surface = cvt_nv12_to_yuv.Execute(surface, self.cc_ctx)

        # --- 2) YUV420 → BGR
        cvt_yuv_to_bgr = nvc.PySurfaceConverter(
            yuv_surface.Width(),
            yuv_surface.Height(),
            nvc.PixelFormat.YUV420,
            nvc.PixelFormat.BGR,
            self.gpu_id,
        )
        bgr_surface = cvt_yuv_to_bgr.Execute(yuv_surface, self.cc_ctx)
        if bgr_surface.Empty():
            return

        # --- 3) Download to CPU
        downloader = nvc.PySurfaceDownloader(
            bgr_surface.Width(), bgr_surface.Height(), nvc.PixelFormat.BGR, self.gpu_id
        )
        frame = np.empty((bgr_surface.Height(), bgr_surface.Width(), 3), dtype=np.uint8)
        success = downloader.DownloadSingleSurface(bgr_surface, frame)
        if not success:
            raise RuntimeError("Surface download failed")

        # --- 4) letterbox
        img, ratio, (padw, padh) = letterbox(frame, self.new_shape)

        # --- 5) BGR→RGB, HWC→CHW
        img = img.transpose(2, 0, 1)[::-1]

        if self.tensor_type == "uint8":
            input_ = np.ascontiguousarray(np.expand_dims(img, 0), dtype=np.uint8)
        else:
            input_ = (
                np.ascontiguousarray(np.expand_dims(img, 0), dtype=np.float32) / 255.0
            )

        preproc_params = {"ratio": ratio, "pad": (padw, padh)}
        return frame, input_, preproc_params


class WarboyRunner:
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
        self.preprocessor = YoloPreProcessor()
        self.postprocessor = ObjDetPostprocess(
            model_name=cfg["model_name"],
            model_cfg=runner_info,
            class_names=cfg["class_names"],
            use_traking=False,
        )
        self.input_shape = cfg["input_shape"]
        self.result_path = output_path
        self.input_path = input_path

        os.makedirs(self.result_path, exist_ok=True)

    async def load(self):
        await self.model.load()

    async def unload(self):
        await self.model.unload()

    async def process(self, video_path):
        nv_decoder = nvc.PyNvDecoder(video_path, 0)

        while True:
            try:
                surface = nv_decoder.DecodeSingleSurface()
                if surface.Empty():
                    break

                frame_bgr, input_, context = self.preprocessor(surface)
                if frame_bgr is None or input_ is None:
                    continue

                output = await self.model.predict(input_)

                out = await asyncio.to_thread(
                    self.postprocessor, output, context, frame_bgr
                )

                # Save a frame snapshot
                cv2.imwrite(
                    os.path.join(
                        self.result_path, f"{os.path.basename(video_path)}.jpg"
                    ),
                    out,
                )

            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                return

    async def run(self):
        # If input_path is a directory, process all videos
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
    input_video = "/home/jisoo/warboy-vision-models/datasets/warboy_tutorial/part5/assets/videos/detection_videos/cow_1.mp4"
    runner = WarboyRunner(
        model=cfg["onnx_i8_path"],
        input_path=input_video,
        output_path="/home/jisoo/warboy-vision-models/datasets/test_output",
        runner_info={
            "iou_thres": cfg["iou_thres"],
            "conf_thres": cfg["conf_thres"],
            "anchors": [None],
        },
    )

    results, infer_time = asyncio.run(inference(runner))
    print(f"Inference done: {results}, Time taken: {infer_time:.2f}s")
