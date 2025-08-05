import asyncio
import json
import os
import subprocess
import time
from glob import glob

import cv2
import numpy as np
import torch
from furiosa.server.model import FuriosaRTModel, FuriosaRTModelConfig
from tqdm import tqdm

from src.warboy.cfg import get_model_params_from_cfg
from src.warboy.yolo.cbox_decode import yolov8_box_decode

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
    def __init__(self, new_shape=(640, 640), tensor_type="uint8"):
        self.new_shape = new_shape
        self.tensor_type = tensor_type

    def __call__(self, frame):
        # --- 1) letterbox
        img, ratio, (padw, padh) = letterbox(frame, self.new_shape)

        # --- 2) BGR→RGB, HWC→CHW
        img = img.transpose(2, 0, 1)[::-1]

        if self.tensor_type == "uint8":
            input_ = np.ascontiguousarray(np.expand_dims(img, 0), dtype=np.uint8)
        else:
            input_ = (
                np.ascontiguousarray(np.expand_dims(img, 0), dtype=np.float32) / 255.0
            )

        preproc_params = {"ratio": ratio, "pad": (padw, padh)}
        return input_, preproc_params


import re
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision


# =====================
# Utility functions
# =====================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# --------- GPU용 scale_coords ----------
def scale_coords_gpu(boxes: torch.Tensor, ratio: float, dwdh, org_input_shape):
    boxes[:, [0, 2]] -= dwdh[0]
    boxes[:, [1, 3]] -= dwdh[1]
    boxes[:, :4] /= ratio
    h, w = org_input_shape
    boxes[:, 0].clamp_(0, w)
    boxes[:, 1].clamp_(0, h)
    boxes[:, 2].clamp_(0, w)
    boxes[:, 3].clamp_(0, h)
    return boxes


# --------- GPU용 NMS ----------
def non_max_suppression_gpu(prediction: List[np.ndarray], iou_thres: float = 0.45):
    output = []
    for x in prediction:
        t = torch.from_numpy(x).cuda()  # (N, 6)
        boxes, scores = t[:, :4], t[:, 4]
        keep = torchvision.ops.nms(boxes, scores, iou_thres)
        output.append(t[keep].cpu().numpy())
    return output


# --------- GPU/CPU 혼합 Draw BBox ----------
def sanitize_predictions(predictions):
    # NaN/Inf 좌표, 음수 클래스 등 제거
    mask = np.isfinite(predictions).all(axis=1)
    predictions = predictions[mask]
    # confidence > 0, class index >= 0
    mask2 = (predictions[:, 4] > 0) & (predictions[:, 5] >= 0)
    predictions = predictions[mask2]
    return predictions


def draw_bbox_cuda(img, predictions, class_names):
    predictions = sanitize_predictions(predictions)

    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(img)

    for box in predictions:
        xyxy = box[:4]
        conf = float(box[4])
        cls = int(box[5]) if not np.isnan(box[5]) and box[5] < len(class_names) else -1
        if cls < 0 or cls >= len(class_names):
            continue  # skip invalid class

        label = f"{class_names[cls]} {conf:.2f}"
        cv2.rectangle(
            img,
            (int(round(xyxy[0])), int(round(xyxy[1]))),
            (int(round(xyxy[2])), int(round(xyxy[3]))),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img,
            label,
            (int(round(xyxy[0])), int(round(xyxy[1]) - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
    return img


# =====================
# YOLOv8 Box Decode
# =====================
class BoxDecoderYOLOv8:
    def __init__(self, stride, conf_thres, reg_max=16):
        self.stride = stride
        self.conf_thres = conf_thres
        self.reg_max = reg_max

    def __call__(self, feats: List[np.ndarray], step: int = 2):
        feats_box, feats_cls = feats[0::step], feats[1::step]
        feats_extra = None
        if step == 3:
            feats_extra = feats[2::step]

        out_boxes_batched = yolov8_box_decode(
            self.stride,
            self.conf_thres,
            self.reg_max,
            feats_box,
            feats_cls,
            feats_extra,
        )
        return out_boxes_batched


# =====================
# object_detection_anchor_decoder (GPU)
# =====================
class object_detection_anchor_decoder:
    def __init__(
        self,
        model_name: str,
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
        anchors: Union[List[List[int]], None] = None,
        use_tracker: bool = True,
    ):
        self.iou_thres = iou_thres
        self.tracker = None if not use_tracker else self._dummy_tracker
        self.anchors = anchors
        self.stride = np.array([(2 ** (i + 3)) for i in range(3)], dtype=np.float32)
        self.conf_thres = conf_thres
        self.box_decoder = BoxDecoderYOLOv8(
            self.stride, self.conf_thres
        )  # <- 여전히 C++ 경유

    def __call__(self, model_outputs, contexts, org_input_shape):
        boxes_dec = self.box_decoder(model_outputs)
        outputs = non_max_suppression_gpu(boxes_dec, self.iou_thres)
        predictions = []
        ratio, dwdh = contexts["ratio"], contexts["pad"]
        for prediction in outputs:
            if prediction.shape[0] == 0:
                predictions.append(prediction)
                continue
            prediction_t = torch.from_numpy(prediction).cuda()
            prediction_t[:, :4] = scale_coords_gpu(
                prediction_t[:, :4], ratio, dwdh, org_input_shape
            )
            pred_np = prediction_t.cpu().numpy()
            pred_np = sanitize_predictions(pred_np)  # <-- NaN / 잘못된 값 제거
            predictions.append(pred_np)
        return predictions

    def _dummy_tracker(self, x):
        return x


# =====================
# ObjDetPostprocess (GPU)
# =====================
class ObjDetPostprocess:
    def __init__(
        self, model_name: str, model_cfg, class_names, use_tracking: bool = True
    ):
        model_cfg.update({"use_tracker": use_tracking})
        # 반드시 인스턴스화
        self.postprocess_func = object_detection_anchor_decoder(model_name, **model_cfg)
        self.class_names = class_names

    def __call__(
        self, outputs: List[np.ndarray], contexts: Dict[str, float], img: np.ndarray
    ) -> np.ndarray:
        # 여기서 self.postprocess_func는 인스턴스이므로 __call__ 실행 가능
        predictions = self.postprocess_func(outputs, contexts, img.shape[:2])
        assert len(predictions) == 1, f"{len(predictions)} != 1"

        predictions = predictions[0]
        if predictions.shape[0] == 0:
            return img.astype(np.uint8)

        bboxed_img = draw_bbox_cuda(img.astype(np.uint8), predictions, self.class_names)
        return bboxed_img


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
            use_tracking=False,
        )
        self.input_shape = cfg["input_shape"]
        self.result_path = output_path
        self.input_path = input_path

        os.makedirs(self.result_path, exist_ok=True)

    async def load(self):
        await self.model.load()

    async def unload(self):
        await self.model.unload()

    def _get_video_resolution(self, video_path):
        """Get width and height using ffprobe (no OpenCV VideoCapture)"""
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            video_path,
        ]
        out = subprocess.check_output(cmd).decode("utf-8")
        info = json.loads(out)
        width = info["streams"][0]["width"]
        height = info["streams"][0]["height"]
        return width, height

    async def process(self, video_path):
        width, height = self._get_video_resolution(video_path)
        frame_size = width * height * 3

        ffmpeg_cmd = [
            "ffmpeg",
            "-hwaccel",
            "cuda",
            "-i",
            video_path,
            "-f",
            "image2pipe",
            "-pix_fmt",
            "bgr24",
            "-vcodec",
            "rawvideo",
            "-",
        ]
        process = subprocess.Popen(
            ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8
        )

        frame_idx = 0
        while True:
            raw_frame = process.stdout.read(frame_size)
            if len(raw_frame) != frame_size:
                break

            frame_bgr = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))

            input_, context = self.preprocessor(frame_bgr)
            output = await self.model.predict(input_)

            out = await asyncio.to_thread(
                self.postprocessor, output, context, frame_bgr
            )

            # 프레임 저장 (OpenCV)
            cv2.imwrite(
                os.path.join(
                    self.result_path,
                    f"{os.path.basename(video_path)}_{frame_idx:05d}.jpg",
                ),
                out,
            )
            frame_idx += 1

        process.stdout.close()
        process.wait()

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
    runner = WarboyRunner(
        model=cfg["onnx_i8_path"],
        input_path=input_video,
        output_path="/home/jisoo/warboy-vision-models/datasets/test_ffmpeg",
        runner_info={
            "iou_thres": cfg["iou_thres"],
            "conf_thres": cfg["conf_thres"],
            "anchors": [None],
        },
    )

    results, infer_time = asyncio.run(inference(runner))
    print(f"Inference done: {results}, Time taken: {infer_time:.2f}s")
