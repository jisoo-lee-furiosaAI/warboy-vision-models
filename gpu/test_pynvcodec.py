import asyncio
import os
import time
from glob import glob

import bbox_cuda  # <- CUDA 커널 모듈
import cv2
import numpy as np
import PyNvCodec as nvc
import PytorchNvCodec as pnvc
import torch
import torch.nn.functional as F
import torchvision
from furiosa.server.model import FuriosaRTModel, FuriosaRTModelConfig

from src.warboy.cfg import get_model_params_from_cfg
from src.warboy.yolo.cbox_decode import yolov8_box_decode

cfg = get_model_params_from_cfg(
    "/home/jisoo/warboy-vision-models/datasets/cfg/yolov8m.yaml"
)


# -------- Letterbox GPU ----------
def letterbox_torch(img: torch.Tensor, new_shape=(640, 640)):
    _, h, w = img.shape
    ratio = min(new_shape[0] / h, new_shape[1] / w)
    new_size = (int(h * ratio), int(w * ratio))
    img = F.interpolate(
        img.unsqueeze(0), size=new_size, mode="bilinear", align_corners=False
    )
    pad_h = new_shape[0] - new_size[0]
    pad_w = new_shape[1] - new_size[1]
    img = F.pad(img, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
    return img.squeeze(0), ratio, (pad_w // 2, pad_h // 2)


# -------- NMS & scale (GPU) ----------
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


def non_max_suppression_gpu(prediction: list, iou_thres: float = 0.45):
    output = []
    for x in prediction:
        t = torch.from_numpy(x).cuda()
        boxes, scores = t[:, :4], t[:, 4]
        keep = torchvision.ops.nms(boxes, scores, iou_thres)
        output.append(t[keep])
    return output


# -------- YOLOv8 Box Decoder ----------
class BoxDecoderYOLOv8:
    def __init__(self, stride, conf_thres, reg_max=16):
        self.stride = stride
        self.conf_thres = conf_thres
        self.reg_max = reg_max

    def __call__(self, feats, step=2):
        feats_box, feats_cls = feats[0::step], feats[1::step]
        feats_extra = None
        if step == 3:
            feats_extra = feats[2::step]
        return yolov8_box_decode(
            self.stride,
            self.conf_thres,
            self.reg_max,
            feats_box,
            feats_cls,
            feats_extra,
        )


class object_detection_anchor_decoder:
    def __init__(self, model_name, conf_thres=0.25, iou_thres=0.7, anchors=None):
        self.iou_thres = iou_thres
        self.anchors = anchors
        self.stride = np.array([(2 ** (i + 3)) for i in range(3)], dtype=np.float32)
        self.conf_thres = conf_thres
        self.box_decoder = BoxDecoderYOLOv8(self.stride, self.conf_thres)

    def __call__(self, model_outputs, contexts, org_input_shape):
        boxes_dec = self.box_decoder(model_outputs)
        outputs = non_max_suppression_gpu(boxes_dec, self.iou_thres)
        predictions = []
        ratio, dwdh = contexts["ratio"], contexts["pad"]
        for prediction in outputs:
            if prediction.shape[0] == 0:
                predictions.append(prediction)
                continue
            prediction[:, :4] = scale_coords_gpu(
                prediction[:, :4], ratio, dwdh, org_input_shape
            )
            predictions.append(prediction)
        return predictions


# -------- GPU Draw (CUDA Kernel) ----------
def draw_bbox_cuda(image: torch.Tensor, boxes: torch.Tensor, color=(0, 255, 0)):
    if boxes.numel() == 0:
        return image
    bbox_cuda.draw_boxes_cuda(
        image, boxes[:, :4].contiguous(), color[2], color[1], color[0]
    )
    return image


# -------- Runner ----------
class WarboyRunnerGPUCUDA:
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
        self.postprocessor = object_detection_anchor_decoder(
            cfg["model_name"], **runner_info
        )
        self.input_path = input_path
        self.result_path = output_path
        os.makedirs(self.result_path, exist_ok=True)

        self.nvDec = nvc.PyNvDecoder(input_path, 0)
        self.to_rgb = nvc.PySurfaceConverter(
            self.nvDec.Width(),
            self.nvDec.Height(),
            nvc.PixelFormat.NV12,
            nvc.PixelFormat.RGB,
            0,
        )

    async def load(self):
        await self.model.load()

    async def unload(self):
        await self.model.unload()

    async def process(self):
        frame_idx = 0
        while True:
            surf = self.nvDec.DecodeSingleSurface()
            if surf.Empty():
                break
            rgb_surf = self.to_rgb.Execute(surf)
            frame_tensor = pnvc.makefromSurface(rgb_surf, "cuda").byte()  # (3,H,W)

            # Letterbox (GPU)
            input_tensor, ratio, dwdh = letterbox_torch(
                frame_tensor.float() / 255.0, (640, 640)
            )
            input_npu = (input_tensor * 255).byte().unsqueeze(0).cpu().numpy()

            outputs = await self.model.predict(input_npu)

            # 후처리 (GPU)
            predictions = self.postprocessor(
                outputs, {"ratio": ratio, "pad": dwdh}, frame_tensor.shape[1:]
            )
            predictions = predictions[0]

            out_frame = draw_bbox_cuda(frame_tensor.clone(), predictions)
            out_frame_cpu = out_frame.permute(1, 2, 0).cpu().numpy()
            cv2.imwrite(
                os.path.join(self.result_path, f"frame_{frame_idx:05d}.jpg"),
                out_frame_cpu,
            )
            frame_idx += 1

    async def run(self):
        await self.process()
        return f"Processed video: {self.input_path}"


async def inference(runner):
    await runner.load()
    t1 = time.time()
    results = await runner.run()
    t2 = time.time()
    await runner.unload()
    return results, t2 - t1


if __name__ == "__main__":
    input_video = "/home/jisoo/warboy-vision-models/datasets/video/high/v_04601.mp4"
    runner = WarboyRunnerGPUCUDA(
        model=cfg["onnx_i8_path"],
        input_path=input_video,
        output_path="/home/jisoo/warboy-vision-models/datasets/test_cuda_kernel",
        runner_info={
            "iou_thres": cfg["iou_thres"],
            "conf_thres": cfg["conf_thres"],
            "anchors": [None],
        },
    )
    results, infer_time = asyncio.run(inference(runner))
    print(f"Inference done: {results}, Time taken: {infer_time:.2f}s")
