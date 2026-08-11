from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
import sys
from typing import Dict, List, Optional
import numpy as np
from PIL import Image
import torch
from transformers import AutoConfig
from transformers.models.sam3.processing_sam3 import Sam3Processor
from transformers.models.sam3.modeling_sam3 import Sam3Model
from pycocotools import mask as coco_mask
from huggingface_hub import login
from torch import nn
from ultralytics import YOLO

from utils.config import BINARY_CLASS_THRESHOLD, DEFAULT_MODEL_CACHE_DIR
from utils.sliced_inference import run_sliced_inference
from binary_classificator.extract_embeddings import load_feature_extractor

@dataclass
class Detection:
    box: List[float]
    score: float
    label: str
    mask_rle: Optional[Dict] = field(default=None)

class BaseDetector(ABC):
    def __init__(self, model_name: str, supports_mask: bool = False) -> None:
        self.model_name = model_name
        self.supports_mask = supports_mask

    @abstractmethod
    def predict(
        self,
        image: Image.Image,
        prompt: List[str],
        negative_prompt: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Detection]:
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_name!r}, "
            f"supports_mask={self.supports_mask})"
        )

def _default_cache_dir() -> str:
    """Return the model cache directory, overridable via ``$MODEL_CACHE_DIR``."""
    return os.environ.get("MODEL_CACHE_DIR", DEFAULT_MODEL_CACHE_DIR)


def _warn_if_cpu(device, model_name: str) -> None:
    """Warn when a zero-shot detector runs on CPU (they are GPU-scale models)."""
    if str(device) == "cpu":
        print(
            f"Warning: {model_name} is running on CPU (no CUDA detected); "
            "this will be slow. YOLO baselines are the CPU-friendly option "
            "(see 'Hardware expectations' in README.md)."
        )


def mask_to_rle(binary_mask: np.ndarray) -> Dict:
    rle = coco_mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    return rle

def rle_to_mask(rle: Dict) -> np.ndarray:
    return coco_mask.decode(rle).astype(np.uint8)

@dataclass
class BinaryClassification:
    label: str
    score: float

class BaseBinaryClassificator(ABC):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def predict(self, image: Image.Image) -> BinaryClassification:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r})"

class BaseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout_rate=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim // 4, 1)
            # nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        return self.net(x).squeeze(1)

class MLPBinaryClassificator(BaseBinaryClassificator):
    def __init__(self, extractor_name: str, pth_weights_path: str):
        super().__init__(model_name=f"{extractor_name}_mlp")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.extractor = load_feature_extractor(extractor_name)
        
        checkpoint = torch.load(pth_weights_path, map_location=self.device, weights_only=True)
        input_dim = checkpoint["input_dim"]
        
        self.classifier = BaseMLP(input_dim=input_dim).to(self.device)
        self.classifier.load_state_dict(checkpoint["model_state_dict"])
        
        self.classifier.eval()

    def predict(self, image: Image.Image) -> BinaryClassification:
        embedding_tensor = self.extractor(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logit = self.classifier(embedding_tensor)
            score = torch.sigmoid(logit).item()
            
        # Label is informational only: "joint"/"mean" score modes combine
        # the raw probability (score) and never read this label
        label = "insect" if score >= BINARY_CLASS_THRESHOLD else "background"
        
        return BinaryClassification(label=label, score=score)

class SAM3Detector(BaseDetector):
    def __init__(
        self,
        model_name: str = "facebook/sam3",
        cache_dir: str = _default_cache_dir(),
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        super().__init__(model_name=model_name, supports_mask=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _warn_if_cpu(self.device, model_name)

        if hf_token:
            login(token=hf_token)

        # "facebook/sam3" is published on the hub as Sam3VideoModel (tracking);
        # its forward requires the video inference-session API. We only need
        # the inner text-prompted detector, Sam3Model.
        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = Sam3Model(config.detector_config).to(self.device)
        self._load_detector_weights(model_name, cache_dir, hf_token)

        # AutoProcessor resolves "facebook/sam3" to Sam3VideoProcessor, whose
        # __call__ does not accept `text` (it forwards all kwargs to the image
        # processor, which rejects them). Sam3Processor tokenizes `text`.
        self.processor = Sam3Processor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model.eval()

    def _load_detector_weights(
        self, model_name: str, cache_dir: str, token: Optional[str] = None
    ) -> None:
        """Load the Sam3Model weights from the Sam3VideoModel checkpoint.

        The hub checkpoint stores the detector under the ``detector_model.``
        prefix, alongside a tracker and a tracker neck we do not need.
        """
        from safetensors import safe_open
        from transformers.utils import cached_file

        weight_file = cached_file(
            model_name, "model.safetensors", cache_dir=cache_dir, token=token
        )
        state_dict = {}
        with safe_open(weight_file, framework="pt") as f:
            for key in f.keys():
                if key.startswith("detector_model."):
                    state_dict[key[len("detector_model."):]] = f.get_tensor(key)
        self.model.load_state_dict(state_dict, strict=True)

    def _process_tile(
        self, tile: Image.Image, x1: int, y1: int,
        text: str, label: str, conf_threshold: float, mask_threshold: float,
        height: int, width: int, **kwargs,
    ) -> List[Detection]:
        """Run SAM3 on a single tile and return detections in full-image coords."""
        tile_w, tile_h = tile.size

        inputs = self.processor(
            images=tile, text=text, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)

        tile_results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=conf_threshold,
            mask_threshold=mask_threshold,
            target_sizes=[(tile_h, tile_w)],
        )[0]

        if len(tile_results["masks"]) == 0:
            return []

        raw: List[Detection] = []
        x2 = x1 + tile_w
        y2 = y1 + tile_h
        for i in range(len(tile_results["masks"])):
            # Place tile-local mask into full-size canvas
            full_mask = torch.zeros((height, width), dtype=torch.bool)
            full_mask[y1:y2, x1:x2] = tile_results["masks"][i][:tile_h, :tile_w].cpu()

            mask_np = full_mask.numpy().astype(np.uint8)
            ys, xs = np.where(mask_np)
            if len(ys) == 0:
                continue  # empty mask after placement

            raw.append(Detection(
                box=[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                score=tile_results["scores"][i].item(),
                label=label,
                mask_rle=mask_to_rle(mask_np),
            ))
        return raw

    def predict(
        self,
        image: Image.Image,
        prompt: List[str],
        slice_size: int = 1024,
        overlap: float = 0.3,
        conf_threshold: float = 0.2,
        mask_threshold: float = 0.5,
        iou_threshold: float = 0.3,
        **kwargs,
    ) -> List[Detection]:
        # backward compat: map old parameter name
        if "det_threshold" in kwargs:
            conf_threshold = kwargs.pop("det_threshold")

        text = ",".join(prompt)
        width, height = image.size

        with torch.no_grad():
            return run_sliced_inference(
                image, slice_size, overlap,
                process_tile=self._process_tile,
                iou_threshold=iou_threshold,
                disable_progress=kwargs.pop("disable_progress", False),
                text=text, label=text,
                conf_threshold=conf_threshold, mask_threshold=mask_threshold,
                height=height, width=width,
            )
    
class GroundingDINODetector(BaseDetector):
    def __init__(
        self,
        model_name: str = "rziga/mm_grounding_dino_large_all",
        cache_dir: str = _default_cache_dir(),
        device: Optional[str] = None,
    ) -> None:
        super().__init__(model_name=model_name, supports_mask=False)
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # type: ignore
        # from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor  # type: ignore

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _warn_if_cpu(self.device, model_name)
        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_name, cache_dir=cache_dir
        ).to(self.device)
        self.model.eval()

    def _process_tile(
        self, tile: Image.Image, x1: int, y1: int,
        text_labels: str, prompt: List[str],
        negative_prompt: Optional[List[str]],
        conf_threshold: float, text_threshold: float,
        **kwargs,
    ) -> List[Detection]:
        """Run GroundingDINO on a single tile, return detections in full-image coords."""
        tile_w, tile_h = tile.size

        inputs = self.processor(
            images=tile, text=text_labels, return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=conf_threshold,
            text_threshold=text_threshold,
            target_sizes=[(tile_h, tile_w)],
        )[0]

        if len(results["boxes"]) == 0:
            return []

        tile_boxes = results["boxes"].detach().cpu()
        offset = torch.tensor([x1, y1, x1, y1], dtype=tile_boxes.dtype)
        full_boxes = (tile_boxes + offset).tolist()

        raw: List[Detection] = []
        for box, score, label in zip(
            full_boxes, results["scores"].tolist(), results["text_labels"]
        ):
            if not label.strip():
                continue
            if not (any(p.lower() in label.lower() for p in prompt) or "insect" in label.lower()):
                continue
            if negative_prompt and any(np_item.lower() in label.lower() for np_item in negative_prompt):
                continue

            raw.append(Detection(box=box, score=score, label=label))

        return raw

    def predict(
        self,
        image: Image.Image,
        prompt: List[str],
        negative_prompt: Optional[List[str]] = None,
        conf_threshold: float = 0.1,
        iou_threshold: float = 0.1,
        slice_size: Optional[int] = None,
        overlap: float = 0.2,
        text_threshold: float = 0.1,
        **kwargs,
    ) -> List[Detection]:
        text_labels = ".".join(prompt) + "." + ".".join(negative_prompt) + "." if negative_prompt else ".".join(prompt) + "."

        print(f"Running GroundingDINO with prompt: {text_labels} and negative prompt: {negative_prompt}")

        with torch.no_grad():
            return run_sliced_inference(
                image, slice_size, overlap,
                process_tile=self._process_tile,
                iou_threshold=iou_threshold,
                text_labels=text_labels, prompt=prompt,
                negative_prompt=negative_prompt,
                conf_threshold=conf_threshold, text_threshold=text_threshold,
            )


class OwlDetector(BaseDetector):
    def __init__(
        self,
        model_name: str = "google/owlvit-large-patch14",
        cache_dir: str = _default_cache_dir(),
        device: Optional[str] = None,
    ) -> None:
        super().__init__(model_name=model_name, supports_mask=False)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _warn_if_cpu(self.device, model_name)

        # Resolve the processor/model classes from the checkpoint itself
        # (owlv2 vs owlvit): a hardcoded registry would silently fall back to
        # OwlViTForObjectDetection for unknown names, leaving the model with
        # randomly-initialized weights.
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # type: ignore
        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_name, cache_dir=cache_dir
        ).to(self.device)
        self.model.eval()

    def _process_tile(
        self, tile: Image.Image, x1: int, y1: int,
        texts: list, prompt: List[str], n_pos: int,
        conf_threshold: float, **kwargs,
    ) -> List[Detection]:
        """Run OWL on a single tile, return detections in full-image coords."""
        tile_w, tile_h = tile.size

        inputs = self.processor(
            text=texts, images=tile, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([[tile_h, tile_w]])
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, threshold=conf_threshold, target_sizes=target_sizes,
        )[0]

        if len(results["boxes"]) == 0:
            return []

        tile_boxes = results["boxes"].detach().cpu()
        offset = torch.tensor([x1, y1, x1, y1], dtype=tile_boxes.dtype)
        full_boxes = (tile_boxes + offset).tolist()

        raw: List[Detection] = []
        for box, score, label_idx in zip(
            full_boxes, results["scores"].tolist(), results["labels"].tolist()
        ):
            if label_idx >= n_pos:
                continue
            raw.append(Detection(box=box, score=score, label=prompt[label_idx]))

        return raw

    def predict(
        self,
        image: Image.Image,
        prompt: List[str],
        negative_prompt: Optional[List[str]] = None,
        conf_threshold: float = 0.1,
        iou_threshold: float = 0.5,
        slice_size: Optional[int] = None,
        overlap: float = 0.2,
        **kwargs,
    ) -> List[Detection]:
        all_labels = prompt + (negative_prompt or [])
        n_pos = len(prompt)
        texts = [[f"a photo of a {p}" for p in all_labels]]

        with torch.no_grad():
            return run_sliced_inference(
                image, slice_size, overlap,
                process_tile=self._process_tile,
                iou_threshold=iou_threshold,
                texts=texts, prompt=prompt, n_pos=n_pos,
                conf_threshold=conf_threshold,
            )

def _load_yolov5_torch_hub(model_name: str, device: torch.device):
    """Load a YOLOv5 model via ``torch.hub``.

    ``torch.hub.load('ultralytics/yolov5', …)`` internally imports a
    ``models`` module that collides with ``utils/models.py``.  We temporarily
    pop our entries from ``sys.modules``, load the model, then restore them.
    """
    _popped = {}
    for _key in ("utils", "models", "utils.models"):
        if _key in sys.modules:
            _popped[_key] = sys.modules.pop(_key)
    try:
        # trust_repo: skip the interactive "do you trust this repository?"
        # prompt (torch.hub would otherwise block on stdin on first use)
        model = torch.hub.load(
            "ultralytics/yolov5", "custom", path=model_name,
            verbose=False, trust_repo=True,
        )
        return model.to(device)
    finally:
        for _key, _mod in _popped.items():
            sys.modules[_key] = _mod


class YOLODetector(BaseDetector):
    def __init__(
        self, model_name: str,
        device: Optional[str] = None,
        slice_size: Optional[int] = None,
        overlap: float = 0.0,
        supports_mask: bool = False,
        use_ultralytics: bool = True,
    ):
        super().__init__(model_name=model_name, supports_mask=supports_mask)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.slice_size = slice_size
        self.overlap = overlap
        self.use_ultralytics = use_ultralytics

        if use_ultralytics:
            self.model = YOLO(model_name).to(self.device)
        else:
            self.model = _load_yolov5_torch_hub(model_name, self.device)

    def _process_tile(
        self, tile: Image.Image, x1: int, y1: int,
        conf_threshold: float, iou_threshold: float,
        height: int, width: int,
        **kwargs,
    ) -> List[Detection]:
        """Run YOLO on a single tile, return detections in full-image coords."""
        tile_w, tile_h = tile.size

        if self.use_ultralytics:
            return self._process_tile_ultralytics(tile, x1, y1, conf_threshold, iou_threshold, height, width, tile_w, tile_h)
        else:
            return self._process_tile_yolov5(tile, x1, y1, conf_threshold, iou_threshold, height, width)

    def _process_tile_ultralytics(
        self, tile, x1, y1, conf_threshold, iou_threshold, height, width, tile_w, tile_h,
    ) -> List[Detection]:
        results = self.model(tile, conf=conf_threshold, iou=iou_threshold, verbose=False)
        raw: List[Detection] = []
        for result in results:
            for idx, box in enumerate(result.boxes):
                tile_box = box.xyxy[0].tolist()
                full_box = [tile_box[0] + x1, tile_box[1] + y1, tile_box[2] + x1, tile_box[3] + y1]

                mask_rle = None
                if self.supports_mask and result.masks is not None and idx < len(result.masks):
                    tile_mask = result.masks[idx].data.cpu().numpy().squeeze().astype(np.uint8)
                    if tile_mask.shape != (tile_h, tile_w):
                        tile_mask = np.array(Image.fromarray(tile_mask).resize((tile_w, tile_h), Image.NEAREST))
                    full_mask = np.zeros((height, width), dtype=np.uint8)
                    full_mask[y1:y1 + tile_h, x1:x1 + tile_w] = tile_mask[:tile_h, :tile_w]
                    mask_rle = mask_to_rle(full_mask)

                raw.append(Detection(box=full_box, score=box.conf.item(),
                                    label=result.names[int(box.cls)], mask_rle=mask_rle))
        return raw

    def _process_tile_yolov5(
        self, tile, x1, y1, conf_threshold, iou_threshold, height, width,
    ) -> List[Detection]:
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        results = self.model(tile)
        predictions_df = results.pandas().xyxy[0]
        raw: List[Detection] = []
        for _, row in predictions_df.iterrows():
            full_box = [row['xmin'] + x1, row['ymin'] + y1, row['xmax'] + x1, row['ymax'] + y1]
            raw.append(Detection(box=full_box, score=row['confidence'],
                                label=row['name'], mask_rle=None))
        return raw

    def predict(
        self,
        image: Image.Image,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        slice_size: Optional[int] = None,
        overlap: Optional[float] = None,
        **kwargs,
    ) -> List[Detection]:
        # Allow predict-time override; fall back to constructor values
        _slice_size = slice_size if slice_size is not None else getattr(self, "slice_size", None)
        _overlap = overlap if overlap is not None else getattr(self, "overlap", 0.0)
        width, height = image.size

        with torch.no_grad():
            return run_sliced_inference(
                image, _slice_size, _overlap,
                process_tile=self._process_tile,
                iou_threshold=iou_threshold,
                conf_threshold=conf_threshold,
                height=height, width=width,
            )

class Bjerge2023Detector(YOLODetector):
    def __init__(self, model_name: str = "./models-cache/insect1201-bestF1-1280v5m6_bjerge2023.pt", device: Optional[str] = None, slice_size: Optional[int] = None, overlap: float = 0.0):
        super().__init__(model_name=model_name, device=device, slice_size=slice_size, overlap=overlap, use_ultralytics=False)

class Stark2023Detector(YOLODetector):
    def __init__(self, model_name: str = "./models-cache/yolov5_s_best_stark2023.pt", device: Optional[str] = None, slice_size: Optional[int] = None, overlap: float = 0.0):
        super().__init__(model_name=model_name, device=device, slice_size=slice_size, overlap=overlap, use_ultralytics=False)

class Marin2025Detector(YOLODetector):
    def __init__(self, model_name: str = "./models-cache/marin2025.pt", device: Optional[str] = None, slice_size: Optional[int] = None, overlap: float = 0.0):
        super().__init__(model_name=model_name, device=device, slice_size=slice_size, overlap=overlap, use_ultralytics=False)

class FlatBugDetector(YOLODetector):
    def __init__(self, model_name: str = "./models-cache/flat_bug_M.pt", device: Optional[str] = None, slice_size: Optional[int] = None, overlap: float = 0.0, supports_mask: bool = True):
        super().__init__(model_name=model_name, device=device, slice_size=slice_size, overlap=overlap, supports_mask=supports_mask, use_ultralytics=True)

class InsectsDCTDetector(YOLODetector):
    def __init__(self, model_name: str = "./models-cache/insectsDCT_insects6ColorExtend.pt", device: Optional[str] = None, slice_size: Optional[int] = None, overlap: float = 0.0):
        super().__init__(model_name=model_name, device=device, slice_size=slice_size, overlap=overlap, use_ultralytics=True)
