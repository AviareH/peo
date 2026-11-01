"""Pretrained ADE20K segmentation. Labels describe appearance, not safe clearance."""

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from drawing import draw_text

from model_runtime import select_device, synchronize

MODELS = {
    "b0": ("nvidia/segformer-b0-finetuned-ade-512-512",
           "489d5cd81a0b59fab9b7ea758d3548ebe99677da"),
    "b2": ("nvidia/segformer-b2-finetuned-ade-512-512",
           "de01bae28967510f9ddd496c60a969357195400c"),
}
COLOURS = {"floor": (60, 210, 60), "wall": (220, 130, 40),
           "other": (50, 60, 230), "unknown": (0, 210, 240)}


def group_predictions(labels, confidence, id2label, threshold):
    """Resolve model label names rather than assuming dataset-specific indices."""
    names = {int(key): value.strip().lower() for key, value in id2label.items()}
    floor_ids = [key for key, name in names.items() if name in {"floor", "rug"}]
    wall_ids = [key for key, name in names.items() if name == "wall"]
    if not floor_ids or not wall_ids:
        raise ValueError("Model label map must contain floor and wall classes.")
    valid = np.isfinite(confidence) & (confidence >= threshold)
    floor = np.isin(labels, floor_ids) & valid
    wall = np.isin(labels, wall_ids) & valid
    return {"floor": floor, "wall": wall, "other": valid & ~(floor | wall),
            "unknown": ~valid}


class SemanticDetector:
    def __init__(self, threshold=.5, device="auto", offline=False, model_name="b0"):
        if model_name not in MODELS:
            raise ValueError(f"Unknown semantic model: {model_name}. Choose b0 or b2.")
        self.model_name = model_name
        self.model_id, self.model_revision = MODELS[model_name]
        try:
            import torch
            from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        except ImportError as exc:
            raise ValueError("Semantic dependencies missing. Install vision/requirements.txt.") from exc
        self.torch = torch
        self.threshold = threshold
        device = select_device(torch, device)
        self.device = device
        cache = Path(__file__).parent / "models"
        print(f"Loading {self.model_id} on {device}. First use downloads pretrained weights.", flush=True)
        options = dict(cache_dir=str(cache), local_files_only=offline, revision=self.model_revision)
        try:
            self.processor = SegformerImageProcessor.from_pretrained(self.model_id, **options)
            # The pinned official B2 revision only publishes pytorch_model.bin.
            # Do not depend on transient Hub safetensors conversion revisions.
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                self.model_id, use_safetensors=self.model_name != "b2",
                weights_only=True, **options).to(device).eval()
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Cannot load SegFormer: {exc}. Check internet/cache; "
                             "--offline requires an earlier successful download.") from exc
        self.id2label = self.model.config.id2label
        group_predictions(np.zeros((1, 1)), np.zeros((1, 1)), self.id2label, threshold)

    def metadata(self):
        return dict(model=self.model_id, model_name=self.model_name,
                    revision=getattr(self.model.config, "_commit_hash", None),
                    device=self.device, confidence_threshold=self.threshold,
                    floor_labels=["floor", "rug"], torch=self.torch.__version__)

    def synchronize(self):
        synchronize(self.torch, self.device)

    def process(self, frame):
        self.synchronize()
        started = perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.mean() < 5 or gray.mean() > 250:
            masks = {key: np.full(gray.shape, key == "unknown", dtype=bool) for key in COLOURS}
            return dict(masks=masks, confidence=np.zeros(gray.shape, np.float32),
                        status="UNUSABLE FRAME", boxes=[], processing_ms=(perf_counter()-started)*1000)
        inputs = self.processor(images=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
            logits = self.torch.nn.functional.interpolate(
                logits, size=frame.shape[:2], mode="bilinear", align_corners=False)
            confidence, labels = logits.softmax(dim=1).max(dim=1)
        confidence = confidence[0].cpu().numpy()
        labels = labels[0].cpu().numpy()
        masks = group_predictions(labels, confidence, self.id2label, self.threshold)
        self.synchronize()
        return dict(masks=masks, confidence=confidence, boxes=[],
                    status=f"SEGFORMER {self.model_name.upper()} ESTIMATES",
                    processing_ms=(perf_counter()-started)*1000)


def annotate_semantic(frame, result):
    classes = np.zeros_like(frame)
    for key, colour in COLOURS.items():
        classes[result["masks"][key]] = colour
    overlay = cv2.addWeighted(frame, .65, classes, .35, 0)
    for key in ("floor", "wall"):
        contours, _ = cv2.findContours(result["masks"][key].astype(np.uint8),
                                      cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, COLOURS[key], 1)
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 54), (0, 0, 0), -1)
    draw_text(overlay, f"{result['status']} | {result['processing_ms']:.1f} ms", (8, 22))
    draw_text(overlay, "Green: floor | Blue: wall | Red: other | Yellow: uncertain", (8, 45), .45)
    confidence = cv2.applyColorMap((np.clip(result["confidence"], 0, 1)*255).astype(np.uint8),
                                   cv2.COLORMAP_VIRIDIS)
    return overlay, classes, confidence
