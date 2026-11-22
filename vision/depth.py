"""Optional relative inverse-depth inference and diagnostic visualization."""

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from drawing import draw_text, title_panel

from model_runtime import select_device, synchronize

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
MODEL_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"


class DepthEstimator:
    def __init__(self, device="auto", offline=False):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise ValueError("Depth dependencies missing; install vision/requirements.txt.") from exc
        self.torch = torch
        device = select_device(torch, device)
        self.device = device
        options = dict(cache_dir=str(Path(__file__).parent / "models"),
                       revision=MODEL_REVISION, local_files_only=offline)
        print(f"Loading {MODEL_ID} on {device}. First use downloads weights.", flush=True)
        try:
            self.processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False, **options)
            self.model = AutoModelForDepthEstimation.from_pretrained(
                MODEL_ID, use_safetensors=True, weights_only=True, **options).to(device).eval()
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Cannot load depth model: {exc}. Check internet/cache; "
                             "--offline requires a previous download.") from exc

    def metadata(self):
        return dict(model=MODEL_ID, revision=MODEL_REVISION, device=self.device,
                    units="relative inverse depth; larger is nearer", affects_path=False)

    def synchronize(self):
        synchronize(self.torch, self.device)

    def process(self, frame):
        self.synchronize()
        started = perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.mean() < 5 or gray.mean() > 250:
            return dict(values=np.full(gray.shape, np.nan, np.float32),
                        processing_ms=(perf_counter() - started) * 1000, status="UNUSABLE FRAME")
        inputs = self.processor(images=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            prediction = self.model(**inputs).predicted_depth
            values = self.torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=frame.shape[:2], mode="bicubic", align_corners=False)
        values = values[0, 0].cpu().numpy()
        self.synchronize()
        return dict(values=values, processing_ms=(perf_counter() - started) * 1000,
                    status="RELATIVE DEPTH")


def depth_summary(values):
    finite = values[np.isfinite(values)]
    return dict(valid_fraction=round(float(np.isfinite(values).mean()), 5),
                relative_min=float(finite.min()) if finite.size else None,
                relative_max=float(finite.max()) if finite.size else None)


class DepthView:
    """Lock display limits on first informative frame; never normalize each frame."""
    def __init__(self):
        self.limits = None

    def render(self, values):
        valid = np.isfinite(values)
        if self.limits is None and valid.any():
            low, high = np.percentile(values[valid], [2, 98])
            if high - low > 1e-6:
                self.limits = (float(low), float(high))
        image = np.zeros(values.shape, np.uint8)
        if self.limits is not None:
            low, high = self.limits
            image[valid] = (np.clip((values[valid] - low) / (high - low), 0, 1) * 255).astype(np.uint8)
        panel = cv2.applyColorMap(image, cv2.COLORMAP_INFERNO)
        panel[~valid] = 0
        title = "Relative depth: dark FAR / bright NEAR (not metres)"
        if not valid.any():
            title = "Depth unavailable / unusable frame"
        elif self.limits is None:
            title = "Flat depth estimate: no display contrast"
        return title_panel(panel, title)


def floor_panel(floor_mask):
    panel = cv2.cvtColor((floor_mask != 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    return title_panel(panel, "Path floor mask: segmentation only")
