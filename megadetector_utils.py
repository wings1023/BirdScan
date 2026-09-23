"""Shared MegaDetector V6 loading and crop geometry helpers."""
from __future__ import annotations

import functools
import importlib
import importlib.metadata
import math
import os
import sys
import types
from pathlib import Path
from typing import Any

MODEL_VERSION = "MDV6-yolov10-e"


def load_detector(device: str) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("缺少 PyTorch 依赖。请先按当前设备安装兼容的 torch。") from exc
    try:
        detector_class = import_megadetector_v6_image_only()
    except Exception as exc:
        raise RuntimeError(f"无法仅加载 PyTorch-Wildlife 的 MegaDetector V6 图像模块：{exc}") from exc
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("当前 Python 环境的 PyTorch 没有可用的 MPS；请改用 --device cpu。")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 Python 环境的 PyTorch 没有可用的 CUDA；请改用 --device cpu。")
    from ultralytics.engine.predictor import BasePredictor
    import ultralytics.nn.tasks as ultralytics_tasks

    # PyTorch-Wildlife's YOLOV8Base resolves this official V6 weight by name.
    # Pass the already cached, fixed official path explicitly so this scope can
    # never opt arbitrary/user-supplied checkpoint paths into unrestricted load.
    checkpoint = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "MDV6-yolov10-e-1280.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"MegaDetector 官方 checkpoint 不存在，未尝试下载：{checkpoint}")
    checkpoint = checkpoint.resolve()

    original_setup_model = BasePredictor.setup_model
    original_torch_load = ultralytics_tasks.torch_load

    @functools.wraps(original_setup_model)
    def setup_model_on_requested_device(predictor: Any, model: Any, verbose: bool = True) -> None:
        predictor.args.device = device
        return original_setup_model(predictor, model, verbose=verbose)

    @functools.wraps(original_torch_load)
    def load_official_checkpoint(*args: Any, **kwargs: Any) -> Any:
        weight = args[0] if args else kwargs.get("f")
        if not isinstance(weight, (str, os.PathLike)) or Path(weight).resolve() != checkpoint:
            return original_torch_load(*args, **kwargs)
        kwargs["weights_only"] = False
        # PyTorch's force flag overrides even an explicit False. Limit its
        # removal to this trusted checkpoint call, then restore it exactly.
        force_flag = os.environ.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)
        try:
            return original_torch_load(*args, **kwargs)
        finally:
            if force_flag is not None:
                os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = force_flag

    BasePredictor.setup_model = setup_model_on_requested_device
    ultralytics_tasks.torch_load = load_official_checkpoint
    try:
        detector = detector_class(
            weights=str(checkpoint), device=device, pretrained=True, version=MODEL_VERSION
        )
    finally:
        BasePredictor.setup_model = original_setup_model
        ultralytics_tasks.torch_load = original_torch_load
    predictor = getattr(detector, "predictor", None)
    actual_device = getattr(predictor, "device", None)
    actual_type = getattr(actual_device, "type", str(actual_device).split(":", 1)[0])
    requested_type = device.split(":", 1)[0]
    print(f"MegaDetector predictor device: {actual_device}")
    if actual_type != requested_type:
        raise RuntimeError(f"MegaDetector requested {device!r}, initialized predictor.device={actual_device!r}.")
    return detector


def import_megadetector_v6_image_only() -> Any:
    try:
        distribution = importlib.metadata.distribution("PytorchWildlife")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("尚未安装 PytorchWildlife；请运行 python -m pip install PytorchWildlife") from exc
    version = distribution.version
    root = Path(distribution.locate_file("PytorchWildlife")).resolve()
    module_layout = root / "models" / "detection" / "ultralytics_based" / "megadetectorv6.py"
    if not module_layout.is_file():
        raise RuntimeError(f"PytorchWildlife {version} 中找不到预期图像检测模块：{module_layout}")
    package_paths = {
        "PytorchWildlife": root,
        "PytorchWildlife.models": root / "models",
        "PytorchWildlife.models.detection": root / "models" / "detection",
        "PytorchWildlife.models.detection.ultralytics_based": root / "models" / "detection" / "ultralytics_based",
        "PytorchWildlife.data": root / "data",
    }
    for name, path in package_paths.items():
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            package.__package__ = name
            sys.modules[name] = package
    module = importlib.import_module("PytorchWildlife.models.detection.ultralytics_based.megadetectorv6")
    return module.MegaDetectorV6


def _as_list(values: Any) -> list[Any]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    return list(values)


def extract_animals(result: dict[str, Any]) -> list[tuple[float, tuple[float, float, float, float]]]:
    detections = result["detections"]
    return [(float(conf), tuple(float(v) for v in box))
            for conf, class_id, box in zip(_as_list(detections.confidence),
                                           _as_list(detections.class_id), _as_list(detections.xyxy))
            if int(class_id) == 0]


def padded_box(xyxy: tuple[float, float, float, float], margin: float,
               image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
    return (max(0, math.floor(x1 - width * margin)), max(0, math.floor(y1 - height * margin)),
            min(image_width, math.ceil(x2 + width * margin)), min(image_height, math.ceil(y2 + height * margin)))


def safe_stem(path: Path) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem)
