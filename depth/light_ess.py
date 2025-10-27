from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import onnxruntime as ort


class LightESS:
    """Minimal wrapper around the light_ess ONNX model."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        providers: Optional[Iterable[str]] = None,
        session_options: Optional[ort.SessionOptions] = None,
    ) -> None:
        if model_path is None:
            model_path = Path(__file__).resolve().parent.parent / "weights" / "light_ess.onnx"
        if providers is None:
            providers = ort.get_available_providers()
        if session_options is None:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=list(providers),
        )
        self._input_names = [tensor.name for tensor in self._session.get_inputs()]
        self._output_names = [tensor.name for tensor in self._session.get_outputs()]
        self._layouts = self._infer_layouts()

    def __call__(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        left = self._prepare_image(np.asarray(left), self._layouts[0])
        right = self._prepare_image(np.asarray(right), self._layouts[1] if len(self._layouts) > 1 else self._layouts[0])

        feed = {_name: _input for _name, _input in zip(self._input_names, (left, right))}

        outputs = self._session.run(self._output_names, feed)
        disparity = outputs[0]
        confidence = outputs[1] if len(outputs) > 1 else np.ones_like(disparity, dtype=np.float32)
        return disparity, confidence

    def _infer_layouts(self) -> Tuple[str, ...]:
        layouts: list[str] = []
        for tensor in self._session.get_inputs():
            shape = tensor.shape
            layout = "unknown"
            if len(shape) == 4:
                if shape[-1] == 3:
                    layout = "NHWC"
                elif shape[1] == 3:
                    layout = "NCHW"
            layouts.append(layout)
        return tuple(layouts)

    def _prepare_image(self, image: np.ndarray, layout: str) -> np.ndarray:
        assert image.shape[0] == 288 and image.shape[1] == 480, f"LightESS expects images of shape (288, 480, 3), got {image.shape}"
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)

        image = (image - 0.5) / 0.5  # normalize using mean 0.5, std 0.5

        if layout == "NCHW" and image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        return image
