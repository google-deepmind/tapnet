"""TAPNextPP — high-level wrapper for TAPNext++ inference."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from tapnet.tapnext.tapnext_torch import TAPNext, TAPNextTrackingState
from tapnet.tapnextpp.votsp2026.utils import (
    display_to_model,
    make_query_tensor,
    model_to_display,
    preprocess_frame,
)


class TAPNextPP(nn.Module):
    """High-level TAPNext++ wrapper for frame-by-frame point tracking.

    Handles preprocessing (BGR frames, pixel coordinates) so callers don't
    need to know about model-space tensors.

    Typical online-tracking usage::

        model = TAPNextPP.from_checkpoint("ckpt.pt", device="cuda")

        # First frame: initialise tracking with query points
        query_xy = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
        positions, visible, state = model.track_frame(frame0, query_xy)

        # Subsequent frames: pass state
        for frame in video[1:]:
            positions, visible, state = model.track_frame(frame, state=state)
    """

    MODEL_SIZE: int = 256

    def __init__(
        self,
        inner: TAPNext,
        device: torch.device,
        input_resolution: int = MODEL_SIZE,
    ) -> None:
        super().__init__()
        self._model = inner
        self.device = device
        self.input_resolution = input_resolution

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path | None = None,
        device: torch.device | str = "cuda",
        half_precision: bool = False,
        compile_model: bool = False,
        input_resolution: int = MODEL_SIZE,
    ) -> "TAPNextPP":
        """Load a TAPNext++ checkpoint and return a ready-to-use wrapper.

        Args:
            path: Path to a ``.pt`` / ``.ckpt`` checkpoint. Required.
            device: Device to run inference on. Defaults to ``"cuda"``.
            half_precision: Convert weights to float16.
            compile_model: Wrap with ``torch.compile(mode="default")``.
            input_resolution: Pixel resolution frames are resized to before
                patch embedding. Use ``256`` for standard checkpoints or
                ``512`` for checkpoints fine-tuned at higher resolution.

        Returns:
            A ``TAPNextPP`` instance with ``eval()`` already called.
        """
        if path is None:
            raise ValueError("path must be provided explicitly.")

        if isinstance(device, str):
            device = torch.device(device)

        inner = TAPNext(image_size=(cls.MODEL_SIZE, cls.MODEL_SIZE))
        if input_resolution % inner.patch_size[0] != 0:
            raise ValueError(
                f"input_resolution ({input_resolution}) must be a multiple "
                f"of the patch size ({inner.patch_size[0]})."
            )
        ckpt = torch.load(path, map_location="cpu", weights_only=True)

        state_dict = ckpt.get("state_dict", ckpt)
        # strip Lightning-style "tapnext." prefix if present
        state_dict = {k.removeprefix("tapnext."): v for k, v in state_dict.items()}
        inner.load_state_dict(state_dict)

        inner = inner.to(device)
        inner.eval()

        if half_precision:
            inner = inner.half()

        if compile_model:
            inner = torch.compile(inner, mode="default")

        return cls(inner, device, input_resolution=input_resolution)

    @torch.no_grad()
    def track_frame(
        self,
        frame_bgr: np.ndarray,
        query_points_xy: np.ndarray | None = None,
        state: TAPNextTrackingState | None = None,
        autocast: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, TAPNextTrackingState]:
        """Process one frame and return tracking results in display coordinates.

        Args:
            frame_bgr: [H, W, 3] uint8 BGR frame (OpenCV layout).
            query_points_xy: [Q, 2] float32 array of [x, y] coordinates in
                display pixels. Must be provided when ``state`` is ``None``.
            state: Recurrent state returned by a previous call to this method.
            autocast: Use ``torch.amp.autocast`` on CUDA.

        Returns:
            positions_xy: [Q, 2] float32 [x, y] in display pixels.
            visible: [Q] bool array.
            state: Updated recurrent state.
        """
        if query_points_xy is None and state is None:
            raise ValueError("Either query_points_xy or state must be provided.")

        h, w = frame_bgr.shape[:2]
        frame_t = preprocess_frame(frame_bgr, self.device, self.input_resolution)

        q_t: torch.Tensor | None = None
        if query_points_xy is not None:
            model_pts = display_to_model(query_points_xy, h, w, self.MODEL_SIZE)
            q_t = make_query_tensor(model_pts, self.device)

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        ctx = (
            torch.amp.autocast("cuda", dtype=dtype)
            if (autocast and self.device.type == "cuda")
            else torch.amp.autocast("cpu", enabled=False)
        )

        with ctx:
            tracks, _, vis_logits, new_state = self._model(
                video=frame_t,
                query_points=q_t,
                state=state,
            )

        # tracks: [1, 1, Q, 2] in [y, x] model space → flip to [x, y] display
        tracks_xy = tracks[0, 0].cpu().float().numpy()[:, ::-1].copy()
        positions_xy = model_to_display(tracks_xy, h, w, self.MODEL_SIZE)
        visible = (vis_logits[0, 0, :, 0] > 0).cpu().numpy()

        return positions_xy, visible, new_state

    @torch.no_grad()
    def warmup(self, n_points: int = 64) -> None:
        """Run dummy forward passes to trigger ``torch.compile`` tracing."""
        if self.device.type != "cuda":
            return
        dummy_video = torch.zeros(
            1, 1, self.input_resolution, self.input_resolution, 3, device=self.device
        )
        dummy_queries = torch.zeros(1, n_points, 3, device=self.device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            _, _, _, state = self._model(video=dummy_video, query_points=dummy_queries)
            for _ in range(3):
                _, _, _, state = self._model(video=dummy_video, state=state)
        torch.cuda.synchronize()
