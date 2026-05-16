import ctypes
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class AlignmentProfile:
    enabled: bool = True
    shift_x: float = 0.0
    shift_y: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    mirror_x: bool = False
    mirror_y: bool = False
    hole_fill_kernel: int = 0
    prefer_native_mapper: bool = True


class RgbDepthAligner:
    DEFAULT_PROFILES = {
        'kinect_v1': AlignmentProfile(
            enabled=True,
            shift_x=0.0,
            shift_y=0.0,
            scale_x=1.0,
            scale_y=1.0,
            hole_fill_kernel=0,
            prefer_native_mapper=False,
        ),
        'kinect_v2': AlignmentProfile(
            enabled=True,
            shift_x=0.0,
            shift_y=0.0,
            scale_x=1.0,
            scale_y=1.0,
            hole_fill_kernel=5,
            prefer_native_mapper=True,
        ),
    }

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.profile_file = self.base_dir / 'data' / 'kinect_alignment_profiles.json'
        self._profiles_cache = None
        self._profiles_mtime = None
        self._v2_color_space_buffer = None
        self._v2_color_space_view = None
        self._v2_color_space_count = 0

    def default_payload(self):
        return {
            key: asdict(profile)
            for key, profile in self.DEFAULT_PROFILES.items()
        }

    def _coerce_profile(self, payload):
        default = AlignmentProfile()
        if not isinstance(payload, dict):
            return default
        safe = {}
        for field_name in AlignmentProfile.__dataclass_fields__:
            value = payload.get(field_name, getattr(default, field_name))
            if field_name in {'enabled', 'mirror_x', 'mirror_y', 'prefer_native_mapper'}:
                safe[field_name] = bool(value)
            elif field_name == 'hole_fill_kernel':
                try:
                    safe[field_name] = max(0, int(value))
                except Exception:
                    safe[field_name] = getattr(default, field_name)
            else:
                try:
                    safe[field_name] = float(value)
                except Exception:
                    safe[field_name] = getattr(default, field_name)
        return AlignmentProfile(**safe)

    def _load_profiles(self):
        if not self.profile_file.exists():
            return self.default_payload()

        try:
            mtime = self.profile_file.stat().st_mtime
        except OSError:
            return self.default_payload()

        if self._profiles_cache is not None and self._profiles_mtime == mtime:
            return self._profiles_cache

        try:
            with self.profile_file.open('r', encoding='utf-8') as profile_file:
                payload = json.load(profile_file)
        except Exception:
            payload = self.default_payload()

        if not isinstance(payload, dict):
            payload = self.default_payload()

        merged = self.default_payload()
        for key, value in payload.items():
            if key in merged and isinstance(value, dict):
                merged[key] = {
                    **merged[key],
                    **value,
                }

        self._profiles_cache = merged
        self._profiles_mtime = mtime
        return merged

    def get_profile(self, backend_name):
        profiles = self._load_profiles()
        payload = profiles.get(backend_name, self.default_payload().get(backend_name, {}))
        return self._coerce_profile(payload)

    def _apply_profile(self, frame, target_size, profile, is_depth):
        if frame is None:
            return None

        target_width, target_height = target_size
        if target_width <= 0 or target_height <= 0:
            return frame

        interpolation = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
        aligned = frame
        if aligned.shape[1] != target_width or aligned.shape[0] != target_height:
            aligned = cv2.resize(aligned, (target_width, target_height), interpolation=interpolation)

        if not profile.enabled:
            return aligned

        if profile.mirror_x:
            aligned = cv2.flip(aligned, 1)
        if profile.mirror_y:
            aligned = cv2.flip(aligned, 0)

        matrix = np.array(
            [
                [
                    float(profile.scale_x),
                    0.0,
                    ((1.0 - float(profile.scale_x)) * (target_width - 1) * 0.5) + float(profile.shift_x),
                ],
                [
                    0.0,
                    float(profile.scale_y),
                    ((1.0 - float(profile.scale_y)) * (target_height - 1) * 0.5) + float(profile.shift_y),
                ],
            ],
            dtype=np.float32,
        )
        aligned = cv2.warpAffine(
            aligned,
            matrix,
            (target_width, target_height),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        if is_depth and int(profile.hole_fill_kernel) >= 2:
            kernel_size = int(profile.hole_fill_kernel)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dilated = cv2.dilate(aligned, kernel, iterations=1)
            aligned = np.where(aligned > 0, aligned, dilated).astype(frame.dtype, copy=False)

        return aligned

    def align_v1_depth(self, depth_raw, color_shape):
        if depth_raw is None:
            return None
        height, width = color_shape[:2]
        profile = self.get_profile('kinect_v1')
        return self._apply_profile(depth_raw, (width, height), profile, is_depth=True)

    def _ensure_v2_color_space_buffer(self, point_count, pykinect_v2_module):
        if self._v2_color_space_buffer is not None and self._v2_color_space_count == point_count:
            return self._v2_color_space_buffer, self._v2_color_space_view

        if pykinect_v2_module is None:
            return None, None

        buffer = (pykinect_v2_module._ColorSpacePoint * point_count)()
        view = np.ctypeslib.as_array(buffer).view(np.float32).reshape(point_count, 2)
        self._v2_color_space_buffer = buffer
        self._v2_color_space_view = view
        self._v2_color_space_count = point_count
        return buffer, view

    def _map_v2_depth_to_color(self, runtime, depth_raw, pykinect_v2_module=None):
        if runtime is None or depth_raw is None:
            return None

        mapper = getattr(runtime, '_mapper', None)
        if mapper is None:
            return None

        depth_frame = np.ascontiguousarray(depth_raw.astype(np.uint16, copy=False))
        point_count = int(depth_frame.size)
        if point_count <= 0:
            return None

        color_height = int(getattr(runtime.color_frame_desc, 'Height', 0) or 0)
        color_width = int(getattr(runtime.color_frame_desc, 'Width', 0) or 0)
        if color_height <= 0 or color_width <= 0:
            return None

        color_space_buffer, color_space_view = self._ensure_v2_color_space_buffer(point_count, pykinect_v2_module)
        if color_space_buffer is None or color_space_view is None:
            return None
        mapper.MapDepthFrameToColorSpace(
            point_count,
            depth_frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort)),
            point_count,
            color_space_buffer,
        )

        xs = np.rint(color_space_view[:, 0]).astype(np.int32, copy=False)
        ys = np.rint(color_space_view[:, 1]).astype(np.int32, copy=False)
        depth_values = depth_frame.reshape(-1)
        valid = (
            np.isfinite(color_space_view[:, 0])
            & np.isfinite(color_space_view[:, 1])
            & (depth_values > 0)
            & (xs >= 0)
            & (ys >= 0)
            & (xs < color_width)
            & (ys < color_height)
        )
        if not np.any(valid):
            return None

        linear_index = (ys[valid] * color_width) + xs[valid]
        valid_depths = depth_values[valid]

        order = np.lexsort((valid_depths, linear_index))
        linear_index = linear_index[order]
        valid_depths = valid_depths[order]
        keep = np.ones(linear_index.shape[0], dtype=bool)
        if linear_index.shape[0] > 1:
            keep[1:] = linear_index[1:] != linear_index[:-1]

        aligned = np.zeros(color_width * color_height, dtype=np.uint16)
        aligned[linear_index[keep]] = valid_depths[keep]
        return aligned.reshape((color_height, color_width))

    def align_v2_depth(self, runtime, depth_raw, color_shape, pykinect_v2_module=None):
        if depth_raw is None:
            return None

        height, width = color_shape[:2]
        profile = self.get_profile('kinect_v2')
        aligned = None
        if profile.prefer_native_mapper:
            try:
                aligned = self._map_v2_depth_to_color(runtime, depth_raw, pykinect_v2_module=pykinect_v2_module)
            except Exception:
                aligned = None

        if aligned is None:
            aligned = depth_raw
        return self._apply_profile(aligned, (width, height), profile, is_depth=True)
