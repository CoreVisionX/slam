import cv2
import numpy as np
import torch

from registration.registration import (
    FeatureFrame,
    FramePair,
    MatchedFramePair,
    StereoDepthFrame,
)


class OrbMatcher:
    def __init__(
        self,
        num_features: int = 1000,
    ):
        self.num_features = num_features

        self.detector = cv2.ORB_create(nfeatures=self.num_features)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def detect_features(self, frames: list[StereoDepthFrame]) -> list[FeatureFrame]:
        feature_frames: list[FeatureFrame] = []

        for frame in frames:
            image = frame.left_rect
            if image.ndim == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            keypoints, descriptors = self.detector.detectAndCompute(gray, None)

            if keypoints is None:
                keypoints = []
            if descriptors is None:
                descriptors = np.zeros((0, 32), dtype=np.uint8)

            keypoints_np = (
                np.array([kp.pt for kp in keypoints], dtype=np.float32)
                if keypoints
                else np.zeros((0, 2), dtype=np.float32)
            )
            scores_np = (
                np.array([kp.response for kp in keypoints], dtype=np.float32)
                if keypoints
                else np.zeros((0,), dtype=np.float32)
            )
            scales_np = (
                np.array([kp.size for kp in keypoints], dtype=np.float32)
                if keypoints
                else np.zeros((0,), dtype=np.float32)
            )
            orientations_np = (
                np.array([kp.angle for kp in keypoints], dtype=np.float32)
                if keypoints
                else np.zeros((0,), dtype=np.float32)
            )

            u = keypoints_np[:, 1].astype(int)
            v = keypoints_np[:, 0].astype(int)

            if keypoints_np.size:
                height, width = frame.left_depth_xyz.shape[:2]
                u = np.clip(u, 0, height - 1)
                v = np.clip(v, 0, width - 1)

            if keypoints_np.size:
                keypoints_3d = frame.left_depth_xyz[u, v, :]
                keypoints_depth = frame.left_depth[u, v]
                keypoints_color = frame.left[u, v]
            else:
                keypoints_3d = np.zeros((0, 3), dtype=frame.left_depth_xyz.dtype)
                keypoints_depth = np.zeros((0,), dtype=frame.left_depth.dtype)
                if frame.left.ndim == 3:
                    keypoints_color = np.zeros((0, frame.left.shape[2]), dtype=frame.left.dtype)
                else:
                    keypoints_color = np.zeros((0,), dtype=frame.left.dtype)

            features = {
                "keypoints": keypoints_np,
                "descriptors": descriptors,
                "scores": scores_np,
                "scales": scales_np,
                "orientations": orientations_np,
                "keypoints_3d": keypoints_3d,
                "keypoints_depth": keypoints_depth,
                "keypoints_color": keypoints_color,
                "image_size": frame.left_rect.shape[:2],
            }

            feature_frame = FeatureFrame(
                left=None,
                right=None,
                left_rect=None,
                right_rect=None,
                left_depth=None,
                left_depth_xyz=None,
                calibration=frame.calibration,
                features=features,
            )
            feature_frames.append(feature_frame)

        return feature_frames

    def match(self, pairs: list[FramePair[FeatureFrame]]) -> list[MatchedFramePair[FeatureFrame]]:
        if not pairs:
            return []

        results: list[MatchedFramePair[FeatureFrame]] = []

        for pair in pairs:
            first_descriptors = pair.first.features["descriptors"]
            second_descriptors = pair.second.features["descriptors"]

            if isinstance(first_descriptors, torch.Tensor):
                first_descriptors_np = first_descriptors.detach().cpu().numpy()
            else:
                first_descriptors_np = np.asarray(first_descriptors)

            if isinstance(second_descriptors, torch.Tensor):
                second_descriptors_np = second_descriptors.detach().cpu().numpy()
            else:
                second_descriptors_np = np.asarray(second_descriptors)

            if first_descriptors_np.size == 0 or second_descriptors_np.size == 0:
                idxs = np.zeros((0, 2), dtype=np.int64)
            else:
                first_descriptors_np = first_descriptors_np.astype(np.uint8, copy=False)
                second_descriptors_np = second_descriptors_np.astype(np.uint8, copy=False)
                matches = list(self.bf_matcher.match(first_descriptors_np, second_descriptors_np))
                matches.sort(key=lambda m: m.distance)
                idxs = np.array([[m.queryIdx, m.trainIdx] for m in matches], dtype=np.int64)

            results.append(
                MatchedFramePair(
                    first=pair.first,
                    second=pair.second,
                    matches=idxs,
                )
            )

        return results


# Preserve user-facing API parity with registration.lighterglue
LighterglueMatcher = OrbMatcher
