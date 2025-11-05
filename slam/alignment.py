"""Utilities for aligning trajectories."""

from __future__ import annotations

from typing import Iterable

import gtsam
import numpy as np


def pose_translation_to_array(pose: gtsam.Pose3) -> np.ndarray:
    """Extract the translation component of a pose as an ndarray."""

    return pose.translation()


def _umeyama_alignment(
    source_points: np.ndarray, target_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the rigid Umeyama alignment (no scaling) that maps source -> target.

    Returns:
        A tuple (R, t) where R is a 3x3 rotation matrix and t is a 3-vector.
    """

    if source_points.shape != target_points.shape:
        raise ValueError("Point sets must have the same shape.")

    n_points = source_points.shape[0]
    if n_points == 0:
        raise ValueError("At least one point is required for Umeyama alignment.")

    source_mean = source_points.mean(axis=0)
    target_mean = target_points.mean(axis=0)

    source_centered = source_points - source_mean
    target_centered = target_points - target_mean

    covariance = source_centered.T @ target_centered / n_points
    U, _, Vt = np.linalg.svd(covariance)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = target_mean - R @ source_mean
    return R, t


def compute_umeyama_alignment_pose(
    source_poses: Iterable[gtsam.Pose3],
    target_poses: Iterable[gtsam.Pose3],
) -> gtsam.Pose3 | None:
    """
    Estimate the pose that best aligns source_poses to target_poses using Umeyama.

    Args:
        source_poses: Poses to align (e.g. ground truth).
        target_poses: Reference poses (e.g. estimates). Must match in length.

    Returns:
        A gtsam.Pose3 representing the alignment transform, or None if alignment
        cannot be computed.
    """

    source_list = list(source_poses)
    target_list = list(target_poses)

    if len(source_list) == 0 or len(source_list) != len(target_list):
        return None

    if len(source_list) == 1:
        delta = pose_translation_to_array(target_list[0]) - pose_translation_to_array(source_list[0])
        return gtsam.Pose3(gtsam.Rot3.Identity(), gtsam.Point3(*delta))

    source_points = np.stack([pose_translation_to_array(pose) for pose in source_list])
    target_points = np.stack([pose_translation_to_array(pose) for pose in target_list])

    try:
        R, t = _umeyama_alignment(source_points, target_points)
    except np.linalg.LinAlgError:
        return None

    return gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(*t))

