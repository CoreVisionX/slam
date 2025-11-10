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
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the similarity Umeyama alignment that maps source -> target.

    Returns:
        A tuple (R, t, s) where R is a 3x3 rotation matrix, t is a 3-vector,
        and s is the scale factor.
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
    U, singular_values, Vt = np.linalg.svd(covariance)

    det = np.linalg.det(U) * np.linalg.det(Vt)
    correction = np.ones(U.shape[0])
    if det < 0:
        correction[-1] = -1.0

    R = U @ (correction[:, None] * Vt)

    source_var = np.mean(np.sum(source_centered**2, axis=1))
    if source_var < np.finfo(source_points.dtype).eps:
        scale = 1.0
    else:
        scale = float(np.sum(singular_values * correction) / source_var)

    t = target_mean - scale * (R @ source_mean)
    return R, t, scale


def compute_umeyama_alignment_pose(
    source_poses: Iterable[gtsam.Pose3],
    target_poses: Iterable[gtsam.Pose3],
) -> tuple[gtsam.Rot3, np.ndarray, float] | None:
    """
    Estimate the similarity transform that best aligns source_poses to target_poses.

    Args:
        source_poses: Poses to align (e.g. ground truth).
        target_poses: Reference poses (e.g. estimates). Must match in length.

    Returns:
        A tuple (R, t, s) where R is a gtsam.Rot3, t is a translation vector, and
        s is the scale. Returns None if alignment cannot be computed.
    """

    source_list = list(source_poses)
    target_list = list(target_poses)

    if len(source_list) == 0 or len(source_list) != len(target_list):
        return None

    if len(source_list) == 1:
        delta = pose_translation_to_array(target_list[0]) - pose_translation_to_array(source_list[0])
        return gtsam.Rot3.Identity(), np.asarray(delta, dtype=float), 1.0

    source_points = np.stack([pose_translation_to_array(pose) for pose in source_list])
    target_points = np.stack([pose_translation_to_array(pose) for pose in target_list])

    try:
        R, t, scale = _umeyama_alignment(source_points, target_points)
    except np.linalg.LinAlgError:
        return None

    return gtsam.Rot3(R), t, scale


def apply_similarity_transform(
    pose: gtsam.Pose3, rotation: gtsam.Rot3, translation: np.ndarray, scale: float
) -> gtsam.Pose3:
    """Apply a similarity transform (scale, rotation, translation) to a pose."""

    rotation_matrix = rotation.matrix()
    translated = rotation_matrix @ np.asarray(pose_translation_to_array(pose), dtype=float)
    aligned_translation = scale * translated + translation
    aligned_rotation = rotation.compose(pose.rotation())
    return gtsam.Pose3(aligned_rotation, gtsam.Point3(*aligned_translation))
