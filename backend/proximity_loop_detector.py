import gtsam
from gtsam.symbol_shorthand import X
import numpy as np

from backend.pose_graph import GtsamPoseGraph
from registration.registration import IndexedFramePair, StereoDepthFrame


class ProximityLoopDetector:
    # TODO: at some point figure out how to properly relax the rotation constraint so that viewpoitns of the same landmarks from different rotations work (maybe check via projection?)
    # but definitely not until the simple orb solvepnp case is solved first
    def __init__(self, max_translation, max_rotation, max_candidates, min_seperation=0): # meters, radians
        self.max_translation = max_translation
        self.max_rotation = max_rotation
        self.max_candidates = max_candidates
        self.min_seperation = int(min_seperation)

    def candidates(self, pose_graph: GtsamPoseGraph, pose_key_idx: int) -> list[IndexedFramePair[StereoDepthFrame]]: # factor graph nodes
        keys = []
        distances = []
        angular_distances = []

        pose = pose_graph.values.atPose3(X(pose_key_idx))
        pose_symbol = gtsam.Symbol(X(pose_key_idx))
        pose_char = pose_symbol.chr()
        
        # find valid candidates
        for key in pose_graph.values.keys():
            symbol = gtsam.Symbol(int(key))
            if symbol.chr() != pose_char:
                continue
            if self.min_seperation > 0 and abs(symbol.index() - pose_key_idx) < self.min_seperation:
                continue

            pose_candidate = pose_graph.values.atPose3(key)
            relative_pose = pose_candidate.inverse() * pose

            relative_distance = np.linalg.norm(relative_pose.translation())
            relative_rotation = np.linalg.norm(relative_pose.rotation().ypr()) # TODO: use axis-angle or quaternions to avoid potential singularity issues?

            if relative_distance < self.max_translation and relative_rotation < self.max_rotation:
                keys.append(gtsam.Symbol(key).index())
                distances.append(relative_distance)
                angular_distances.append(relative_rotation)

        keys = np.array(keys)
        distances = np.array(distances)
        angular_distances = np.array(angular_distances)

        # sort by distance and take the top candidates
        keys = keys[np.argsort(distances)][:self.max_candidates]

        # # sort by angle
        # keys = keys[np.argsort(angular_distances)][:self.max_candidates]

        return [IndexedFramePair[StereoDepthFrame](
            first=pose_graph.frames[pose_key_idx],
            second=pose_graph.frames[key],
            first_idx=pose_key_idx,
            second_idx=key
        ) for key in keys if key in pose_graph.frames]


