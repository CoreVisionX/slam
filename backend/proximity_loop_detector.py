import gtsam
import numpy as np


class ProximityLoopDetector:
    # TODO: at some point figure out how to properly relax the rotation constraint so that viewpoitns of the same landmarks from different rotations work (maybe check via projection?)
    # but definitely not until the simple orb solvepnp case is solved first
    def __init__(self, max_translation, max_rotation, max_candidates, min_seperation=0): # meters, radians
        self.max_translation = max_translation
        self.max_rotation = max_rotation
        self.max_candidates = max_candidates
        self.min_seperation = int(min_seperation)

    def candidates(self, nodes, pose_key): # factor graph nodes
        keys = []
        distances = []

        pose = nodes.atPose3(pose_key)
        pose_symbol = gtsam.Symbol(int(pose_key))
        pose_char = pose_symbol.chr()
        pose_index = pose_symbol.index()
        
        # find valid candidates
        for key in nodes.keys():
            symbol = gtsam.Symbol(int(key))
            if symbol.chr() != pose_char:
                continue
            if self.min_seperation > 0 and abs(symbol.index() - pose_index) < self.min_seperation:
                continue

            pose_candidate = nodes.atPose3(key)
            relative_pose = pose_candidate.inverse() * pose

            relative_distance = np.linalg.norm(relative_pose.translation())
            relative_rotation = np.linalg.norm(relative_pose.rotation().ypr()) # TODO: use axis-angle or quaternions to avoid potential singularity issues?

            if relative_distance < self.max_translation and relative_rotation < self.max_rotation:
                keys.append(key)
                distances.append(relative_distance)

        keys = np.array(keys)
        distances = np.array(distances)

        # sort by distance and take the top candidates
        keys = keys[np.argsort(distances)][:self.max_candidates]

        return keys
