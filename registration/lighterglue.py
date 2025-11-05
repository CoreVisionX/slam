from dataclasses import asdict
import kornia as K
import kornia.feature as KF
from kornia.feature.lightglue import pad_to_length
import numpy as np
import torch

from registration.registration import FramePair, MatchedFramePair, StereoDepthFrame, FeatureFrame
from registration.utils import get_matching_keypoints


# TODO: the name of this class might be misleading
# also should we decouple feature detection and matching so models don't need to be loaded in every process?
class LighterglueMatcher:
    def __init__(self, device: str = "cuda", compile: bool = False, num_features: int = 4096, use_lighterglue_matching: bool = False):
        self.device = device
        self.num_features = num_features
        self.use_lighterglue_matching = use_lighterglue_matching

        self.matcher = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = self.num_features).eval().to(self.device)

        if compile:
            raise NotImplementedError("Compilation is not supported yet for XFeat")

    def detect_features(self, frames: list[StereoDepthFrame]) -> list[FeatureFrame]:
        features = []

        for frame in frames:
            output = self.matcher.detectAndCompute(frame.left_rect, top_k = self.num_features)[0]

            # move all features to the CPU to avoid multiprocessing issues
            for key in output.keys():
                if isinstance(output[key], torch.Tensor):
                    output[key] = output[key].detach().cpu().numpy()

            # calculate the 3d positions of the features
            u = output['keypoints'][:, 1].astype(int)
            v = output['keypoints'][:, 0].astype(int)
            output['keypoints_3d'] = frame.left_depth_xyz[u, v, :]
            output['keypoints_depth'] = frame.left_depth[u, v]
            output['keypoints_color'] = frame.left[u, v]

            output.update({
                'image_size': frame.left_rect.shape[:2] # TODO: double check the order here
            })
            features.append(output)

        new_frames = []

        for frame, features in zip(frames, features):
            new_frame = FeatureFrame(
                left=None,
                right=None,
                left_rect=None,
                right_rect=None,
                left_depth=None,
                left_depth_xyz=None,
                calibration=frame.calibration,
                features=features
            )
            new_frames.append(new_frame)

        return new_frames

    def match(self, pairs: list[FramePair[FeatureFrame]]) -> list[MatchedFramePair[FeatureFrame]]:
        if len(pairs) == 0:
            return []

        results = []

        for pair in pairs:
            # move all features back to the GPU for matching
            pair.first.features['keypoints'] = torch.tensor(pair.first.features['keypoints']).to(self.device)
            pair.second.features['keypoints'] = torch.tensor(pair.second.features['keypoints']).to(self.device)

            pair.first.features['descriptors'] = torch.tensor(pair.first.features['descriptors']).to(self.device)
            pair.second.features['descriptors'] = torch.tensor(pair.second.features['descriptors']).to(self.device)
            
            first_features = pair.first.features
            second_features = pair.second.features

            if self.use_lighterglue_matching:
                _mkpts1, _mkpts2, idxs = self.matcher.match_lighterglue(first_features, second_features)
            else:
                idxs1, idxs2 = self.matcher.match(first_features['descriptors'], second_features['descriptors'], min_cossim=-1)
                idxs = np.stack([idxs1.cpu().numpy(), idxs2.cpu().numpy()], axis=1)

            result = MatchedFramePair(
                first=pair.first,
                second=pair.second,
                matches=idxs
            )
            results.append(result)

        return results
