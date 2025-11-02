from dataclasses import asdict
import kornia as K
import kornia.feature as KF
from kornia.feature.lightglue import pad_to_length
import numpy as np
import torch
import time

from registration.registration import FramePair, RectifiedStereoFrame, StereoDepthFrame, FeatureFrame
from registration.utils import get_matching_keypoints, np_to_kornia

class LightglueMatcher:
    def __init__(self, device: str = "cuda", compile: bool = True, num_features: int = 2048, mp=True):
        self.device = device
        self.num_features = num_features
        self.mp = mp

        self.detector =  KF.DISK.from_pretrained("depth").eval().to(self.device)
        self.matcher = KF.LightGlueMatcher("disk", params={"mp": mp}).eval().to(self.device)   

        if compile:
            self.detector.unet.compile(mode="reduce-overhead", fullgraph=True, dynamic=False)
            self.matcher.matcher.compile()

    def detect_features(self, frames: list[StereoDepthFrame]) -> list[FeatureFrame]:
        imgs = [np_to_kornia(frame.left_rect).to(self.device) for frame in frames]
        inp = torch.cat(imgs, dim=0)

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", enabled=self.mp):
                features = self.detector(inp, self.num_features, pad_if_not_divisible=True)

        new_frames = []

        for frame, features in zip(frames, features):
            prev_frame = frame.__dict__
            if 'features' in prev_frame:
                prev_frame.pop('features')

            new_frame = FeatureFrame(
                **prev_frame,
                features=KF.DISKFeatures(
                    keypoints=features.keypoints.detach().clone(),
                    descriptors=features.descriptors.detach().clone(),
                    detection_scores=features.detection_scores.detach().clone(),
                )
            )
            new_frames.append(new_frame)

        return new_frames

    def match(self, pairs: list[FramePair[FeatureFrame]]) -> list[tuple[np.ndarray, np.ndarray]]:
        if len(pairs) == 0:
            return []

        # TODO: double check the order here
        image_sizes1 = torch.cat([torch.tensor(pair.first.left_rect.shape[:2], device=self.device).unsqueeze(0) for pair in pairs], dim=0)
        image_sizes2 = torch.cat([torch.tensor(pair.second.left_rect.shape[:2], device=self.device).unsqueeze(0) for pair in pairs], dim=0)
        assert image_sizes1.shape == (len(pairs), 2)
        assert image_sizes2.shape == (len(pairs), 2)

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", enabled=self.mp):
                torch.compiler.cudagraph_mark_step_begin()
                
                kps1, mask1 = zip(*[pad_to_length(pair.first.features.keypoints[None], self.num_features) for pair in pairs])
                kps1 = torch.cat(kps1, dim=0)
                mask1 = torch.cat(mask1, dim=0)

                kps2, mask2 = zip(*[pad_to_length(pair.second.features.keypoints[None], self.num_features) for pair in pairs])
                kps2 = torch.cat(kps2, dim=0)
                mask2 = torch.cat(mask2, dim=0)

                assert kps1.shape == (len(pairs), self.num_features, 2)
                assert kps2.shape == (len(pairs), self.num_features, 2)
                assert mask1.shape == (len(pairs), self.num_features, 1)
                assert mask2.shape == (len(pairs), self.num_features, 1)


                descs1, desc_mask1 = zip(*[pad_to_length(pair.first.features.descriptors[None], self.num_features) for pair in pairs])
                descs1 = torch.cat(descs1, dim=0)
                desc_mask1 = torch.cat(desc_mask1, dim=0)

                descs2, desc_mask2 = zip(*[pad_to_length(pair.second.features.descriptors[None], self.num_features) for pair in pairs])
                descs2 = torch.cat(descs2, dim=0)
                desc_mask2 = torch.cat(desc_mask2, dim=0)

                assert descs1.shape == (len(pairs), self.num_features, descs1.shape[2])
                assert descs2.shape == (len(pairs), self.num_features, descs2.shape[2])
                assert desc_mask1.shape == mask1.shape
                assert desc_mask2.shape == mask2.shape

                output = self.matcher.matcher({
                    "image0": {
                        "keypoints": kps1,
                        "descriptors": descs1,
                        "image_size": image_sizes1,
                        "mask": mask1,
                    },
                    "image1": {
                        "keypoints": kps2,
                        "descriptors": descs2,
                        "image_size": image_sizes2,
                        "mask": mask2,
                    },
                })

                # TODO: use the scores for noise/filtering somehow? this is why easy iteration is so useful!
                batched_matches = output["matches"]
                assert len(batched_matches) == len(pairs)
                        
                results = []
                for kps1, kps2, matches in zip(kps1, kps2, batched_matches):
                    matches = matches.cpu().numpy()
                    assert matches.shape == (matches.shape[0], 2)

                    mkpts1, mkpts2 = get_matching_keypoints(kps1, kps2, matches)
                    results.append((np.array(mkpts1.cpu(), dtype=np.float32), np.array(mkpts2.cpu(), dtype=np.float32)))

                return results
