import kornia as K
import kornia.feature as KF
import numpy as np
import torch

from registration.registration import FramePair, RectifiedStereoFrame
from registration.utils import get_matching_keypoints, np_to_kornia

class LightglueMatcher:
    def __init__(self, device: str = "cpu", compile: bool = False, num_features: int = 2048):
        self.device = device
        self.num_features = num_features

        self.detector =  KF.DISK.from_pretrained("depth").to(self.device)
        self.matcher = KF.LightGlueMatcher("disk").eval().to(self.device)   

        if compile:
            self.detector = self.detector.compile()
            self.matcher = self.matcher.compile()

    def match(self, pair: FramePair[RectifiedStereoFrame]) -> tuple[np.ndarray, np.ndarray]:
        img1 = np_to_kornia(pair.first.left_rect)
        img2 = np_to_kornia(pair.second.left_rect)

        disk = KF.DISK.from_pretrained("depth").to(self.device)

        hw1 = torch.tensor(img1.shape[2:], device=self.device)
        hw2 = torch.tensor(img2.shape[2:], device=self.device)

        with torch.inference_mode():
            inp = torch.cat([img1, img2], dim=0)
            features1, features2 = disk(inp, self.num_features, pad_if_not_divisible=True)
            kps1, descs1 = features1.keypoints, features1.descriptors
            kps2, descs2 = features2.keypoints, features2.descriptors
            lafs1 = KF.laf_from_center_scale_ori(kps1[None], torch.ones(1, len(kps1), 1, 1, device=self.device))
            lafs2 = KF.laf_from_center_scale_ori(kps2[None], torch.ones(1, len(kps2), 1, 1, device=self.device))
            _dists, idxs = self.matcher(descs1, descs2, lafs1, lafs2, hw1=hw1, hw2=hw2)

            print('kps1', kps1.shape)
            print('kps2', kps2.shape)
            print('idxs', idxs.shape)

        mkpts1, mkpts2 = get_matching_keypoints(kps1, kps2, idxs)
        
        return np.array(mkpts1), np.array(mkpts2)
