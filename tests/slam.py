# %%
import atexit
from dataclasses import dataclass
from queue import Empty, Full
import argparse
import sys
import os
import time
import gtsam
from gtsam.symbol_shorthand import X
import numpy as np
import rerun as rr
# import torch.multiprocessing as mp
# from torch.multiprocessing import Process, Queue
from multiprocessing import Process, Queue, SimpleQueue
import multiprocessing as mp

# os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1" 
# import torch, torch._inductor.config as ind
# ind.compile_threads = 1

# TODO: pose interoplation between keyframes for realtime odometry output
# TODO: velocity factors for smoother trajectories (have process_odometry accept dt and use it to compute the velocity)
# TODO: encourage wider baseline loop closures, like maybe fan out? adding a vpr model like dino salad v2 would definitely help a ton as well

# TODO: why is the main thread still running so slow even with loop closure running in a separate process?
# there shouldn't be anything heavy blocking??

# TODO: watch memory usage to make sure the jetson isn't running out of memory
# TODO: use python logger and make sure rerun it picks it up for easier debugging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.pose_graph import GtsamPoseGraph
from backend.proximity_loop_detector import ProximityLoopDetector
from tests.test_utils import get_tartanair_iterator_with_odometry, tartanair_calib
from depth.sgbm import SGBM
from registration.registration import FramePair, IndexedFramePair, FeatureFrame
# from registration.lightglue import LightglueMatcher
from registration.lighterglue import LighterglueMatcher
# from registration.orb import OrbMatcher
from registration.utils import fundamental_fitler, solve_pnp
from viz import rr_log_map_points, rr_log_pose, rr_log_trajectory, rr_log_graph_edges
from util import share_feature_frame


ALIGN_GT_KEYFRAMES = True
USE_HUBER_LOSS = True # definitely turn it off when you're debugging things related to the factor graph since it can cover up outliers that may be causing issues


# TODO: definitely factor all this multiprocessing logic out, it needs to be robust and reusable
# TODO: log loop closure candidate average age somehow to make sure they're not too stale or doing anything weird
@dataclass
class LoopClosureResult:
    # candidate: IndexedFramePair[FeatureFrame]
    first_idx: int
    second_idx: int
    # first_to_second: gtsam.Pose3
    R_first_to_second: np.ndarray
    t_first_to_second: np.ndarray
    # noise: gtsam.noiseModel.Diagonal
    # noise_diagonal: np.ndarray
    # mkpts1: np.ndarray
    # mkpts1_3d: np.ndarray
    # mkpts1_color: np.ndarray


def _pose_translation_to_array(pose: gtsam.Pose3) -> np.ndarray:
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
    assert source_points.shape == target_points.shape, "Point sets must have the same shape"
    n_points = source_points.shape[0]
    if n_points == 0:
        raise ValueError("At least one point is required for Umeyama alignment")

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
    source_poses: list[gtsam.Pose3], target_poses: list[gtsam.Pose3]
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
    if len(source_poses) == 0 or len(source_poses) != len(target_poses):
        return None

    if len(source_poses) == 1:
        delta = _pose_translation_to_array(target_poses[0]) - _pose_translation_to_array(source_poses[0])
        return gtsam.Pose3(gtsam.Rot3.Identity(), gtsam.Point3(*delta))

    source_points = np.stack([_pose_translation_to_array(pose) for pose in source_poses])
    target_points = np.stack([_pose_translation_to_array(pose) for pose in target_poses])

    try:
        R, t = _umeyama_alignment(source_points, target_points)
    except np.linalg.LinAlgError:
        return None

    return gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(*t))
    
def loop_closure_worker(loop_closure_candidates_queue: Queue, loop_closures_queue: Queue, loop_noise: gtsam.noiseModel.Diagonal, min_inlier_count: int, worker_id: int = 0):
    """
    Worker process for loop closing

    # Approach

    Queue message passing:
    - Every frame the main process the odometry, add it to the pose graph,
    find loop closure candidates, and queue the candidate pairs from the loop closure worker
    - The loop closure worker will get the latest candidate pairs (this will be implement by having a maximum queue size that's fairly low to prevent latency?),
    attempt to estimate their relative pose, and send the results back to the main process via a loop closures queue
    - The worker will be checking the loop closures queue and adding the loop closures to the pose graph as they come in
    """

    # TODO: consider factoring this out into it's own class too
    # TODO: log match images somehow?
    # TODO: log numbers and changes in inlier counts somehow?
    # setup two-view pose estimation
    # matcher = LightglueMatcher(num_features=1536, compile=False, mp=True, device='cuda')
    matcher = LighterglueMatcher(num_features=4096, compile=False, device='cuda', use_lighterglue_matching=True)
    # matcher = OrbMatcher(num_features=2000)
    print(f"Loop closure worker {worker_id} started")

    times = []
    i = 0
    log_every = 20
    max_queue_size = 20

    candidates = []

    while True:
        if len(candidates) == 0 and loop_closure_candidates_queue.empty():
            time.sleep(0.01)
            continue

        # maintain the queue size on the worker side to prevent race conditions since only one actor is reading from the queue
        while not loop_closure_candidates_queue.empty():
            # candidates.append(loop_closure_candidates_queue.get(block=True, timeout=0.1))
            candidates.append(loop_closure_candidates_queue.get())
            # print(f"Added candidate {candidates[-1].first_idx} and {candidates[-1].second_idx} to candidates")

        if len(candidates) > max_queue_size:
            candidate_idxs = [candidate.first_idx for candidate in candidates]
            print(f"[Worker {worker_id}] Candidates overflowing idxs: ({len(candidate_idxs)} total) ...{candidate_idxs[-5:]}")

        # candidates = candidates[-max_queue_size:]

        candidate = candidates.pop(-1) # process the latest candidates first
        # candidate = candidates.pop(0)

        # print(f"Processing candidate {candidate.first_idx} and {candidate.second_idx}")

        start_time = time.perf_counter()

        # mkpts1, mkpts2 = matcher.match([candidate])[0]
        matched_pair = matcher.match([candidate])[0]

        try:
            # TODO: refactor to support fundemental filtering with feature frames
            # filtered_mkpts1, filtered_mkpts2, inlier_mask = fundamental_fitler(mkpts1, mkpts2)
            
            first_to_second, matched_pair = solve_pnp(matched_pair)
        except Exception as e:
            print(f"[Worker {worker_id}] Failed to solve PnP for loop closure between {candidate.first_idx} and {candidate.second_idx} with {len(matched_pair.matches)} inliers: {e}")
            continue

        if len(matched_pair.matches) > min_inlier_count:
            # TODO: would sending back a matched pair be better?
            # idk think about composition over inheritance for this kind of thing
            result = LoopClosureResult(
                # candidate=candidate,
                first_idx=candidate.first_idx,
                second_idx=candidate.second_idx,
                # first_to_second=first_to_second,
                R_first_to_second=first_to_second.rotation().matrix(),
                t_first_to_second=first_to_second.translation(),
                # noise=loop_noise,
                # noise_diagonal=loop_noise.sigmas(),
                # noise_diagonal=None,
                # mkpts1=matched_pair.mkpts1,
                # mkpts1_3d=matched_pair.mkpts1_3d,
                # mkpts1_color=matched_pair.mkpts1_color,
            )
            loop_closures_queue.put(result)
            # print(f"(Current keyframe index: {result.candidate.first_idx}) Added result for candidate {candidate.first_idx} and {candidate.second_idx}")
            print(f"[Worker {worker_id}] (Current keyframe index: {result.first_idx}) Added result for candidate {result.first_idx} and {result.second_idx}")
        else:
            # print(f"Failed to solve PnP for loop closure between {candidate.first_idx} and {candidate.second_idx} with {len(matched_pair.matches)} inliers")
            pass

        end_time = time.perf_counter()
        times.append(end_time - start_time)
        i += 1
        
        if i % log_every == 0:
            print(f"[Worker {worker_id}] Mean Loop Closure Processing Time: {np.mean(times):.2f} seconds ({1 / np.mean(times):.2f} fps)")
            times = []


if __name__ == "__main__":
    mp.set_start_method('spawn')

    timestamps = np.load(os.path.join(os.path.dirname(__file__), 'data', 'AbandonedFactory', 'Data_easy', 'P001', 'imu', 'cam_time.npy'))

    # TODO: the noise should probably be much higher than the real noise since we're integrating the odometry poses between keyframes
    # odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([np.deg2rad(1.0), np.deg2rad(1.0), np.deg2rad(1.0), 0.02, 0.02, 0.02]) * 100_000) # make it exteremly high so it doesn't influence anything
    odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([np.deg2rad(0.3), np.deg2rad(0.3), np.deg2rad(0.3), 0.007, 0.007, 0.007]))

    # TODO: actually robust rerun setup (factor this out somewhere reusable)
    rr.init("slam")
    # rr.connect_tcp("192.168.88.1:9876")
    rr.connect_tcp("192.168.1.20:9876")

    sgbm = SGBM(num_disparities=16 * 4, block_size=5, image_color='RGB')

    # setup backend
    pose_graph = GtsamPoseGraph(K=tartanair_calib.K_left_rect)

    # TODO: figure out a proximity loop closer with a minimum seperation between candidates it finds, not just the current pose. maybe that's what keyframes should be doing and I need a better way of maintaining the local trajectory smoothness? idk.
    # show encourage longer baseline loop closures
    loop_detector = ProximityLoopDetector(max_translation=12.0, max_rotation=np.deg2rad(60), max_candidates=20, min_seperation=1)

    # TODO: figure out the proper noise model for this
    # an interactive UI with sliders and visualizaation of the resulting optimization might be really useful
    # also adding an optoin for a basic huber loss might help with outliers
    # in that vein proper geometric verification in terms of making sure a new loop doesn't cause a massive weird change in the map or anything would be really useful
    # in allowing for wide baseline loop closures that help a lot with large areas and long sessions
    # while still avoding any bad loop closures that cause catastrophic damage to the map
    if USE_HUBER_LOSS:
        print("Using Huber loss for loop closures")
        loop_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber(1.0),
            gtsam.noiseModel.Diagonal.Sigmas(np.array([np.deg2rad(0.3), np.deg2rad(0.3), np.deg2rad(0.3), 0.014, 0.014, 0.014]))
        )
    else:
        print("Not using Huber loss for loop closures")
        loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([np.deg2rad(0.3), np.deg2rad(0.3), np.deg2rad(0.3), 0.014, 0.014, 0.014]))
    # loop_noise = gtsam.noiseModel.Gaussian.Covariance(np.array([
    #     [ 0.000006,  0.000000, -0.000000,  0.000002,  0.000042,  0.000008],
    #     [ 0.000000,  0.000006, -0.000000, -0.000053,  0.000005,  0.000019],
    #     [-0.000000, -0.000000,  0.000003, -0.000001,  0.000001, -0.000006],
    #     [ 0.000002, -0.000053, -0.000001,  0.000711,  0.000028, -0.000017],
    #     [ 0.000042,  0.000005,  0.000001,  0.000028,  0.000541,  0.000226],
    #     [ 0.000008,  0.000019, -0.000006, -0.000017,  0.000226,  0.000637],
    # ]))
    # loop_noise = odometry_noise # set it to the odometry noise so they're weighted equally which will make debugging the influence of the loop closures easier

    # TODO: might be too high and preventing wide baseline loop closures
    # try adding geometric and graph consistency checks to allow for more robust <100 inlier loop closures?
    # varying the noise based on the inlier count might also help?
    # a basic robust loss might also help with outliers
    # velocity factors might help as well
    min_inlier_count = 100

    # TODO: refactor keyframing logic into it's own class
    # goal: encourage high quality wide baseline loop closures
    keyframe_translation_threshold = 1.5 # I notice that higher values let more wide baseline loop closures be detected
    keyframe_rotation_threshold = np.deg2rad(24)
    cur_keyframe_pose = None

    raw_trajectory: list[gtsam.Pose3] = []
    gt_trajectory: list[gtsam.Pose3] = []
    optimized_trajectory: list[gtsam.Pose3] = []

    raw_keyframe_trajectory: list[gtsam.Pose3] = []
    gt_keyframe_trajectory: list[gtsam.Pose3] = []

    map_points: list[tuple[int, np.ndarray, np.ndarray]] = []


    # frames with features already added should be okay since
    # xfeat runs at like 15 fps
    # should be Queue[IndexedFramePair[StereoDepthFrameWithFeatures]] but python doesn't support it?
    # TODO: low max size and overwrite old candidates to maintain low latency
    # loop_closure_candidates_queue = Queue(maxsize=5000)
    loop_closure_candidates_queue = SimpleQueue()

    # should be Queue[LoopClosureResult] but python doesn't support it?
    loop_closures_queue = Queue(maxsize=5000)
    # loop_closures_queue = SimpleQueue()

    # inefficient but we need an instance of the matcher here too for feature detection
    # we could either split the matcher into a feature detection and matching part or just have the worker do both?
    matcher = LighterglueMatcher(num_features=4096, compile=False, device='cuda', use_lighterglue_matching=True)
    # matcher = OrbMatcher(num_features=2000)

    try:
        available_cpus = mp.cpu_count()
    except NotImplementedError:
        available_cpus = 1

    if available_cpus and available_cpus > 1:
        # num_loop_closure_workers = min(4, max(2, available_cpus // 2))
        num_loop_closure_workers = 1
    else:
        num_loop_closure_workers = 1

    print(f"Starting {num_loop_closure_workers} loop closure workers")

    loop_closure_worker_processes: list[Process] = []
    for worker_idx in range(num_loop_closure_workers):
        loop_closure_worker_process = Process(
            target=loop_closure_worker,
            args=(loop_closure_candidates_queue, loop_closures_queue, loop_noise, min_inlier_count, worker_idx),
            daemon=True,
        )
        loop_closure_worker_process.start()
        loop_closure_worker_processes.append(loop_closure_worker_process)

    def _shutdown_loop_closure_workers():
        for process in loop_closure_worker_processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

    atexit.register(_shutdown_loop_closure_workers)

    print(f"Waiting for {num_loop_closure_workers} loop closure workers to start...")
    time.sleep(6) # wait for the worker to start

    exc_ts_start = None

    odometry_iterator = get_tartanair_iterator_with_odometry(
        env='AbandonedFactory', 
        difficulty='easy', 
        traj='P001', 
        include_ground_truth=True,
        rotation_noise_sigmas=np.array([np.deg2rad(0.3), np.deg2rad(0.3), np.deg2rad(0.3)]),
        translation_noise_sigmas=np.array([0.007, 0.007, 0.007])
    )

    print("Starting trajectory processing...")
    for i, (frame, noisy_prev_robot_to_robot, first_robot_to_robot) in enumerate(odometry_iterator):
        ts = timestamps[i]
        
        if exc_ts_start is None:
            exc_ts_start = time.perf_counter()

        exc_ts = time.perf_counter() - exc_ts_start

        # update raw/gt trajectories
        gt_trajectory.append(first_robot_to_robot)

        if len(raw_trajectory) == 0:
            raw_trajectory.append(noisy_prev_robot_to_robot)
        else:
            raw_trajectory.append(raw_trajectory[-1].compose(noisy_prev_robot_to_robot))

        # simulate realtime processing

        # warn if we're behind realtime by more than 0.5 seconds
        if exc_ts > (ts + 0.5):
            print(f"Behind realtime by {exc_ts - ts:.2f} seconds, skipping frame")
            continue

        # sleep until the true time if we're ahead of realtime
        if exc_ts < ts:
            time.sleep(ts - exc_ts)

        # # TODO: get rid of this lmao we aren't using the hard sequence rn
        # time.sleep(0.2) # wait to simulate realtime processing

        start_time = time.perf_counter()

        # update slam if a new keyframe is detected
        if cur_keyframe_pose is None:
            cur_keyframe_pose = first_robot_to_robot

            # TODO: definitely need to clean this up and refactor it out somewhere reusable and debuggable and testable, it's a big perf sink
            # preprocess frame
            start_preprocess_time = time.perf_counter()
            rectified_frame = frame.rectify()
            depth_frame = sgbm.compute_depth(rectified_frame)
            feature_frame = matcher.detect_features([depth_frame])[0]
            feature_frame = share_feature_frame(feature_frame)
            end_preprocess_time = time.perf_counter()
            # print(f"Preprocess Time: {end_preprocess_time - start_preprocess_time:.2f} seconds ({1 / (end_preprocess_time - start_preprocess_time):.2f} fps)")

            start_process_odometry_time = time.perf_counter()
            pose_graph.process_odometry(noisy_prev_robot_to_robot, odometry_noise, feature_frame)
            end_process_odometry_time = time.perf_counter()
            print(f"Process Odometry Time: {end_process_odometry_time - start_process_odometry_time:.2f} seconds ({1 / (end_process_odometry_time - start_process_odometry_time):.2f} fps)")

            gt_keyframe_trajectory.append(first_robot_to_robot)

            # TODO: figure out why doing this causes the estimated trajectory to be offset? I think it's because even this is actually the second frame not the first one or something?
            # # TODO: this is a hack to get the loop detector to work with the first keyframe, fix this later
            # pose_graph.frames[0] = depth_frame_with_features

        # TODO: setup a proper reusable VO pipeline if you're going to use this
        # # use visual odometry instead of the given raw odometry
        # if prev_frame is not None:
        #     try:
        #         pair = FramePair[StereoDepthFrame](first=prev_frame, second=depth_frame_with_features)
        #         mkpts1, mkpts2 = matcher.match(pair)
        #         filtered_mkpts1, filtered_mkpts2, inlier_mask = fundamental_fitler(mkpts1, mkpts2)
        #         first_to_second, pnp_filtered_mkpts1, pnp_filtered_mkpts2, pnp_inliers = solve_pnp(pair, filtered_mkpts1, filtered_mkpts2)

        #         if len(pnp_inliers) < min_inlier_count:
        #             raise ValueError("Not enough final inliers to solve PnP")
            
        #         noisy_prev_robot_to_robot = first_to_second
        #         print(f"Computed visual odometry between {len(pnp_inliers)} inliers")

        #     except ValueError:
        #         print(f"Failed to solve PnP for visual odometry between {prev_frame.idx} and {depth_frame_with_features.idx}")

        # use raw odometry for keyframe calculations
        cur_keyframe_to_robot = cur_keyframe_pose.inverse() * raw_trajectory[-1]
        dist = np.linalg.norm(cur_keyframe_to_robot.translation())
        rot = np.linalg.norm(cur_keyframe_to_robot.rotation().ypr()) # TODO: use axis-angle or quaternions to avoid potential singularity issues?

        if dist > keyframe_translation_threshold or rot > keyframe_rotation_threshold:
            # preprocess frame
            start_preprocess_time = time.perf_counter()

            start_rectify_time = time.perf_counter()
            rectified_frame = frame.rectify()
            end_rectify_time = time.perf_counter()
            # print(f"Rectify Time: {end_rectify_time - start_rectify_time:.2f} seconds ({1 / (end_rectify_time - start_rectify_time):.2f} fps)")

            start_compute_depth_time = time.perf_counter()
            depth_frame = sgbm.compute_depth(rectified_frame)
            end_compute_depth_time = time.perf_counter()
            # print(f"Compute Depth Time: {end_compute_depth_time - start_compute_depth_time:.2f} seconds ({1 / (end_compute_depth_time - start_compute_depth_time):.2f} fps)")

            start_detect_features_time = time.perf_counter()
            feature_frame = matcher.detect_features([depth_frame])[0]
            feature_frame = share_feature_frame(feature_frame)
            end_detect_features_time = time.perf_counter()

            end_preprocess_time = time.perf_counter()
            print(f"Preprocess Time: {end_preprocess_time - start_preprocess_time:.2f} seconds ({1 / (end_preprocess_time - start_preprocess_time):.2f} fps)")

            gt_keyframe_trajectory.append(first_robot_to_robot)
            raw_keyframe_trajectory.append(raw_trajectory[-1])


            # update keyframe pose
            cur_keyframe_pose = raw_trajectory[-1]

            # update slam with new keyframe
            # cur_keyframe_to_robot accumulates all of the odometry since the last keyframe
            pose_graph.process_odometry(cur_keyframe_to_robot, odometry_noise, feature_frame)

            start_match_time = time.perf_counter()

            # queue new loop closure candidates
            loop_candidates = loop_detector.candidates(pose_graph, pose_graph.kf_idx)

            # TODO: figure out how to not overwrite the queue too much? or at least preserve the closest candidates?
            start_loop_closure_candidates_time = time.perf_counter()

            # should the order here be reversed?
            for candidate in reversed(loop_candidates):
                # # TODO: this is a hack get rid of this later
                # # detach all tensors from the GPU and move to the CPU before queuing to avoid cuda issues
                # for key in candidate.first.features.__dict__:
                #     if isinstance(candidate.first.features.__dict__[key], torch.Tensor):
                #         print(f"Detaching {key} from GPU")
                #         candidate.first.features.__dict__[key] = candidate.first.features.__dict__[key].detach().cpu()

                # for key in candidate.second.features.__dict__:
                #     if isinstance(candidate.second.features.__dict__[key], torch.Tensor):
                #         print(f"Detaching {key} from GPU")
                #         candidate.second.features.__dict__[key] = candidate.second.features.__dict__[key].detach().cpu()

                # TODO: definitely refactor this out somewhere but also
                # make sure the order we're removing candidates from the right side so that we're minimizing
                # latency?
                # try:

                # TODO: this is a big bottleneck, we need to use shared memory or something
                # it takes like 0.4 seconds to copy the candidate to the worker
                loop_closure_candidates_queue.put(candidate)

                # except Full:
                #     print(f"Loop closure candidate queue is full, dropping oldest candidate to maintain low latency")
                #     loop_closure_candidates_queue.get_nowait()
                #     loop_closure_candidates_queue.put_nowait(candidate)

            end_loop_closure_candidates_time = time.perf_counter()
            print(f"Loop Closure Candidates Time: {end_loop_closure_candidates_time - start_loop_closure_candidates_time:.2f} seconds ({1 / (end_loop_closure_candidates_time - start_loop_closure_candidates_time):.2f} fps)")

            # TODO: consider doing this in the outer loop so we don't have to wait for a keyframe to process loop closures?
            # process any loop closure results from the worker
            start_loop_closure_results_time = time.perf_counter()
            # while True:
            while not loop_closures_queue.empty():
                # drain by checking for exceptions instead of using empty() because apparently empty() isn't reliable?
                try:
                    result = loop_closures_queue.get_nowait()
                    # result = loop_closures_queue.get(timeout=0.5)
                    # result = loop_closures_queue.get()
                    assert result is not None
                except Empty:
                    break

                # print(f"(Current keyframe index: {pose_graph.kf_idx}, Delayed {pose_graph.kf_idx - result.candidate.first_idx} frames) Processing result for candidate {result.candidate.first_idx} and {result.candidate.second_idx}")
                print(f"(Current keyframe index: {pose_graph.kf_idx}, Delayed {pose_graph.kf_idx - result.first_idx} frames) Processing result for candidate {result.first_idx} and {result.second_idx}")

                first_to_second = gtsam.Pose3(gtsam.Rot3(result.R_first_to_second), gtsam.Point3(result.t_first_to_second))
                # noise = gtsam.noiseModel.Diagonal.Sigmas(result.noise_diagonal)
                noise = loop_noise

                first_idx = result.first_idx
                second_idx = result.second_idx
                pose_graph.add_between_pose_factor(first_idx, second_idx, first_to_second, noise)
                
                # rand_idxs = np.random.choice(len(result.mkpts1_3d), size=min(200, len(result.mkpts1_3d)), replace=False)
                # map_points.append((result.candidate.first_idx, result.mkpts1_3d[rand_idxs], result.mkpts1_color[rand_idxs]))
                # if len(map_points) > 1000: # cap the map points to prevent memory issues
                #     map_points.pop(0)
            
            end_loop_closure_results_time = time.perf_counter()
            print(f"Loop Closure Results Time: {end_loop_closure_results_time - start_loop_closure_results_time:.2f} seconds ({1 / (end_loop_closure_results_time - start_loop_closure_results_time):.2f} fps)")

            # # single-process loop closure processing
            # candidate_matches = matcher.match(loop_candidates)
            # end_match_time = time.perf_counter()
            # print(f"Match Time: {end_match_time - start_match_time:.2f} seconds ({1 / (end_match_time - start_match_time):.2f} fps)")

            # for candidate, (mkpts1, mkpts2) in zip(loop_candidates, candidate_matches):
            #     try:
            #         start_filter_time = time.perf_counter()
            #         filtered_mkpts1, filtered_mkpts2, inlier_mask = fundamental_fitler(mkpts1, mkpts2)
            #         end_filter_time = time.perf_counter()
            #         print(f"Filter Time: {end_filter_time - start_filter_time:.2f} seconds ({1 / (end_filter_time - start_filter_time):.2f} fps)")

            #         start_pnp_time = time.perf_counter()
            #         first_to_second, pnp_filtered_mkpts1, pnp_filtered_mkpts2, pnp_3d_points, pnp_color_mkpts1, pnp_inliers = solve_pnp(candidate, filtered_mkpts1, filtered_mkpts2)
            #         end_pnp_time = time.perf_counter()
            #         print(f"PnP Time: {end_pnp_time - start_pnp_time:.2f} seconds ({1 / (end_pnp_time - start_pnp_time):.2f} fps)")
            #     except Exception as e:
            #         print(f"Failed to solve PnP for loop closure between {candidate.first_idx} and {candidate.second_idx} with {len(mkpts1)} inliers: {e}")
            #         continue

            #     if len(pnp_inliers) > min_inlier_count:
            #         pose_graph.add_between_pose_factor(candidate.first_idx, candidate.second_idx, first_to_second, loop_noise)
            #         # print(f"Added loop closure between {candidate.first_idx} and {candidate.second_idx} with {len(pnp_inliers)} inliers")

            #         map_points.append((candidate.first_idx, pnp_3d_points, pnp_color_mkpts1))
            #     else:
            #         # print(f"Failed to add loop closure between {candidate.first_idx} and {candidate.second_idx} with {len(pnp_inliers)} inliers")
            #         pass

            #     # TODO: log the error the relative to the ground truth relative pose? see if there's any correlations between various parts of bad loop closures?

            start_optimize_time = time.perf_counter()
            pose_graph.optimize()
            end_optimize_time = time.perf_counter()
            print(f"Optimize Time: {end_optimize_time - start_optimize_time:.2f} seconds ({1 / (end_optimize_time - start_optimize_time):.2f} fps)")

            # time.sleep(0.2) # sleep for a moment to simulate real-time odometry
        else:
            continue

        # logging
        start_logging_time = time.perf_counter()

        if len(gt_keyframe_trajectory) < 2:
            continue

        rr.set_time_sequence("frame", sequence=i)

        estimated_keyframe_trajectory: list[gtsam.Pose3] = []
        for idx in range(1, pose_graph.kf_idx + 1):
            key = X(idx)
            if pose_graph.values.exists(key):
                estimated_keyframe_trajectory.append(pose_graph.values.atPose3(key))
            else:
                estimated_keyframe_trajectory = []
                break

        assert len(estimated_keyframe_trajectory) == len(gt_keyframe_trajectory), f"Estimated keyframe trajectory length {len(estimated_keyframe_trajectory)} does not match ground truth keyframe trajectory length {len(gt_keyframe_trajectory)}"

        gt_keyframes_for_logging = gt_keyframe_trajectory
        if ALIGN_GT_KEYFRAMES and len(estimated_keyframe_trajectory) == len(gt_keyframe_trajectory):
            alignment_pose = compute_umeyama_alignment_pose(gt_keyframe_trajectory, estimated_keyframe_trajectory)
            if alignment_pose is not None:
                gt_keyframes_for_logging = [alignment_pose.compose(pose) for pose in gt_keyframe_trajectory]

        rr_log_trajectory("gt_keyframe_trajectory", gt_keyframes_for_logging, color=(0, 255, 0))
        # rr_log_trajectory("gt_trajectory", gt_trajectory, color=(0, 255, 0))
        # # rr_log_pose("gt", first_robot_to_robot, depth_frame_with_features)

        rr_log_trajectory("raw_keyframe_trajectory", raw_keyframe_trajectory, color=(0, 0, 255))
        # rr_log_trajectory("raw_trajectory", raw_trajectory, color=(0, 0, 255))
        # # rr_log_pose("raw", raw_trajectory[-1], depth_frame_with_features)

        # optimized_trajectory.append(pose_graph.get_latest_pose())
        # rr_log_trajectory("optimized_trajectory", optimized_trajectory, color=(255, 0, 0))
        rr_log_graph_edges(path="graph", nodes=pose_graph.values, graph=pose_graph.graph)
        # rr_log_pose(path="optimized", pose=pose_graph.get_latest_pose(), frame=depth_frame)

        # this might be causing perf issues? not sure
        # rr_log_map_points("map_points", pose_graph, map_points)

        # trans_ate = np.linalg.norm(gt_keyframe_trajectory[-1].translation() - pose_graph.get_latest_pose().translation())
        # rot_ate = np.rad2deg(np.linalg.norm(gt_keyframe_trajectory[-1].rotation().ypr() - pose_graph.get_latest_pose().rotation().ypr()))
        if len(gt_keyframes_for_logging) > 0 and len(gt_keyframes_for_logging) == len(estimated_keyframe_trajectory):
            trans_errors = [
                np.linalg.norm(
                    _pose_translation_to_array(gt_pose) - _pose_translation_to_array(est_pose)
                )
                for gt_pose, est_pose in zip(gt_keyframes_for_logging, estimated_keyframe_trajectory)
            ]
            trans_ate = float(np.mean(trans_errors))
            rot_errors = [
                np.linalg.norm(gt_pose.rotation().ypr() - est_pose.rotation().ypr())
                for gt_pose, est_pose in zip(gt_keyframes_for_logging, estimated_keyframe_trajectory)
            ]
            rot_ate = float(np.mean(rot_errors))

            rr.log("/translation/ate", rr.Scalar(trans_ate))
            rr.log("/rotation/ate", rr.Scalar(rot_ate))

        end_logging_time = time.perf_counter()
        print(f"Logging Time: {end_logging_time - start_logging_time:.2f} seconds ({1 / (end_logging_time - start_logging_time):.2f} fps)")

        end_time = time.perf_counter()
        print(f"Total Frame Processing Time: {end_time - start_time:.2f} seconds ({1 / (end_time - start_time):.2f} fps)")

# %%
