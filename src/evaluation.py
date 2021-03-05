import numpy as np
import joblib
import math
from sklearn.metrics import average_precision_score

def get_prediction(prediction):
    ''' Get prediction without threshold
    Args:
        prediction: the prediction from the model
    Returns:
        the binary classification result of each prediction
    '''
    predictions = []
    for p in prediction:
        tmp = np.argmax(p)
        if tmp == 0:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions

def get_prediction_prob(prediction, threshold=0.5):
    ''' Get prediction with threshold
    Args:
         prediction: the prediction from the model
         threshold: a threshold to filter the prediction
    Returns:
        the binary classification result of each prediction
    '''
    predictions = []
    for p in prediction:
        tmp = 1 / 1 + np.exp(-p[0])
        if tmp >= threshold:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions

def get_target_dict(imdb_ids, path, shot_num=4):
    ''' Get ground-truth dictionary
    Args:
        imdb_ids: a list of id of each movie
        path: train / validation / test
        shot_num: number of shots
    Returns:
        a dict of ground-truth of each movie
    '''
    target_dict = {}
    for id in imdb_ids:
        tmp = joblib.load(f'{path}/{id}.pkl')
        new_length = int(len(tmp['shot_end_frame']) // 100)
        target_dict[id] = tmp['scene_transition_boundary_ground_truth'][shot_num//2-1:new_length-shot_num//2]
        # target_dict[id] = tmp['scene_transition_boundary_ground_truth'](shot_num//2-1:len(tmp['shot_end_frame'])-shot_num//2)
    return target_dict

def get_shot_to_end_frame_dict(imdbs_ids, path):
    shot_to_end_frame_dict = {}
    for id in imdbs_ids:
        tmp = joblib.load(f'{path}/{id}.pkl')
        shot_to_end_frame_dict[id] = tmp['shot_end_frame']
    return shot_to_end_frame_dict

def calc_ap(gt_dict, pr_dict):
    """Average Precision (AP) for scene transitions.
    Args:
        gt_dict: Scene transition ground-truths.
        pr_dict: Scene transition predictions.
    Returns:
        AP, mean AP, and a dict of AP for each movie.
    """
    assert gt_dict.keys() == pr_dict.keys()

    AP_dict = dict()
    gt = list()
    pr = list()
    for imdb_id in gt_dict.keys():
        # AP_dict[imdb_id] = average_precision_score(gt_dict[imdb_id], pr_dict[imdb_id])
        score = average_precision_score(gt_dict[imdb_id], pr_dict[imdb_id])
        if math.isnan(score):
            AP_dict[imdb_id] = 0
        else:
            AP_dict[imdb_id] = score

        gt.append(gt_dict[imdb_id])
        pr.append(pr_dict[imdb_id])

    mAP = sum(AP_dict.values()) / len(AP_dict)

    gt = np.concatenate(gt)
    pr = np.concatenate(pr)
    AP = average_precision_score(gt, pr)

    return AP, mAP, AP_dict

def calc_miou(gt_dict, pr_dict, shot_to_end_frame_dict, threshold=0.5):
    """Maximum IoU (Miou) for scene segmentation.
    Miou measures how well the predicted scenes and ground-truth scenes overlap. The descriptions can be found in
    https://arxiv.org/pdf/1510.08893.pdf. Note the length of intersection or union is measured by the number of frames.
    Args:
        gt_dict: Scene transition ground-truths.
        pr_dict: Scene transition predictions.
        shot_to_end_frame_dict: End frame index for each shot.
        threshold: A threshold to filter the predictions.
    Returns:
        Mean MIoU, and a dict of MIoU for each movie.
    """
    def iou(x, y):
        s0, e0 = x
        s1, e1 = y
        smin, smax = (s0, s1) if s1 > s0 else (s1, s0)
        emin, emax = (e0, e1) if e1 > e0 else (e1, e0)
        return (emin - smax + 1) / (emax - smin + 1)

    def scene_frame_ranges(scene_transitions, shot_to_end_frame):
        end_shots = np.where(scene_transitions)[0]
        scenes = np.zeros((len(end_shots) + 1, 2), dtype=end_shots.dtype)
        scenes[:-1, 1] = shot_to_end_frame[end_shots]
        scenes[-1, 1] = shot_to_end_frame[len(scene_transitions)]
        scenes[1:, 0] = scenes[:-1, 1] + 1
        return scenes

    def miou(gt_array, pr_array, shot_to_end_frame):
        gt_scenes = scene_frame_ranges(gt_array, shot_to_end_frame)
        # pr_scenes = scene_frame_ranges(pr_array >= threshold, shot_to_end_frame)
        pr_scenes = scene_frame_ranges(pr_array, shot_to_end_frame)
        assert gt_scenes[-1, -1] == pr_scenes[-1, -1]

        m = gt_scenes.shape[0]
        n = pr_scenes.shape[0]

        # IoU for (gt_scene, pr_scene) pairs
        iou_table = np.zeros((m, n))

        j = 0
        for i in range(m):
            # j start prior to i end
            while pr_scenes[j, 0] <= gt_scenes[i, 1]:
                iou_table[i, j] = iou(gt_scenes[i], pr_scenes[j])
                if j < n - 1:
                    j += 1
                else:
                    break
            # j end prior to (i + 1) start
            if pr_scenes[j, 1] < gt_scenes[i, 1] + 1:
                break
            # j start later than (i + 1) start
            if pr_scenes[j, 0] > gt_scenes[i, 1] + 1:
                j -= 1
        assert np.isnan(iou_table).sum() == 0
        assert iou_table.min() >= 0

        # Miou
        return (iou_table.max(axis=0).mean() + iou_table.max(axis=1).mean()) / 2

    assert gt_dict.keys() == pr_dict.keys()

    miou_dict = dict()

    for imdb_id in gt_dict.keys():
        miou_dict[imdb_id] = miou(gt_dict[imdb_id], pr_dict[imdb_id], shot_to_end_frame_dict[imdb_id])
    mean_miou = sum(miou_dict.values()) / len(miou_dict)

    return mean_miou, miou_dict


