import torch
import config
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
sys.path.append('../src')
from dataset import SceneSegDataset
from models import LGSS
from evaluation import get_prediction, get_prediction_prob, get_target_dict, get_shot_to_end_frame_dict, calc_ap, calc_miou

import warnings
warnings.filterwarnings('ignore')

DEVICE = 'cuda'
# DEVICE = 'cpu'

def find_best_model(metric='validation_loss', last=True):
    dfhistory = pd.read_csv('../input/model/dfhistory.csv')
    inds = []
    if metric == 'validation_loss':
        tmp = np.array(dfhistory[metric])
        inds = np.argwhere(tmp == np.amin(tmp))
    elif metric == 'mAP':
        tmp = np.array(dfhistory[metric])
        inds = np.argwhere(tmp == np.amax(tmp))
    elif metric == 'mean_miou':
        tmp = np.array(dfhistory[metric])
        inds = np.argwhere(tmp == np.amax(tmp))
    else:
        pass
    if len(inds) and last:
        ind = inds[-1][0]
    elif len(inds) and not last:
        ind = inds[0][0]
    else:
        print('Have found the metric! Will use the model from the last training epoch!')
        print(dfhistory.shape)
        ind = dfhistory.shape[0] - 1
    return ind + 1


def predict(metric='validation_loss', last=True):
    ind = find_best_model(metric=metric, last=last)
    print(f'Using parameters at {ind} epoch!')
    test_dataset = SceneSegDataset('test', shot_num=config.SHOT_NUM, max_len=config.MAX_LEN)
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1
    )

    model = LGSS(seq_len=config.SEQ_LEN, shot_num=config.SHOT_NUM, sim_channel=config.SIM_CHANNEL,
                 ratio=config.RATIO, num_layers=config.NUM_LAYERS,
                 lstm_hidden_size=config.LSTM_HIDDEN_SIZE, bidirectional=config.BIDIRECTIONAL)
    path = config.MODEL_PATH + str(ind) + '.bin'
    model.load_state_dict(torch.load(path))
    model.to(DEVICE)

    with torch.no_grad():
        prediction_dict = {}
        prediction_withthreshold_dict = {}
        imdb_ids_all = []
        for step, d in tqdm(enumerate(test_dataloader), total=int(len(test_dataset) / test_dataloader.batch_size)):
            place_feats = d['place_feats']
            cast_feats = d['cast_feats']
            audio_feats = d['audio_feats']
            action_feats = d['action_feats']
            targets = d['targets']
            masks = d['masks']
            imdb_ids = d['imdb_id']

            place_feats = place_feats.to(DEVICE, dtype=torch.float)
            cast_feats = cast_feats.to(DEVICE, dtype=torch.float)
            audio_feats = audio_feats.to(DEVICE, dtype=torch.float)
            action_feats = action_feats.to(DEVICE, dtype=torch.float)
            targets = targets.to(DEVICE, dtype=torch.long)
            masks = masks.to(DEVICE, dtype=torch.long)

            predictions, _ = model(place_feats, cast_feats, action_feats, audio_feats, targets, masks)

            prediction = torch.chunk(predictions, len(imdb_ids), dim=0)
            mask = torch.chunk(masks, len(imdb_ids), dim=0)
            for i in range(len(imdb_ids)):
                ind = list(mask[i].squeeze().cpu().detach()).index(0)
                prediction_dict[imdb_ids[i]] = get_prediction(np.array(prediction[i][:ind].squeeze().cpu().detach()))
                prediction_withthreshold_dict[imdb_ids[i]] = get_prediction_prob(np.array(prediction[i][:ind].squeeze().cpu().detach()), threshold=config.THRESHOLD)
                imdb_ids_all.append(imdb_ids[i])

        target_dict = get_target_dict(imdb_ids_all, path=config.TEST_PATH)
        _, mAP, AP_dict = calc_ap(target_dict, prediction_dict)

        shot_to_end_frame_dict = get_shot_to_end_frame_dict(imdb_ids_all, path=config.TEST_PATH)
        mean_miou, miou_dict = calc_miou(target_dict, prediction_withthreshold_dict, shot_to_end_frame_dict)

    print('Test Result: ')
    print(f'mAP: {mAP}, mean_miou : {mean_miou}')
    for id in imdb_ids_all:
        print(f'imdb_id: {id}, AP: {AP_dict[id]}, miou: {miou_dict[id]}')
    return mAP, AP_dict, mean_miou, miou_dict

if __name__ == '__main__':
    predict(metric='validation_loss', last=True)