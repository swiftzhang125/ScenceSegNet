import os
import ast
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from dataset import SceneSegDataset
from models import LGSS
from evaluation import get_prediction, get_prediction_prob, get_target_dict, get_shot_to_end_frame_dict, calc_ap, calc_miou

import warnings
warnings.filterwarnings('ignore')

DEVICE = 'cuda'
# DEVICE = 'cpu'
SEQ_LEN = int(os.environ.get('SEQ_LEN'))
SHOT_NUM = int(os.environ.get('SHOT_NUM'))
SIM_CHANNEL = int(os.environ.get('SIM_CHANNEL'))
RATIO = ast.literal_eval(os.environ.get('RATIO'))
NUM_LAYERS = int(os.environ.get('NUM_LAYERS'))
LSTM_HIDDEN_SIZE = int(os.environ.get('LSTM_HIDDEN_SIZE'))
BIDIRECTIONAL = bool(os.environ.get('LSTM_HIDDEN_SIZE'))
MAX_LEN = int(os.environ.get('MAX_LEN'))
TRAIN_BATCH_SIZE = int(os.environ.get('TRAIN_BATCH_SIZE'))
VALID_BATCH_SIZE = int(os.environ.get('VALID_BATCH_SIZE'))
VALID_PATH = os.environ.get('VALID_PATH')
EPOCHS = int(os.environ.get('EPOCHS'))
LR = float(os.environ.get('LR'))
OPT = int(os.environ.get('OPT'))
THRESHOLD = float(os.environ.get('THRESHOLD'))

'''
DEVICE = 'cuda'
SEQ_LEN = 10
SHOT_NUM = 4
SIM_CHANNEL = 512
RATIO = [0.5, 0.2, 0.2, 0.1]
NUM_LAYERS = 1
LSTM_HIDDEN_SIZE = 512
BIDIRECTIONAL = True
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 2
EPOCHS = 3
LR = 0.01
THRESHOLD = 0.5
'''
def train_loop_fn(dataset, dataloader, model, optimizer):
    model.train()
    loss_sum = 0
    counter = 0
    for step, d in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
        counter += 1
        place_feats = d['place_feats']
        cast_feats = d['cast_feats']
        action_feats = d['action_feats']
        audio_feats = d['audio_feats']
        targets = d['targets']
        masks = d['masks']

        place_feats = place_feats.to(DEVICE, dtype=torch.float)
        cast_feats = cast_feats.to(DEVICE, dtype=torch.float)
        action_feats = action_feats.to(DEVICE, dtype=torch.float)
        audio_feats = audio_feats.to(DEVICE, dtype=torch.float)
        targets = targets.to(DEVICE, dtype=torch.long)
        masks = masks.to(DEVICE, dtype=torch.long)

        optimizer.zero_grad()

        _, loss = model(place_feats, cast_feats, action_feats, audio_feats, targets, masks)

        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
    return loss_sum / counter

def eval_loop_fn(dataset, dataloader, model):
    model.eval()
    loss_sum = 0
    counter = 0
    prediction_dict = {}
    prediction_withthreshold_dict = {}
    imdb_ids_all = []
    for step, d in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
        counter += 1
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

        predictions, loss = model(place_feats, cast_feats, action_feats, audio_feats, targets, masks)
        loss_sum += loss.item()

        prediction = torch.chunk(predictions, len(imdb_ids), dim=0)
        mask = torch.chunk(masks, len(imdb_ids), dim=0)
        for i in range(len(imdb_ids)):
            ind = list(mask[i].squeeze().cpu().detach()).index(0)
            prediction_dict[imdb_ids[i]] = get_prediction(np.array(prediction[i][:ind].squeeze().cpu().detach()))
            prediction_withthreshold_dict[imdb_ids[i]] = get_prediction_prob(np.array(prediction[i][:ind].squeeze().cpu().detach()), threshold=THRESHOLD)
            imdb_ids_all.append(imdb_ids[i])

    target_dict = get_target_dict(imdb_ids_all, path=VALID_PATH, shot_num=SHOT_NUM)
    _, mAP, _ = calc_ap(target_dict, prediction_dict)

    shot_to_end_frame_dict = get_shot_to_end_frame_dict(imdb_ids_all, path=VALID_PATH)
    mean_miou, _ = calc_miou(target_dict, prediction_withthreshold_dict, shot_to_end_frame_dict)

    return loss_sum / counter, mAP, mean_miou


def run():
    model = LGSS(seq_len=SEQ_LEN, shot_num=SHOT_NUM, sim_channel=SIM_CHANNEL,
                 ratio=RATIO, num_layers=NUM_LAYERS,
                 lstm_hidden_size=LSTM_HIDDEN_SIZE, bidirectional=BIDIRECTIONAL)
    model.to(DEVICE)

    train_dataset = SceneSegDataset('train', shot_num=SHOT_NUM, max_len=MAX_LEN)
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=1
    )

    valid_dataset = SceneSegDataset('validation', shot_num=SHOT_NUM, max_len=MAX_LEN)
    valid_dataloader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=VALID_BATCH_SIZE,
        shuffle=True,
        num_workers=1
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    dfhistory = pd.DataFrame(columns=['epoch', 'train_loss', 'validation_loss', 'mAP', 'mean_miou'])
    for epoch in range(1, EPOCHS + 1):
        if epoch == OPT:
            optimizer = torch.optim.Adam(model.parameters(), lr=LR / 10)
        train_loss = train_loop_fn(train_dataset, train_dataloader, model, optimizer)

        valid_loss, mAP, mean_miou = eval_loop_fn(valid_dataset, valid_dataloader, model)

        print(f'epoch: {epoch}, train_loss: {train_loss}, validation_loss: {valid_loss}, mAP: {mAP}, mean_miou: {mean_miou}')
        info = (int(epoch), train_loss, valid_loss, mAP, mean_miou)
        dfhistory.loc[epoch-1] = info
        torch.save(model.state_dict(), f'../input/model/SenceSegNet_epoch{epoch}.bin')
    dfhistory.to_csv('../input/model/dfhistory.csv', index=False)


if __name__ == '__main__':
    run()