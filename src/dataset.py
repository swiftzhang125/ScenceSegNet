import joblib
import glob
import torch
from torch.utils.data import DataLoader
import numpy as np

class SceneSegDataset():
    def __init__(self, path, shot_num=4, max_len=51):
        self.ids = glob.glob(f'../input/data/{path}/*.pkl')
        self.shot_num = shot_num
        self.shot_boundary_range = range(-self.shot_num//2+1, self.shot_num//2+1)
        self.max_len = max_len

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        tmp = joblib.load(f'{self.ids[item]}')
        place_feats, cast_feats, action_feats, audio_feats, targets, masks = [], [], [], [], [], []
        places = np.array(tmp['place'])
        casts = np.array(tmp['cast'])
        actions = np.array(tmp['action'])
        audios = np.array(tmp['audio'])
        labels = np.array(tmp['scene_transition_boundary_ground_truth'])
        new_length = int(len(tmp['shot_end_frame'])//100)
        for id in range(self.shot_num//2-1, new_length-self.shot_num//2):
        # for id in range(self.shot_num//2-1, len(tmp['shot_end_frame'])-self.shot_num//2):
            place_feat, cast_feat, action_feat, audio_feat = [], [], [], []
            for ind in self.shot_boundary_range:
                place_feat.append(places[id + ind, :])
                cast_feat.append(casts[id + ind, :])
                action_feat.append(actions[id + ind, :])
                audio_feat.append(audios[id + ind, :])
            place_feats.append(place_feat)
            cast_feats.append(cast_feat)
            action_feats.append(action_feat)
            audio_feats.append(audio_feat)
        for i in range(self.shot_num//2-1, new_length-self.shot_num//2):
        # for i in range(self.shot_num//2-1, len(tmp['shot_end_frame'])-self.shot_num//2):
            if labels[i]:
                targets.append([1])
            else:
                targets.append([0])
            masks.append([1])

        # padding
        place_padding = np.zeros((4, 2048))
        cast_padding = np.zeros((4, 512))
        audio_padding = np.zeros((4, 512))
        action_padding = np.zeros((4, 512))

        for i in range(new_length-self.shot_num//2, self.max_len):
        # for _ in range(len(tmp['shot_end_frame'])-self.shot_num//2, self.max_len):
            place_feats.append(place_padding)
            cast_feats.append(cast_padding)
            audio_feats.append((audio_padding))
            action_feats.append((action_padding))
            targets.append([0])
            masks.append([0])

        return {
            'place_feats': torch.tensor(place_feats, dtype=torch.float),
            'cast_feats': torch.tensor(cast_feats, dtype=torch.float),
            'action_feats': torch.tensor(action_feats, dtype=torch.float),
            'audio_feats': torch.tensor(audio_feats, dtype=torch.float),
            'targets': torch.tensor(targets, dtype=torch.long),
            'masks': torch.tensor(masks, dtype=torch.long),
            'imdb_id': tmp['imdb_id'],
        }





if __name__ == '__main__':
    '''
    ids = glob.glob(f'../input/data/test/*.pkl')
    print(ids)
    a = joblib.load(f'../input/data/data/tt0052357.pkl')
    print(a.keys())
    print(a['place'].shape)
    print(a['cast'].shape)
    print(a['audio'].shape)
    print(a['scene_transition_boundary_ground_truth'][4])
    print(len(a['scene_transition_boundary_ground_truth']))
    print(len(a['shot_end_frame']))
    '''
    testset = SceneSegDataset('train')
    dataloader = DataLoader(testset, batch_size=2, shuffle=False)
    print(len(dataloader))
    for batch_idx, dic in enumerate(dataloader):
        print('place_feats', dic['place_feats'].shape)
        print('cast_feats', dic['cast_feats'].shape)
        print('audio_feats', dic['audio_feats'].shape)
        print('action_feats', dic['action_feats'].shape)
        print('targets', dic['targets'].shape)
        print('imdb_id', dic['imdb_id'])
        # print('targets', dic['targets'])
        print('masks', dic['masks'].shape)
        break
    '''
    a = joblib.load(f'../input/data/data/tt0976051.pkl')
    print(len(a['scene_transition_boundary_ground_truth']))
    print(a['cast'].shape)
    print(a['scene_transition_boundary_ground_truth'][-1])

    b = joblib.load(f'../input/data/data/tt1001508.pkl')
    print(len(b['scene_transition_boundary_ground_truth']))
    print(b['cast'].shape)
    print(b['scene_transition_boundary_ground_truth'][-1])
    '''

    a = joblib.load(f'../input/data/data/tt1001508.pkl')
    print(a['cast'].shape)






