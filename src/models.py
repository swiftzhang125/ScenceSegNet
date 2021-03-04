import torch
from torch import nn
from torch.nn import functional as F

def loss_fn(predictions, targets, masks):
    ce = nn.CrossEntropyLoss()
    active_loss = masks.view(-1) == 1
    active_logits = predictions.view(-1, 2)
    active_lables = torch.where(
        active_loss,
        targets.view(-1),
        torch.tensor(ce.ignore_index).type_as(targets)
    )
    loss = ce(active_logits, active_lables)
    return loss

class Cos(nn.Module):
    def __init__(self, shot_num=4, sim_channel=512):
        super(Cos, self).__init__()
        self.shot_num = shot_num
        self.channel = sim_channel
        self.conv1 = nn.Conv2d(1, self.channel, kernel_size=(self.shot_num//2, 1))

    def forward(self, x):
        x = x.view(-1, 1, x.shape[2], x.shape[3])
        part1, part2 = torch.split(x, [self.shot_num // 2] * 2, dim=2)
        part1 = self.conv1(part1).squeeze()
        part2 = self.conv1(part2).squeeze()
        x = F.cosine_similarity(part1, part2, dim=2)
        return x

class Bnet(nn.Module):
    def __init__(self, shot_num=4, sim_channel=512):
        super(Bnet, self).__init__()
        self.shot_num = shot_num
        self.channel = sim_channel
        self.conv1 = nn.Conv2d(1, self.channel, kernel_size=(self.shot_num, 1))
        self.max3d = nn.MaxPool3d(kernel_size=(self.channel, 1, 1))
        self.cos = Cos(shot_num=self.shot_num, sim_channel=self.channel)

    def forward(self, x):
        context = x.view(x.shape[0] * x.shape[1], 1, -1, x.shape[-1])
        context = self.conv1(context)
        context = self.max3d(context)
        context = context.squeeze()
        sim = self.cos(x)
        bound = torch.cat((context, sim), dim=1)
        return bound

class LGSSone(nn.Module):
    def __init__(self, mode='place', seq_len=10, shot_num=4, sim_channel=512, num_layers=1, lstm_hidden_size=512, bidirectional=True):
        super(LGSSone, self).__init__()
        self.seq_len = seq_len
        self.shot_num = shot_num
        self.channel = sim_channel
        self.num_layers = num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.bidirectional = bidirectional
        if mode == 'place':
            self.input_dim = 2048 + self.channel
            self.bnet = Bnet(shot_num=self.shot_num, sim_channel=self.channel)
        elif mode == 'cast':
            self.input_dim = 512 + self.channel
            self.bnet = Bnet(shot_num=self.shot_num, sim_channel=self.channel)
        elif mode == 'action':
            self.input_dim = 512 + self.channel
            self.bnet = Bnet(shot_num=self.shot_num, sim_channel=self.channel)
        elif mode == 'audio':
            self.input_dim = 512 + self.channel
            self.bnet = Bnet(shot_num=self.shot_num, sim_channel=self.channel)
        else:
            pass
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.lstm_hidden_size,
                            batch_first=True,
                            bidirectional=self.bidirectional)
        if self.bidirectional:
            self.fc1 = nn.Linear(self.lstm_hidden_size*2, 100)
        else:
            self.fc1 = nn.Linear(self.lstm_hidden_size, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.bnet(x)
        x = x.view(-1, self.seq_len, x.shape[-1])
        self.lstm.flatten_parameters()
        out, (_, _) = self.lstm(x, None)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = out.view(-1, 2)
        return out

class LGSS(nn.Module):
    def __init__(self, seq_len=10, shot_num=4, sim_channel=512, ratio=(0.5, 0.2, 0.2, 0.1), num_layers=1, lstm_hidden_size=512, bidirectional=True):
        super(LGSS, self).__init__()
        self.seq_len = seq_len
        self.shot_num = shot_num
        self.channel = sim_channel
        self.ratio = ratio
        self.num_layers = num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.bidirectional = bidirectional
        self.bnet_place = LGSSone(mode='place', seq_len=self.seq_len, shot_num=self.shot_num,
                                  sim_channel=self.channel, num_layers=self.num_layers,
                                  lstm_hidden_size=self.lstm_hidden_size, bidirectional=self.bidirectional)
        self.bnet_cast = LGSSone(mode='cast', seq_len=self.seq_len, shot_num=self.shot_num,
                                 sim_channel=self.channel, num_layers=self.num_layers,
                                 lstm_hidden_size=self.lstm_hidden_size, bidirectional=self.bidirectional)
        self.bnet_action = LGSSone(mode='action', seq_len=self.seq_len, shot_num=self.shot_num,
                                   sim_channel=self.channel, num_layers=self.num_layers,
                                   lstm_hidden_size=self.lstm_hidden_size, bidirectional=self.bidirectional)
        self.bnet_audio = LGSSone(mode='audio', seq_len=self.seq_len, shot_num=self.shot_num,
                                  sim_channel=self.channel, num_layers=self.num_layers,
                                  lstm_hidden_size=self.lstm_hidden_size, bidirectional=self.bidirectional)

    def forward(self, place_feats, cast_feats, action_feats, audio_feats, targets, masks):
        place_bound = self.bnet_place(place_feats)
        cast_bound = self.bnet_cast(cast_feats)
        action_bound = self.bnet_action(action_feats)
        audio_bound = self.bnet_audio(audio_feats)
        out = self.ratio[0] * place_bound + self.ratio[1] * cast_bound + self.ratio[2] * action_bound + self.ratio[3] * audio_bound
        loss = loss_fn(out, targets, masks)
        return out, loss

if __name__ == '__main__':
    '''
    model = LGSSone()
    place_feat = torch.randn(2, 15, 4, 2048)
    output = model(place_feat)
    '''
    model = LGSS()
    place_feats = torch.randn(2, 50, 4, 2048)
    cast_feats = torch.randn(2, 50, 4, 512)
    act_feats = torch.randn(2, 50, 4, 512)
    aud_feats = torch.randn(2, 50, 4, 512)
    targets = torch.tensor([[[0] * 10 + [1] * 40], [[0] * 15 + [1] * 35]])
    print(targets.size())
    masks = torch.tensor([[[1] * 30 + [0] * 20], [[1] * 40 + [0] * 10]])
    print(masks.size())
    output, loss = model(place_feats, cast_feats, act_feats, aud_feats, targets, masks)
    print(output.data[0])
    print(output.data.size())
    print(loss.item())









