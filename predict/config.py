'''
    Parameter setting must be as same as train.py
'''
MAX_LEN = 51
SHOT_NUM = 4
SEQ_LEN = 10
SIM_CHANNEL = 512
RATIO = [0.5, 0.2, 0.2, 0.1]
NUM_LAYERS = 1
LSTM_HIDDEN_SIZE = 512
BIDIRECTIONAL = True
MODEL_PATH = '../input/model/SenceSegNet_epoch'
TEST_PATH = '../input/data/test'
THRESHOLD = 0.5