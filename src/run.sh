export SEQ_LEN=10
export SHOT_NUM=4
export SIM_CHANNEL=512
export RATIO='(0.5,0.2,0.2,0.1)'
export NUM_LAYERS=1
export LSTM_HIDDEN_SIZE=512
export BIDIRECTIONAL='True'
export MAX_LEN=51
export TRAIN_BATCH_SIZE=2
export VALID_BATCH_SIZE=2
export VALID_PATH='../input/data/validation'
export EPOCHS=3
export LR=1e-2
export OPT=2
export THRESHOLD=5e-1

python train.py