import joblib
import glob
from tqdm import tqdm

'''Only run once
Spilt train : validation : test = 10 : 3 : 3

original data: ../input/data/data/*.pkl

train : ../input/data/train/imdb_id.pkl
validation : ../input/data/validation/imdb_id.pkl
test : ../input/data/test/imdb_id.pkl
'''

if __name__ == '__main__':
    files = glob.glob('../input/data/data/tt*.pkl')
    length = len(files)
    for id, f in tqdm(enumerate(files), total=len(files)):
        tmp = joblib.load(f'{f}')
        f_name = f.split('\\')[-1]
        if id < 10 * int(length/16):
            path = 'train'
        elif id < 13 * int(length/16):
            path = 'validation'
        else:
            path = 'test'

        joblib.dump(tmp, f'../input/data/{path}/{f_name}')
    print('split file ->  train:val:test = 10:3:3')