import os
import os.path as osp
import pickle

import fire
import librosa
import lmdb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import find_files


class AudioFolder(Dataset):
    def __init__(self, data_dir, fps=16000):
        # define the state of the object
        self.data_dir = data_dir
        self.fps = fps
        # setup the files for reading
        self.files = list(find_files(data_dir, '*.wav'))
        self.classes = [f for f in sorted(os.listdir(data_dir))]
        # self.classes.remove('.DS_Store')
        self.classes.sort()
        print(self.classes)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio = self.files[idx][0]
        name = self.files[idx][0].split('/')[-2]
        try:
            audio = librosa.load(audio, 16000)[0]
        except:
            print(self.files[idx][0])
        return audio, self.classes.index(name)


def folder2lmdb(dataset, outputFile, write_frequency=5000, num_workers=0 if os.name == 'nt' else 60):
    data_loader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=num_workers)

    print("Generate LMDB to %s" % outputFile)
    db = lmdb.open(outputFile, subdir=os.path.isdir(outputFile),
                   map_size=10e+11, readonly=False,
                   meminit=False, map_async=True)

    idx = 0
    txn = db.begin(write=True)
    for audio, label in tqdm(data_loader):
        txn.put(u'{}'.format(idx).encode('ascii'), pickle.dumps((audio, label)))
        idx += 1
        if idx % write_frequency == 0:
            txn.commit()
            txn = db.begin(write=True)
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys))
        txn.put(b'__len__', pickle.dumps(len(keys)))
    print("Flushing database ...")
    db.sync()
    db.close()


def DatasetConverter(Dataset_Path, OutputFile='speech', num_workers=0 if os.name == 'nt' else 60):
    dataset = AudioFolder(Dataset_Path)
    folder2lmdb(dataset, osp.join(Dataset_Path, "%s.lmdb" % OutputFile), num_workers=num_workers)


if __name__ == '__main__':
    fire.Fire(DatasetConverter)
