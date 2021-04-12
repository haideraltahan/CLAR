import os
import os.path as osp
import pickle

import fire
import lmdb
from tqdm import tqdm
from utils import get_data_loader, AudioFolder


def folder2lmdb(dataset, outputFile, write_frequency=5000, num_workers=0 if os.name == 'nt' else 60):
    data_loader = get_data_loader(dataset, batch_size=1, num_workers=num_workers)

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
    # split_folders.ratio(Dataset_Path, output=Dataset_Path, seed=1337, ratio=(.8, .1, .1))
    train_dataset = AudioFolder(Dataset_Path)
    folder2lmdb(train_dataset, osp.join(Dataset_Path, "%s.lmdb" % OutputFile), num_workers=num_workers)


if __name__ == '__main__':
    fire.Fire(DatasetConverter)
