import fnmatch
import json
import math
import os
import os.path as osp
import pickle
import random
import re
from collections import OrderedDict

import librosa
import lmdb
import numpy as np
import pytorch_lightning as pl
import torch
from librosa import effects
from pytorch_lightning.metrics.functional import accuracy
from torch.nn import functional as F
from torch.utils.data import Dataset


def get_spec_features(x, sr=16000):
    return np.float32(np.stack((
        librosa.power_to_db(np.abs(librosa.feature.melspectrogram(x, sr, hop_length=128))),
        librosa.amplitude_to_db(librosa.feature.mfcc(x, sr, n_mfcc=128, hop_length=128)),
        librosa.amplitude_to_db(np.abs(librosa.stft(x, n_fft=254, hop_length=128)), ref=np.max)
    ), 0))


def pad(x, sr=16000, length=1):
    pad_length = (sr * length) - x.shape[-1]
    if pad_length == 0:
        return x.squeeze().numpy()
    elif pad_length < 0:
        return x[:, :(sr * length)].squeeze().numpy()
    return torch.nn.functional.pad(x, [0, pad_length, 0, 0]).squeeze().numpy()


def pitch_shift(x, sr=16000, n_steps=15):
    return effects.pitch_shift(x, sr, torch.randint(low=-n_steps, high=n_steps, size=[1]).item())


def add_white_noise(x, max_SNR=20):
    SNR = torch.randint(low=0, high=max_SNR, size=[1])
    RMS_s = math.sqrt(np.mean(x ** 2))
    RMS_n = math.sqrt(RMS_s ** 2 / (pow(10., SNR / 10.)))
    noise = np.random.normal(0, RMS_n, x.shape[0])
    return x + np.float32(noise)


def add_noise(x, y, sr=16000):
    start = np.random.randint(y.shape[0] - sr)
    y = y[start: start + sr]
    return np.clip(np.float32(x * np.random.uniform(0.5, 1.2, x.shape) + y * np.random.uniform(0, .9, y.shape)), 0, 1)


def add_fade(x, max_fade_size=.5):
    def _fade_in(fade_shape, waveform_length, fade_in_len):
        fade = np.linspace(0, 1, fade_in_len)
        ones = np.ones(waveform_length - fade_in_len)
        if fade_shape == 0:
            fade = fade
        if fade_shape == 1:
            fade = np.power(2, (fade - 1)) * fade
        if fade_shape == 2:
            fade = np.log10(.1 + fade) + 1
        if fade_shape == 3:
            fade = np.sin(fade * math.pi / 2)
        if fade_shape == 4:
            fade = np.sin(fade * math.pi - math.pi / 2) / 2 + 0.5
        return np.clip(np.concatenate((fade, ones)), 0, 1)

    def _fade_out(fade_shape, waveform_length, fade_out_len):
        fade = torch.linspace(0, 1, fade_out_len)
        ones = torch.ones(waveform_length - fade_out_len)
        if fade_shape == 0:
            fade = - fade + 1
        if fade_shape == 1:
            fade = np.power(2, - fade) * (1 - fade)
        if fade_shape == 2:
            fade = np.log10(1.1 - fade) + 1
        if fade_shape == 3:
            fade = np.sin(fade * math.pi / 2 + math.pi / 2)
        if fade_shape == 4:
            fade = np.sin(fade * math.pi + math.pi / 2) / 2 + 0.5
        return np.clip(np.concatenate((ones, fade)), 0, 1)

    waveform_length = x.shape[0]
    fade_shape = np.random.randint(5)
    fade_out_len = np.random.randint(int(x.shape[0] * max_fade_size))
    fade_in_len = np.random.randint(int(x.shape[0] * max_fade_size))
    return np.float32(
        _fade_in(fade_shape, waveform_length, fade_in_len) * _fade_out(fade_shape, waveform_length, fade_out_len) * x)


def time_masking(x, sr=0.125):
    if torch.randint(low=0, high=2, size=[1]).item() == 0:
        sr = int(x.shape[0] * sr)
        start = np.random.randint(x.shape[0] - sr)
        x[start: start + sr] = np.float32(np.random.normal(0, 0.01, sr))
    return x


def time_shift(x, shift_rate=8000):
    return np.roll(x, torch.randint(low=-shift_rate, high=shift_rate, size=[1]).item())


def time_stret(x, sr=16000, length=1):
    x = effects.time_stretch(x, np.random.uniform(.5, 1.5, [1])[0])
    x = librosa.resample(x, x.shape[0] / length, sr)
    if x.shape[0] > (sr * length):
        return x[:(sr * length)]
    return np.pad(x, [0, (sr * length) - x.shape[0]])


class Mixed_Noise(torch.nn.Module):
    def __init__(self,
                 data_dir,
                 sr=16000
                 ):
        super(Mixed_Noise, self).__init__()
        self.sr = sr

        self.brown_wav, _ = librosa.load(osp.join(data_dir, 'brown_noise.wav'), sr)
        self.pink_wav, _ = librosa.load(osp.join(data_dir, 'pink_noise.wav'), sr)
        self.fan_noise, _ = librosa.load(osp.join(data_dir, 'fan_noise.wav'), sr)

    def forward(self, x):
        i = torch.randint(low=0, high=5, size=[1]).item()
        if i == 0:
            x = add_noise(x, self.brown_wav, self.sr)
        elif i == 1:
            x = add_noise(x, self.pink_wav, self.sr)
        elif i == 2:
            x = add_noise(x, self.fan_noise, self.sr)
        elif i == 3:
            x = add_white_noise(x)
        return x


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield (filename, basename)


class AudioFolder(Dataset):
    """ pyTorch Dataset wrapper for the generic flat directory dataset """

    def __init__(self, data_dir, fps=16000):
        """
        constructor for the class
        :param data_dir: path to the directory containing the data
        :param transform: transforms to be applied to the images
        """
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
        """
        compute the length of the dataset
        :return: len => length of dataset
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        obtain the image (read and transform)
        :param idx: index of the file required
        :return: img => image array
        """
        audio = self.files[idx][0]
        name = self.files[idx][0].split('/')[-2]
        # print(name)

        try:
            audio = librosa.load(audio, 16000)[0]
        except:
            print(self.files[idx][0])
        return audio, self.classes.index(name)


def get_data_loader(dataset, batch_size, num_workers):
    """
    generate the data_loader from the given dataset
    :param dataset: F2T dataset
    :param batch_size: batch size of the data
    :param num_workers: num of parallel readers
    :return: dl => dataloader for the dataset
    """
    from torch.utils.data import DataLoader

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )

    return dl


def nt_xent_loss(out_1, out_2, temperature):
    """
    Loss used in SimCLR
    """
    out = torch.cat([out_1, out_2], dim=0)
    n_samples = len(out)

    # Full similarity matrix
    cov = torch.mm(out, out.t().contiguous())
    sim = torch.exp(cov / temperature)

    # Negative similarity
    mask = ~torch.eye(n_samples, device=sim.device).bool()
    neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

    # Positive similarity :
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)
    loss = -torch.log(pos / neg).mean()

    return loss


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def get_last(model_names, num=-1):
    if 'last.ckpt' in model_names:
        model_names.remove('last.ckpt')
    model_l = list(map(lambda x: int(re.findall('\d+', x)[-1]), model_names))
    model_names = zip(model_names, model_l)
    model_names = sorted(model_names, key=lambda x: x[1])
    return list(model_names)[num][0]


def get_all_checkpoints(model_names):
    if 'last.ckpt' in model_names:
        model_names.remove('last.ckpt')
    model_l = list(map(lambda x: int(re.findall('\d+', x)[-1]), model_names))
    model_names = zip(model_names, model_l)
    model_names = sorted(model_names, key=lambda x: x[1])
    return list(model_names)


class SSLOnlineEvaluator(pl.Callback):  # pragma: no-cover

    def __init__(self, drop_p: float = 0.2, hidden_dim: int = 512, z_dim: int = None, num_classes: int = None,
                 data_dir='', dataset='', train_transform=None, batch_size=128, nsynth_class=''):
        """
        Attaches a MLP for finetuning using the standard self-supervised protocol.
        Example::
            from pl_bolts.callbacks.self_supervised import SSLOnlineEvaluator
            # your model must have 2 attributes
            model = Model()
            model.z_dim = ... # the representation dim
            model.num_classes = ... # the num of classes in the model
        Args:
            drop_p: (0.2) dropout probability
            hidden_dim: (1024) the hidden dimension for the finetune MLP
        """
        super().__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.drop_p = drop_p
        self.optimizer = None
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.output = OrderedDict()
        if 'esc' in dataset:
            data_path = osp.join(data_dir, dataset + ".lmdb")
            self.train_dataset = NormalLMDB_ESC(data_path, transform=train_transform,
                                                aug_transform=None,
                                                spec_transform=None,
                                                folds=[1, 2, 3, 4])
        elif 'nsynth' in dataset:
            data_path = osp.join(data_dir, "train_" + dataset + ".lmdb")
            self.train_dataset = NormalLMDBG_NSYNTH(data_path, transform=train_transform,
                                                    aug_transform=None,
                                                    spec_transform=None,
                                                    target=nsynth_class,
                                                    perc=1)
        else:
            data_path = osp.join(data_dir, "train_" + dataset + ".lmdb")
            self.train_dataset = NormalLMDBG(data_path, transform=train_transform,
                                             aug_transform=None,
                                             spec_transform=None,
                                             perc=1
                                             )

    def on_pretrain_routine_start(self, trainer, pl_module):
        from pl_bolts.models.self_supervised.evaluator import SSLEvaluator

        # attach the evaluator to the module

        if hasattr(pl_module, 'z_dim'):
            self.z_dim = pl_module.z_dim
        if hasattr(pl_module, 'num_classes'):
            self.num_classes = pl_module.num_classes

        pl_module.non_linear_evaluator = SSLEvaluator(
            n_input=self.z_dim,
            n_classes=self.num_classes,
            p=self.drop_p,
            n_hidden=self.hidden_dim
        ).to(pl_module.device)

        self.optimizer = torch.optim.SGD(pl_module.non_linear_evaluator.parameters(), lr=1e-3)

    def get_representations(self, pl_module, x):
        """
        Override this to customize for the particular model
        Args:
            pl_module:
            x:
        """
        if len(x) == 2 and isinstance(x, list):
            x = x[0]

        representations = pl_module(x)
        representations = representations.reshape(representations.size(0), -1)
        return representations

    def to_device(self, batch, device):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        return x, y

    def on_test_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        # print(representations.shape)
        # forward pass
        mlp_preds = pl_module.non_linear_evaluator(representations)
        # print(mlp_preds.shape, y.shape)
        mlp_loss = F.cross_entropy(mlp_preds, y)

        # log metrics
        if trainer.datamodule is not None:
            acc = accuracy(mlp_preds, y, num_classes=trainer.datamodule.num_classes)
        else:
            acc = accuracy(mlp_preds, y, num_classes=self.num_classes)

        if 'test_acc' in self.output:
            self.output['test_acc'].append(acc.item())
        else:
            self.output['test_acc'] = [acc.item()]

    def on_train_epoch_end(self, trainer, pl_module):
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_dataset,
            num_replicas=4,
            rank=int(str(pl_module.device)[-1])
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            ##############################
            shuffle=False,  #
            ##############################
            num_workers=0,
            pin_memory=True,
            #############################
            sampler=train_sampler)

        for batch in train_loader:
            x, y = self.to_device(batch, pl_module.device)

            with torch.no_grad():
                representations = self.get_representations(pl_module, x)

            # print(representations.shape)
            # forward pass
            mlp_preds = pl_module.non_linear_evaluator(representations)

            # print(mlp_preds.shape, y.shape)
            mlp_loss = F.cross_entropy(mlp_preds, y)

            # update finetune weights
            mlp_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # log metrics
            if trainer.datamodule is not None:
                acc = accuracy(mlp_preds, y, num_classes=trainer.datamodule.num_classes)
            else:
                acc = accuracy(mlp_preds, y)

            metrics = {'mlp_train/loss': mlp_loss, 'mlp_train/acc': acc}
            pl_module.logger.log_metrics(metrics, step=trainer.global_step)

    def on_validation_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        # print(representations.shape)
        # forward pass
        mlp_preds = pl_module.non_linear_evaluator(representations)
        # print(mlp_preds.shape, y.shape)
        mlp_loss = F.cross_entropy(mlp_preds, y)

        # log metrics
        if trainer.datamodule is not None:
            acc = accuracy(mlp_preds, y, num_classes=trainer.datamodule.num_classes)
        else:
            acc = accuracy(mlp_preds, y, num_classes=self.num_classes)

        metrics = {'mlp_val/loss': mlp_loss, 'mlp_val/acc': acc}
        pl_module.logger.log_metrics(metrics, step=trainer.global_step)

    def on_test_end(self, trainer, pl_module):
        print(np.array(self.output['test_acc']).mean())
        if osp.exists(f"{pl_module.model_name}.p"):
            re = pickle.load(open(f"{pl_module.model_name}.p", "rb"))
            re.append(np.array(self.output['test_acc']).mean())
            pickle.dump(re, open(f"{pl_module.model_name}.p", "wb"))
        else:
            re = [np.array(self.output['test_acc']).mean()]
            pickle.dump(re, open(f"{pl_module.model_name}.p", "wb"))

    # def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
    #     x, y = self.to_device(batch, pl_module.device)
    #
    #     with torch.no_grad():
    #         representations = self.get_representations(pl_module, x)
    #
    #     # print(representations.shape)
    #     # forward pass
    #     mlp_preds = pl_module.non_linear_evaluator(representations)
    #
    #     # print(mlp_preds.shape, y.shape)
    #     mlp_loss = F.cross_entropy(mlp_preds, y)
    #
    #     # update finetune weights
    #     mlp_loss.backward()
    #     self.optimizer.step()
    #     self.optimizer.zero_grad()
    #
    #     # log metrics
    #     if trainer.datamodule is not None:
    #         acc = accuracy(mlp_preds, y, num_classes=trainer.datamodule.num_classes)
    #     else:
    #         acc = accuracy(mlp_preds, y)
    #
    #     metrics = {'mlp_train/loss': mlp_loss, 'mlp_train/acc': acc}
    #     pl_module.logger.log_metrics(metrics, step=trainer.global_step)


class NormalLMDB(Dataset):
    def __init__(self, db_path, transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))

        self.transform = transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        audio, target = pickle.loads(byteflow)
        if self.transform:
            audio = self.transform(audio)
        return audio, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


class NormalLMDBT(Dataset):
    def __init__(self, db_path, transform=None, aug_transform=None, spec_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))
        self.transform = transform
        self.aug_transform = aug_transform
        self.spec_transform = spec_transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        audio, target = pickle.loads(byteflow)
        if self.transform:
            audio = self.transform(audio)
        if self.spec_transform:
            return (self.spec_transform(audio), self.spec_transform(self.aug_transform(audio))), target
        return (self.aug_transform(audio), self.aug_transform(audio)), target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


class NormalLMDBP(Dataset):
    def __init__(self, db_path, transform=None, aug_transform=None, spec_transform=None, perc=1):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))
        self.transform = transform
        self.aug_transform = aug_transform
        self.spec_transform = spec_transform
        if perc < 1:
            random.Random(1337).shuffle(self.keys)
            self.keys = self.keys[:int(self.length * perc)]
            self.length = len(self.keys)
        else:
            self.t = []
        print(self.length)

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        audio, target = pickle.loads(byteflow)
        # print(target)
        if self.transform:
            audio = self.transform(audio)
        if self.spec_transform:
            return (self.spec_transform(audio), self.spec_transform(self.aug_transform(audio))), target
        return (self.aug_transform(audio), self.aug_transform(audio)), target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


class NormalLMDBG(Dataset):
    def __init__(self, db_path, transform=None, aug_transform=None, spec_transform=None, perc=1):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))
        self.transform = transform
        self.aug_transform = aug_transform
        self.spec_transform = spec_transform
        if perc < 1:
            random.Random(1337).shuffle(self.keys)
            self.t = self.keys[int(self.length * perc):]
        else:
            self.t = []
        print(f"dataset: {self.length}")
        print(f"ignored: {len(self.t)}")

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        audio, target = pickle.loads(byteflow)
        # print(target)
        target = int(target.item())
        if self.keys[index] in self.t:
            target = -1
        if self.transform:
            audio = self.transform(audio)
        if self.spec_transform:
            return (self.spec_transform(audio), self.spec_transform(self.aug_transform(audio))), target
        if self.aug_transform:
            return (self.aug_transform(audio), self.aug_transform(audio)), target
        return (audio, audio), target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


class NormalLMDBG_NSYNTH(Dataset):
    def __init__(self, db_path, transform=None, aug_transform=None, spec_transform=None, perc=1,
                 target='instrument_family'):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))
        with open(db_path.replace('.lmdb', '.json')) as f:
            self.jsonfile = json.load(f)

        self.transform = transform
        self.aug_transform = aug_transform
        self.target = target
        self.spec_transform = spec_transform
        if perc < 1:
            random.Random(1337).shuffle(self.keys)
            self.t = self.keys[int(self.length * perc):]
        else:
            self.t = []
        print(self.length)
        print(self.t)

    # def __getitem__(self, index):
    #     env = self.env
    #     with env.begin(write=False) as txn:
    #         byteflow = txn.get(self.keys[index])
    #     audio, target = pickle.loads(byteflow)
    #     if self.transform:
    #         audio = self.transform(audio.numpy())
    #     return torch.tensor(audio, dtype=torch.double), self.jsonfile[target]['pitch'] - 24
    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        audio, target = pickle.loads(byteflow)
        # print(target)
        target = self.jsonfile[target[0]][self.target]
        # print(target)
        # target = int(target.item())/
        if self.keys[index] in self.t:
            target = -1
        if self.transform:
            audio = self.transform(audio)
        if self.spec_transform:
            return (self.spec_transform(audio), self.spec_transform(self.aug_transform(audio))), target
        if self.aug_transform:
            return (self.aug_transform(audio), self.aug_transform(audio)), target
        return (audio, audio), target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


class NormalLMDB_ESC(Dataset):
    def __init__(self, db_path, transform=None, aug_transform=None, spec_transform=None, folds=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))
        self.files = []
        for i in range(self.length):
            with self.env.begin(write=False) as txn:
                byteflow = txn.get(self.keys[i])
            audio, target, fold = pickle.loads(byteflow)
            if fold.item() in folds:
                self.files.append(i)
        self.transform = transform
        self.aug_transform = aug_transform
        self.spec_transform = spec_transform
        print(len(self.files))

    def __getitem__(self, index):
        env = self.env
        index = self.files[index]
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        audio, target, _ = pickle.loads(byteflow)
        if self.transform:
            audio = self.transform(audio)
        if self.spec_transform:
            return (self.spec_transform(audio), self.spec_transform(self.aug_transform(audio))), target
        if self.aug_transform:
            return (self.aug_transform(audio), self.aug_transform(audio)), target
        return (audio, audio), target

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


class NormalLMDB_ESC_(Dataset):
    def __init__(self, db_path, transform=None, aug_transform=None, spec_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))
        self.transform = transform
        self.aug_transform = aug_transform
        self.spec_transform = spec_transform
        print(f"dataset: {self.length}")

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        audio = pickle.loads(byteflow)
        # print(target)
        if self.transform:
            audio = self.transform(audio)
        if self.aug_transform:
            return (self.aug_transform(audio), self.aug_transform(audio))
        return (audio, audio)

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
