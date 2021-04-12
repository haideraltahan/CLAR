import os
# import torchsummary
from functools import partial

import fire
import torchaudio
from pl_bolts.models.self_supervised.resnets import resnet18
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer import Trainer
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm

from Spectrograms import *
from models import *
from utils import *


class Main(LightningModule):
    def __init__(self,
                 batch_size,
                 sampling_rate,
                 nsynth_class,
                 train_transform,
                 aug_transform,
                 spec_transform,
                 warmup_epochs=10,
                 optimizer='lars',
                 perc=1,
                 val_batch_size=128,
                 learning_rate=1.0,
                 lars_momentum=0.9,
                 lars_eta=0.001,
                 lr_sched_step=30,
                 lr_sched_gamma=0.5,
                 weight_decay=1e-4,
                 opt_weight_decay=1e-6,
                 loss_temperature=0.5,
                 n_classes=10,
                 data_dir="../sc",
                 dataset="sc09",
                 model="resnet_2d",
                 num_workers=32):
        """
        Args:
            batch_size: the batch size
            num_samples: num samples in the dataset
            warmup_epochs: epochs to warmup the lr for
            lr: the optimizer learning rate
            opt_weight_decay: the optimizer weight decay
            loss_temperature: the loss temperature
        """
        super().__init__()
        self.save_hyperparameters()
        self.test_transform = train_transform
        self.train_transform = train_transform
        self.aug_transform = aug_transform
        self.nsynth_class = nsynth_class
        self.dataset = dataset
        self.data_dir = data_dir
        self.sampling_rate = sampling_rate
        self.n_classes = n_classes
        self.val_batch_size = val_batch_size
        self.model_name = model
        self.perc = perc
        self.init_encoder()
        self.spec_transform = spec_transform
        self.nt_xent_loss = nt_xent_loss
        # h -> || -> z
        self.projection = Projection(
            512,
            512,
        )
        self.adaptive = nn.AdaptiveAvgPool1d(1) if "1d" in self.model_name else nn.AdaptiveAvgPool2d((1, 1))

    def init_encoder(self):
        if "2d" in self.model_name:
            self.model = resnet18(return_all_feature_maps=False)
        else:
            self.model = resnet18_1D()
            # torchsummary.summary(self.model.cuda(), (1, self.sampling_rate))

    # This function is called on every GPU
    def setup(self, stage):
        global_batch_size = self.trainer.world_size * self.hparams.batch_size
        self.load_datasets()
        self.train_iters_per_epoch = len(self.train_dataset) // global_batch_size
        if "2d" in self.model_name:
            self.stft = STFT(n_fft=2048,
                             hop_length=128,
                             sr=16000,
                             freq_bins=128,
                             freq_scale='log',
                             fmin=40,
                             fmax=8000,
                             verbose=False)
            self.mel_stft = MelSpectrogram(n_fft=2048,
                                           hop_length=128,
                                           n_mels=128,
                                           sr=16000,
                                           fmin=40,
                                           fmax=8000,
                                           verbose=False)

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {'params': params, 'weight_decay': weight_decay},
            {'params': excluded_params, 'weight_decay': 0.}
        ]

    def get_features(self, x):
        return torch.stack((
            torchaudio.functional.amplitude_to_DB(self.stft(x, 'Magnitude').abs(), 20, 1e-05, 1),
            self.stft(x, 'Phase'),
            torchaudio.functional.amplitude_to_DB(self.mel_stft(x).abs(), 10, 1e-05, 1),
        ), dim=1).detach().squeeze()

    def load_datasets(self):
        if 'esc' in self.dataset:
            data_path = os.path.join(self.data_dir, self.dataset + ".lmdb")
            self.train_dataset = NormalLMDB_ESC(data_path, transform=self.train_transform,
                                                aug_transform=self.aug_transform,
                                                spec_transform=self.spec_transform,
                                                folds=[1, 2, 3, 4])
            self.val_dataset = NormalLMDB_ESC(data_path, transform=self.test_transform,
                                              aug_transform=self.aug_transform,
                                              spec_transform=self.spec_transform,
                                              folds=[5])
            self.test_dataset = NormalLMDB_ESC(data_path, transform=self.test_transform,
                                               aug_transform=self.aug_transform,
                                               spec_transform=self.spec_transform,
                                               folds=[5])
        elif 'nsynth' in self.dataset:
            data_path = os.path.join(self.data_dir, "train_" + self.dataset + ".lmdb")
            self.train_dataset = NormalLMDBG_NSYNTH(data_path, transform=self.train_transform,
                                                    aug_transform=self.aug_transform,
                                                    spec_transform=self.spec_transform,
                                                    target=self.nsynth_class,
                                                    perc=self.perc)
            data_path = os.path.join(self.data_dir, "test_" + self.dataset + ".lmdb")
            self.val_dataset = NormalLMDBG_NSYNTH(data_path, transform=self.test_transform,
                                                  aug_transform=self.aug_transform,
                                                  spec_transform=self.spec_transform,
                                                  target=self.nsynth_class)
            self.test_dataset = NormalLMDBG_NSYNTH(data_path, transform=self.test_transform,
                                                   aug_transform=self.aug_transform,
                                                   spec_transform=self.spec_transform,
                                                   target=self.nsynth_class)
        else:
            data_path = os.path.join(self.data_dir, "train_" + self.dataset + ".lmdb")
            self.train_dataset = NormalLMDBP(data_path, transform=self.train_transform,
                                             aug_transform=self.aug_transform,
                                             spec_transform=self.spec_transform,
                                             perc=self.perc
                                             )
            data_path = os.path.join(self.data_dir, "test_" + self.dataset + ".lmdb")
            self.val_dataset = NormalLMDBG(data_path, transform=self.test_transform, aug_transform=self.aug_transform,
                                           spec_transform=self.spec_transform)
            data_path = os.path.join(self.data_dir, "test_" + self.dataset + ".lmdb")
            self.test_dataset = NormalLMDBG(data_path, transform=self.test_transform, aug_transform=self.aug_transform,
                                            spec_transform=self.spec_transform)

    def configure_optimizers(self):
        # TRICK 1 (Use lars + filter weights)
        # exclude certain parameters
        parameters = self.exclude_from_wt_decay(
            self.named_parameters(),
            weight_decay=self.hparams.opt_weight_decay
        )

        optimizer = LARSWrapper(Adam(parameters, lr=self.hparams.learning_rate))

        # Trick 2 (after each step)
        self.hparams.warmup_epochs = self.hparams.warmup_epochs * self.train_iters_per_epoch
        max_epochs = self.trainer.max_epochs * self.train_iters_per_epoch

        linear_warmup_cosine_decay = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=max_epochs,
            warmup_start_lr=0,
            eta_min=0
        )

        scheduler = {
            'scheduler': linear_warmup_cosine_decay,
            'interval': 'step',
            'frequency': 1
        }

        if self.perc == 0.01:
            return [optimizer], []
        else:
            return [optimizer], [scheduler]

    def forward(self, x):
        if isinstance(x, list):
            x = x[0]

        if "2d" in self.model_name:
            x = self.model(self.get_features(x))
        else:
            x = self.model(x.view(x.shape[0], 1, -1))

        if isinstance(x, list):
            x = x[-1]
        x = self.adaptive(x)
        return x

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)

        result = pl.TrainResult(minimize=loss)
        result.log_dict({
            'loss/train': loss
        })
        return result

    # def validation_step(self, batch, batch_idx):
    #     loss = self.shared_step(batch, batch_idx)
    #
    #     result = pl.EvalResult(checkpoint_on=loss)
    #     result.log_dict({
    #         'loss/val': loss
    #     })
    #     return result

    def test_step(self, batch, batch_idx):
        loss = 0
        # self.trainer.callbacks
        # print(loss)

    def shared_step(self, batch, batch_idx):
        (aud1, aud2), y = batch

        # ENCODE
        # encode -> representations
        # (b, 3, 32, 32) -> (b, 2048, 2, 2)
        h1 = self.forward(aud1)
        h2 = self.forward(aud2)

        # print(h1.shape)

        # PROJECT
        # img -> E -> h -> || -> z
        # (b, 2048, 2, 2) -> (b, 128)
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.nt_xent_loss(z1, z2, self.hparams.loss_temperature)

        return loss

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           shuffle=True,
                                           batch_size=self.hparams.batch_size,
                                           num_workers=self.hparams.num_workers,
                                           pin_memory=True)

    # def val_dataloader(self):
    #     return torch.utils.data.DataLoader(self.val_dataset,
    #                                        batch_size=self.val_batch_size,
    #                                        num_workers=self.hparams.num_workers,
    #                                        pin_memory=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.val_batch_size,
                                           num_workers=self.hparams.num_workers,
                                           pin_memory=True)


MODELS_FOLDER = './models'
LOGS_FOLDER = './logs'


def main(model_name='usp_1d', max_epochs=1020, data_dir='./data', dataset='sc09', ps=False, wn=False, mx=False, perc=1,
         ts=False, fd=False, tts=False, tm=False, train=True, order=True, model_num=None):
    model_name = model_name + '_' + str(int(perc * 100))
    dataset_f = dataset
    nsynth_class = None
    if dataset == 'sc09':
        sample_rate = 16000
        n_classes = 10
        length = 1
        batch_size = 256
        train_transform = transforms.Compose([
            torchaudio.transforms.Resample(orig_freq=16000, new_freq=sample_rate),
            pad,
        ])
    elif dataset == 'sc':
        sample_rate = 16000
        n_classes = 35
        batch_size = 128
        length = 1
        train_transform = transforms.Compose([
            torchaudio.transforms.Resample(orig_freq=16000, new_freq=sample_rate),
            partial(pad, length=length),
        ])
    elif dataset == 'nsynth11':
        sample_rate = 16000
        n_classes = 11
        batch_size = 32
        max_epochs = 120
        dataset = 'nsynth'
        nsynth_class = 'instrument_family'
        length = 4
        train_transform = transforms.Compose([
            torchaudio.transforms.Resample(orig_freq=16000, new_freq=sample_rate),
            partial(pad, length=length),
        ])
    elif dataset == 'nsynth128':
        sample_rate = 16000
        n_classes = 128
        batch_size = 16
        max_epochs = 120
        dataset = 'nsynth'
        nsynth_class = 'pitch'
        length = 4
        train_transform = transforms.Compose([
            torchaudio.transforms.Resample(orig_freq=16000, new_freq=sample_rate),
            partial(pad, length=length),
        ])
    elif dataset == 'esc50':
        sample_rate = 16000
        max_epochs = 2000
        n_classes = 50
        batch_size = 32
        length = 5
        train_transform = transforms.Compose([
            torchaudio.transforms.Resample(orig_freq=44100, new_freq=sample_rate),
            partial(pad, length=length),
        ])
    elif dataset == 'esc10':
        sample_rate = 16000
        n_classes = 10
        max_epochs = 2000
        batch_size = 32
        length = 5
        train_transform = transforms.Compose([
            torchaudio.transforms.Resample(orig_freq=44100, new_freq=sample_rate),
            partial(pad, length=length),
        ])

    # model_name = model_name + '_' + dataset
    spec_transform = None
    aug_transform = []
    if order:
        if fd:
            aug_transform.append(add_fade)
            model_name = model_name + '_fd'
        if tm:
            aug_transform.append(time_masking)
            model_name = model_name + '_tm'
        if tts:
            aug_transform.append(partial(time_stret, length=length))
            model_name = model_name + '_tts'
        if ps:
            aug_transform.append(pitch_shift)
            model_name = model_name + '_ps'
        if ts:
            aug_transform.append(time_shift)
            model_name = model_name + '_ts'
        if wn:
            aug_transform.append(add_white_noise)
            model_name = model_name + '_wn'
        if mx:
            m_x = Mixed_Noise(data_dir, sample_rate)
            aug_transform.append(m_x)
            model_name = model_name + '_mx'
    else:
        if mx:
            m_x = Mixed_Noise(data_dir, sample_rate)
            aug_transform.append(m_x)
            model_name = model_name + '_mx'
        if wn:
            aug_transform.append(add_white_noise)
            model_name = model_name + '_wn'
        if ts:
            aug_transform.append(time_shift)
            model_name = model_name + '_ts'
        if ps:
            aug_transform.append(pitch_shift)
            model_name = model_name + '_ps'
        if fd:
            aug_transform.append(add_fade)
            model_name = model_name + '_fd'
        if tts:
            aug_transform.append(partial(time_stret, length=length))
            model_name = model_name + '_tts'
        if tm:
            aug_transform.append(time_masking)
            model_name = model_name + '_tm'

    aug_transform = transforms.Compose(aug_transform)
    print(f"Model: {model_name}")

    net = Main(batch_size=batch_size,
               sampling_rate=sample_rate,
               data_dir=data_dir,
               dataset=dataset,
               perc=perc,
               nsynth_class=nsynth_class,
               n_classes=n_classes,
               train_transform=train_transform,
               aug_transform=aug_transform,
               spec_transform=spec_transform,
               model=model_name)

    model_path = os.path.join(MODELS_FOLDER, model_name, dataset_f)
    os.makedirs(model_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        filepath=model_path,
        save_last=True,
        mode='min',
        period=10,
        save_top_k=20000000,
    )
    if model_num is not None:
        checkpoint = os.path.join(model_path, get_last(os.listdir(model_path), model_num))
    elif os.path.exists(model_path) and len(os.listdir(model_path)) > 0:
        checkpoint = os.path.join(model_path, get_last(os.listdir(model_path)))
    else:
        checkpoint = None

    logger = TensorBoardLogger(
        save_dir=LOGS_FOLDER,
        version=dataset_f,
        name=model_name
    )

    # finetune in real-time
    print(f"Loading model: {checkpoint}")

    def to_device(batch, device):
        (x1, x2), y = batch
        x1 = x1.to(device)
        y = y.to(device).squeeze()
        return x1, y

    online_eval = SSLOnlineEvaluator(hidden_dim=512,
                                     z_dim=512,
                                     num_classes=n_classes,
                                     train_transform=train_transform,
                                     data_dir=data_dir,
                                     dataset=dataset,
                                     batch_size=batch_size,
                                     nsynth_class=nsynth_class
                                     )
    online_eval.to_device = to_device

    trainer = Trainer(resume_from_checkpoint=checkpoint,
                      distributed_backend='ddp',
                      max_epochs=max_epochs,
                      sync_batchnorm=True,
                      checkpoint_callback=checkpoint_callback,
                      logger=logger,
                      gpus=-1 if train else 1,
                      log_save_interval=25,
                      callbacks=[online_eval]
                      )
    if train:
        trainer.fit(net)
    else:
        trainer.test(net)


if __name__ == '__main__':
    fire.Fire(main)
