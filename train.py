"""Trainining script for Tacotron speech synthesis model.

usage: train.py [options]

options:
    --data-root=<dir>         Directory contains preprocessed features.
    --checkpoint-dir=<dir>    Directory where to save model checkpoints [default: checkpoints].
    --checkpoint-path=<name>  Restore model from checkpoint path if given.
    --pretrained-path=<name>  Restore model from pretrained checkpoint path if given.
    --hparams=<parmas>        Hyper parameters [default: ].
    -h, --help                Show this help message and exit
"""
from docopt import docopt

# Use text & audio modules from existing Tacotron implementation.
import sys
from os.path import dirname, join
tacotron_lib_dir = join(dirname(__file__), "lib", "tacotron")
sys.path.append(tacotron_lib_dir)
from text import text_to_sequence, symbols, prosody_symbols, phone_to_sequence
from text.prosody_symbols import phones, syllables
print(len(phones))
from util import audio
from util.plot import plot_alignment
from tqdm import tqdm, trange

# The tacotron model
from tacotron_pytorch import Tacotron

import torch
from torch.utils import data as data_utils
from torch.autograd import Variable
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from os.path import join, expanduser

import librosa.display
from matplotlib import pyplot as plt
import sys
import os
import tensorboard_logger
from tensorboard_logger import log_value
from hparams import hparams, hparams_debug_string

# Default DATA_ROOT
DATA_ROOT = join(expanduser("~"), "tacotron", "training")

fs = hparams.sample_rate

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False


def _pad(seq, max_len):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=0)


def _pad_2d(x, max_len):
    x = np.pad(x, [(0, max_len - len(x)), (0, 0)],
               mode="constant", constant_values=0)
    return x


class TextDataSource(FileDataSource):
    def __init__(self, col):
        self.col = col
        self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]

    def collect_files(self):
        meta = join(DATA_ROOT, "train.txt")
        with open(meta, "rb") as f:
            lines = f.readlines()
        lines = list(map(lambda l: l.decode("utf-8").split("|")[self.col], lines))
        return lines

    def collect_features(self, text):
        return np.asarray(text_to_sequence(text, self._cleaner_names),
                          dtype=np.int32)

class PhoneDataSource(FileDataSource):
    def __init__(self, col):
        self.col = col
        self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]


    def collect_files(self):
        meta = join(DATA_ROOT, "train.txt")
        with open(meta, "rb") as f:
            lines = f.readlines()
        lines = list(map(lambda l: l.decode("utf-8").split("|")[self.col], lines))
        return lines

    def collect_features(self, text):
        return np.asarray(phone_to_sequence(text, self._cleaner_names),dtype=np.int32)


class _NPYDataSource(FileDataSource):
    def __init__(self, col):
        self.col = col

    def collect_files(self):
        meta = join(DATA_ROOT, "train.txt")
        with open(meta, "rb") as f:
            lines = f.readlines()
        lines = list(map(lambda l: l.decode("utf-8").split("|")[self.col], lines))
        paths = list(map(lambda f: join(DATA_ROOT, f), lines))
        return paths

    def collect_features(self, path):
        return np.load(path)


class MelSpecDataSource(_NPYDataSource):
    def __init__(self):
        super(MelSpecDataSource, self).__init__(1)

class F0DataSource(_NPYDataSource):
    def __init__(self,col_num):
        super(F0DataSource, self).__init__(col_num)

class EmbeddingDataSource(_NPYDataSource):
    def __init__(self):
        super(EmbeddingDataSource, self).__init__(4)


class LinearSpecDataSource(_NPYDataSource):
    def __init__(self):
        super(LinearSpecDataSource, self).__init__(0)


class PyTorchDataset(object):
    def __init__(self, X, Mel, Y, Phones, F0):
        self.X = X
        self.Mel = Mel
        self.Y = Y
        self.Phones = Phones
        self.F0 = F0

    def __getitem__(self, idx):
        return self.X[idx], self.Mel[idx], self.Y[idx], self.Phones[idx], self.F0[idx]

    def __len__(self):
        return len(self.X)


def collate_fn(batch):
    """Create batch-one hot phone embedding with framewise F0"""
    r = hparams.outputs_per_step
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = np.max(input_lengths)
    # Add single zeros frame at least, so plus 1
    max_target_len = np.max([len(x[1]) for x in batch]) + 1
    max_f0_len = np.max([len(x[4]) for x in batch]) + 1 
    phone_lengths = [len(x[3]) for x in batch]
    max_phone_len = np.max(phone_lengths)
    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0

    a = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.int)
    x_batch = torch.LongTensor(a)

    input_lengths = torch.LongTensor(input_lengths)
      
    b = np.array([_pad_2d(x[1], max_target_len) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    c = np.array([_pad_2d(x[2], max_target_len) for x in batch],
                 dtype=np.float32)
    y_batch = torch.FloatTensor(c)
    d = np.array([_pad(x[3], max_phone_len) for x in batch], dtype=np.int)
    phone_batch = torch.LongTensor(d)

    phone_lengths = torch.LongTensor(phone_lengths)
    e = np.array([_pad_2d(x[4], max_f0_len) for x in batch], dtype=np.float32)
    f0_batch = torch.FloatTensor(e)

    return x_batch, input_lengths, mel_batch, y_batch, phone_batch, phone_lengths, f0_batch


def save_alignment(path, attn):
    plot_alignment(attn.T, path, info="tacotron, step={}".format(global_step))


def save_spectrogram(path, linear_output):
    spectrogram = audio._denormalize(linear_output)
    plt.figure(figsize=(16, 10))
    plt.imshow(spectrogram.T, aspect="auto", origin="lower")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, format="png")
    plt.close()


def _learning_rate_decay(init_lr, global_step):
    warmup_steps = 4000.0
    step = global_step + 1.
    lr = init_lr * warmup_steps**0.5 * np.minimum(
        step * warmup_steps**-1.5, step**-0.5)
    return lr


def save_states(global_step, mel_outputs, linear_outputs, attn, y,
                input_lengths, phone_lengths=None, checkpoint_dir=None):
    print("Save intermediate states at step {}".format(global_step))

    # idx = np.random.randint(0, len(input_lengths))
    idx = min(1, len(input_lengths) - 1)
    input_length = input_lengths[idx]

    # Alignment
    path = join(checkpoint_dir, "step{}_alignment.png".format(
        global_step))
    # alignment = attn[idx].cpu().data.numpy()[:, :input_length]
    alignment = attn[idx].cpu().data.numpy()
    save_alignment(path, alignment)

    # Predicted spectrogram
    path = join(checkpoint_dir, "step{}_predicted_spectrogram.png".format(
        global_step))
    linear_output = linear_outputs[idx].cpu().data.numpy()
    save_spectrogram(path, linear_output)

    # Predicted audio signal
    signal = audio.inv_spectrogram(linear_output.T)
    path = join(checkpoint_dir, "step{}_predicted.wav".format(
        global_step))
    audio.save_wav(signal, path)

    # Target spectrogram
    path = join(checkpoint_dir, "step{}_target_spectrogram.png".format(
        global_step))
    linear_output = y[idx].cpu().data.numpy()
    save_spectrogram(path, linear_output)


def train(model, data_loader, optimizer,
          init_lr=0.002,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None,
          clip_thresh=1.0):
    model.train()
    if use_cuda:
        model = model.cuda()
    linear_dim = model.linear_dim

    criterion = nn.L1Loss()

    global global_step, global_epoch
    while global_epoch < nepochs:
        running_loss = 0.
        running_f0_loss = 0.
        running_acoustic_loss = 0.
        for step, (x, input_lengths, mel, y, phone, phone_lengths, f0) in tqdm(enumerate(data_loader)):
            # Decay learning rate
            #current_lr = _learning_rate_decay(init_lr, global_step)
            for param_group in optimizer.param_groups:
                print(" lr: ", param_group['lr']),
            print
            optimizer.zero_grad()

            # Sort by length
            sorted_lengths, indices = torch.sort(
                input_lengths.view(-1), dim=0, descending=True)
            sorted_lengths = sorted_lengths.long().numpy()
          
            x, mel, y, phone, phone_lengths, f0 = x[indices], mel[indices], y[indices], phone[indices], phone_lengths[indices], f0[indices]
            # Feed data
            x, mel, y, phone, f0 = Variable(x), Variable(mel), Variable(y), Variable(phone), Variable(f0)
            if use_cuda:
                x, mel, y, phone, f0 = x.cuda(), mel.cuda(), y.cuda(), phone.cuda(), f0.cuda()
            mel_outputs, linear_outputs, attn, prosody_outputs = model(
                inputs=x, phone_inputs=phone, targets=mel, input_lengths=sorted_lengths, phone_lengths=phone_lengths, f0=f0)

            #mel_outputs, linear_outputs, attn = model(
            #    x, mel, input_lengths=sorted_lengths, phone_lengths=phone_lengths)

            # Loss
            mel_loss = criterion(mel_outputs, mel)
            n_priority_freq = int(3000 / (fs * 0.5) * linear_dim)
            linear_loss = 0.5 * criterion(linear_outputs, y) \
                + 0.5 * criterion(linear_outputs[:, :, :n_priority_freq],
                                  y[:, :, :n_priority_freq])
            F0_loss = criterion(prosody_outputs, f0) 
            Acoustic_loss = mel_loss + linear_loss 
            loss = Acoustic_loss + F0_loss

            if global_step > 0 and global_step % checkpoint_interval == 0:
                save_states(
                    global_step, mel_outputs, linear_outputs, attn, y,
                    sorted_lengths, phone_lengths, checkpoint_dir)
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            # Update
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm(
                model.parameters(), clip_thresh)
            optimizer.step()

            # Logs
            log_value("loss", float(loss.data[0]), global_step)
            log_value("mel loss", float(mel_loss.data[0]), global_step)
            log_value("F0 loss", float(F0_loss.data[0]), global_step)
            log_value("linear loss", float(linear_loss.data[0]), global_step)
            log_value("gradient norm", grad_norm, global_step)
            #log_value("learning rate", current_lr, global_step)

            global_step += 1
            running_loss += loss.data[0]
            running_f0_loss += F0_loss.data[0]
            running_acoustic_loss += Acoustic_loss.data[0]

        averaged_loss = running_loss / (len(data_loader))
        log_value("loss (per epoch)", averaged_loss, global_epoch)
        print("Loss: {}".format(running_loss / (len(data_loader))))
        print("Acoustic Loss: {}".format(running_acoustic_loss / (len(data_loader))))
        print("F0 Loss: {}".format(running_f0_loss / (len(data_loader))))

        global_epoch += 1


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{}.pth".format(global_step))
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def _transfer(pretrained_dict, model_dict):
    pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    return model_dict, pretrained_dict


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    checkpoint_path = args["--checkpoint-path"]
    pretrained_path = args["--pretrained-path"]
    data_root = args["--data-root"]
    if data_root:
        DATA_ROOT = data_root

    # Override hyper parameters
    hparams.parse(args["--hparams"])

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Input dataset definitions
    X = FileSourceDataset(TextDataSource(3))
    Mel = FileSourceDataset(MelSpecDataSource())
    Y = FileSourceDataset(LinearSpecDataSource())
    Phones = FileSourceDataset(PhoneDataSource(4))
    if hparams.f0_type == "framewise":
        col_num = 5; hparams.f0_dim =1
    else:
        col_num = 6; hparams.f0_dim=10

    F0 = FileSourceDataset(F0DataSource(col_num))
    # Dataset and Dataloader setup
    dataset = PyTorchDataset(X, Mel, Y, Phones, F0)
    data_loader = data_utils.DataLoader(
        dataset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers, shuffle=True,
        collate_fn=collate_fn, pin_memory=hparams.pin_memory)

    # Model
    model = Tacotron(n_vocab=len(symbols), n_phone=len(phones),
                     embedding_dim=256,
                     mel_dim=hparams.num_mels,
                     linear_dim=hparams.num_freq,
                     f0_dim=hparams.f0_dim,
                     r=hparams.outputs_per_step,
                     padding_idx=hparams.padding_idx,
                     use_memory_mask=hparams.use_memory_mask,
                     )
    #optimizer = optim.Adam(model.parameters(),
    #                       lr=hparams.initial_learning_rate, betas=(
    #                           hparams.adam_beta1, hparams.adam_beta2),
    #                       weight_decay=hparams.weight_decay)
   

    # Load pre-trained model
    if pretrained_path:
        print("Load checkpoint from: {}".format(pretrained_path))
        pretrained_model = torch.load(pretrained_path)
        model_dict, pretrained_dict = _transfer(pretrained_model["state_dict"], model.state_dict())
        model.load_state_dict(model_dict)
        current_lr = _learning_rate_decay(hparams.initial_learning_rate, pretrained_model["global_step"])
        #ignore_list=[model.treeencoder.parameters(), model.decoder.project_prosody_emb_to_decoder_in.parameters(), model.rnndecoder.parameters()]
        ignored_params=[]
        for prams in [model.treeencoder.parameters(), model.decoder.project_prosody_emb_to_decoder_in.parameters(), model.rnndecoder.parameters()]:
            ignored_params +=list(prams)
        ignored_params_ids = []
        for param_list in [model.treeencoder.parameters(), model.decoder.project_prosody_emb_to_decoder_in.parameters(), model.rnndecoder.parameters()]:
            ignored_params_ids.extend(list(map(id, param_list)))
        print(len(ignored_params_ids))
        base_params = filter(lambda p: id(p) not in ignored_params_ids, model.parameters())
        if hparams.frozen:
            print("Freezing weights for pretrained model")    
            for params in base_params:
                params.requires_grad = False
            optimizer = optim.Adam([{'params':ignored_params, 'lr': hparams.initial_learning_rate}], weight_decay=1.0)
  
        else:
            optimizer = optim.Adam([{'params':base_params, 'lr':current_lr},  {'params':ignored_params, 'lr': hparams.initial_learning_rate}], weight_decay=1.0)
 
       # optimizer_dict, pretrain_dict = _transfer(pretrained_model["optimizer"], optimizer.state_dict())
       # optimizer.load_state_dict(optimizer_dict)
          
                    
   


    # Load checkpoint
    if checkpoint_path:
        print("Load checkpoint from: {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        try:
            global_step = checkpoint["global_step"]
            global_epoch = checkpoint["global_epoch"]
        except:
            # TODO
            pass

    # Setup tensorboard logger
    tensorboard_logger.configure("log/run-test")

    print(hparams_debug_string())

    # Train!
    try:
        train(model, data_loader, optimizer,
              init_lr=hparams.initial_learning_rate,
              checkpoint_dir=checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=hparams.nepochs,
              clip_thresh=hparams.clip_thresh)
    except KeyboardInterrupt:
        save_checkpoint(
            model, optimizer, global_step, checkpoint_dir, global_epoch)

    print("Finished")
    sys.exit(0)
