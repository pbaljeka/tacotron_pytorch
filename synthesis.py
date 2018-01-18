# coding: utf-8
"""
Synthesis waveform from trained model.

usage: tts.py [options] <checkpoint> <text_list_file> <dst_dir>

options:
    --file-name-suffix=<s>   File name suffix [default: ].
    --max-decoder-steps=<N>  Max decoder steps [default: 500].
    --f0-type=<s> F0 Type [default: dctf0].
    -h, --help               Show help message.
"""
from docopt import docopt

# Use text & audio modules from existing Tacotron implementation.
import sys
import os
from os.path import dirname, join
tacotron_lib_dir = join(dirname(__file__), "lib", "tacotron")
sys.path.append(tacotron_lib_dir)
from text import text_to_sequence, symbols, prosody_symbols, phone_to_sequence
from text.prosody_symbols import phones, syllables
from util import audio
from util.plot import plot_alignment

import torch
from torch.autograd import Variable
import numpy as np
import nltk

from tacotron_pytorch import Tacotron
from hparams import hparams

from tqdm import tqdm

use_cuda = torch.cuda.is_available()


def tts(model, text, phone_seq):
    """Convert text to speech waveform given a Tacotron model.
    """
    if use_cuda:
        model = model.cuda()
    # TODO: Turning off dropout of decoder's prenet causes serious performance
    # regression, not sure why.
    #model.decoder.eval()
    model.encoder.eval()
    model.treeencoder.eval()
    model.postnet.eval()
    model.rnndecoder.eval()

    sequence = np.array(text_to_sequence(text, [hparams.cleaners]))
    sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0)
    phone_sequence = np.array(phone_to_sequence(phone_seq, [hparams.cleaners]))
    phone_sequence = Variable(torch.from_numpy(phone_sequence)).unsqueeze(0)
    phone_lengths = torch.LongTensor(len(phone_sequence))
    if use_cuda:
        sequence = sequence.cuda()
        phone_sequence = phone_sequence.cuda()

    # Greedy decoding
    mel_outputs, linear_outputs, alignments, prosody_outputs = model(sequence, phone_sequence, phone_lengths=phone_lengths)

    linear_output = linear_outputs[0].cpu().data.numpy()
    spectrogram = audio._denormalize(linear_output)
    alignment = alignments[0].cpu().data.numpy()

    # Predicted audio signal
    waveform = audio.inv_spectrogram(linear_output.T)

    return waveform, alignment, spectrogram


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_path = args["<checkpoint>"]
    text_list_file_path = args["<text_list_file>"]
    dst_dir = args["<dst_dir>"]
    max_decoder_steps = int(args["--max-decoder-steps"])
    file_name_suffix = args["--file-name-suffix"]
    f0_type = "dctf0"
    if f0_type == "dctf0":
        f0_dim = 10
    elif f0_type == "framewisef0":
        f0_dim = 1
    else:
        print("Wrong f0 type")
    print(f0_type, f0_dim)
    model = Tacotron(n_vocab=len(symbols), n_phone=len(phones),
                     embedding_dim=256,
                     mel_dim=hparams.num_mels,
                     f0_dim=f0_dim,
                     linear_dim=hparams.num_freq,
                     r=hparams.outputs_per_step,
                     padding_idx=hparams.padding_idx,
                     use_memory_mask=hparams.use_memory_mask,
                     )
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.decoder.max_decoder_steps = max_decoder_steps

    os.makedirs(dst_dir, exist_ok=True)

    with open(text_list_file_path, "rb") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            line_items = line.decode("utf-8")[:-1].split('|')
            text = line_items[3]
            phone_seq = line_items[4]
            words = nltk.word_tokenize(text)
            print("{}: {} ({} chars, {} words)".format(idx, text, len(text), len(words)))
            waveform, alignment, _ = tts(model, text, phone_seq)
            dst_wav_path = join(dst_dir, "{}{}.wav".format(idx, file_name_suffix))
            dst_alignment_path = join(dst_dir, "{}_alignment.png".format(idx))
            plot_alignment(alignment.T, dst_alignment_path,
                           info="tacotron, {}".format(checkpoint_path))
            audio.save_wav(waveform, dst_wav_path)

    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
    sys.exit(0)
