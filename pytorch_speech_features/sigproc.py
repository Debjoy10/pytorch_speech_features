# This file includes routines for basic signal processing including framing and computing power spectra.
# Original Author from python_speech_features: James Lyons 2012
# Including only a subset of functions

import numpy
import math
import logging
import torch

def round_half_up(number):
    return numpy.round(number)

def rolling_window(a, window, step=1):
    # https://diegslva.github.io/2017-05-02-first-post/
    # Unfold dimension to make our rolling window
    return a.unfold(0, window, step)

def framesig(sig, frame_len, frame_step, winfunc=lambda x:numpy.ones((x,)), stride_trick=True):
    """Frame a signal into overlapping frames.
    :param sig: the audio signal to frame. -- DoubleTensor
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :param stride_trick: use stride trick to compute the rolling window and window multiplication faster
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))
    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = torch.zeros((padlen - slen,)).cuda()
    padsignal = torch.cat((sig, zeros))
    if stride_trick:
        win = torch.DoubleTensor(winfunc(frame_len)).cuda()
        frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    else:
        raise Exception("Alternative signal framing method not implemented ...")
    return frames * win

def magspec(frames, NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).
    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    """
    if frames.shape[1] > NFFT:
        logging.warn(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
            frames.shape[1], NFFT)
    complex_spec = torch.fft.rfft(frames, NFFT)
    return torch.absolute(complex_spec)

def powspec(frames, NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).
    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
    """
    return 1.0 / NFFT * torch.square(magspec(frames, NFFT))

def logpowspec(frames, NFFT, norm=1):
    """Compute the log power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).
    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :param norm: If norm=1, the log power spectrum is normalised so that the max value (across all frames) is 0.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the log power spectrum of the corresponding frame.
    """
    ps = powspec(frames, NFFT);
    ps[ps <= 1e-30] = 1e-30
    lps = 10 * torch.log10(ps)
    if norm:
        return lps - torch.max(lps)
    else:
        return lps

def preemphasis(signal, coeff=0.95):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    signalroll = torch.roll(signal, 1)
    signalroll[0] = 0
    return signal - coeff * signalroll