# -*- coding: utf-8 -*-
"""Modal decomposition and reconstruction utilities."""

from numpy import *

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None


class overlaps():

    def __init__(self):
        pass

    @classmethod
    def Modal_decomposition(cls, Modes2D, frames):
        N_ele = len(frames.shape)
        N_frames = 0

        if N_ele == 2:
            N_px = frames.shape[0] * frames.shape[1]
            N_frames = 1
        else:
            N_px = frames.shape[1] * frames.shape[2]
            N_frames = frames.shape[0]

        frames = reshape(frames, (N_frames, N_px))
        coef = matmul(frames, Modes2D)
        return coef

    @classmethod
    def Modal_reconstruction(cls, Modes2D, coefs):
        N_ele = len(coefs.shape)
        N_frames = 0
        N_px = int(sqrt(Modes2D.shape[0]))

        if N_ele == 1:
            N_frames = 1
        else:
            N_frames = coefs.shape[0]
        rec = matmul(coefs, transpose(Modes2D))
        rec = reshape(rec, (N_frames, N_px, N_px))
        return rec

    @classmethod
    def Modal_decomposition_gpu(cls, Modes2D, frames):
        N_ele = len(frames.shape)
        N_frames = 0

        if N_ele == 2:
            N_px = frames.shape[0] * frames.shape[1]
            N_frames = 1
        else:
            N_px = frames.shape[1] * frames.shape[2]
            N_frames = frames.shape[0]

        frames = cp.reshape(frames, (N_frames, N_px))
        coef = cp.matmul(frames, Modes2D)
        return coef

    @classmethod
    def Modal_reconstruction_gpu(cls, Modes2D, coefs):
        N_ele = len(coefs.shape)
        N_frames = 0
        N_px = int(cp.sqrt(Modes2D.shape[0]))

        if N_ele == 1:
            N_frames = 1
        else:
            N_frames = coefs.shape[0]
        rec = cp.matmul(coefs, cp.transpose(Modes2D))
        rec = cp.reshape(rec, (N_frames, N_px, N_px))
        return rec
