#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional

import math
from multiprocessing import Lock
from .SharedMemory import SharedMemory

import numpy as np
from numpy import typing as npt


def dtypes_are_equal(dtype1: np.dtype, dtype2: np.dtype) -> bool:
    return np.issubdtype(dtype2, dtype1) and np.issubdtype(dtype1, dtype2)


class GrowingSharedArrayReaderView:
    basename_X: str = 'X'
    basename_y: str = 'y'

    def __init__(self, lock: Lock):
        self.lock = lock
        self.shm_id: Optional[int] = None
        self.shm_X: Optional[SharedMemory] = None
        self.shm_y: Optional[SharedMemory] = None
        self.size: Optional[int] = None

    def open(self, shm_id: int, size: int):
        if shm_id != self.shm_id:
            self.close()
            self.shm_X = SharedMemory(f'{self.basename_X}_{shm_id}', track=False)
            self.shm_y = SharedMemory(f'{self.basename_y}_{shm_id}', track=False)
            self.shm_id = shm_id
        self.size = size

    def close_impl(self, unlink=False):
        if self.shm_X is not None:
            self.shm_X.close()
            if unlink:
                self.shm_X.unlink()
            del self.shm_X
            self.shm_X = None
        if self.shm_y is not None:
            self.shm_y.close()
            if unlink:
                self.shm_y.unlink()
            del self.shm_y
            self.shm_y = None
        self.shm_id = None
        self.size = None

    def close(self):
        self.close_impl()

    def __del__(self):
        self.close()

    @property
    def capacity(self) -> int:
        if self.shm_y is None:
            return 0
        assert self.shm_y.size % np.dtype(np.float64).itemsize == 0
        return self.shm_y.size // np.dtype(np.float64).itemsize

    @property
    def row_size(self) -> Optional[int]:
        if self.shm_X is None:
            return None
        if self.shm_X.size == 0:
            return None
        assert self.shm_X.size % self.shm_y.size == 0
        return self.shm_X.size // self.shm_y.size

    @property
    def X(self):
        X = np.ndarray(shape=(self.capacity, self.row_size), dtype=np.float64, buffer=self.shm_X.buf)
        return X[:self.size]

    @property
    def y(self):
        y = np.ndarray(shape=(self.capacity,), dtype=np.float64, buffer=self.shm_y.buf)
        return y[:self.size]

    def get_data(self, shm_id: int, size: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        with self.lock:
            self.open(shm_id, size)
            X, y = np.array(self.X), np.array(self.y)  # make copies and release lock to minimize critical section

        return X, y


class GrowingSharedArray(GrowingSharedArrayReaderView):
    def __init__(self):
        self.growth_rate = 1.5
        super().__init__(lock=Lock())

    def close(self):
        self.close_impl(unlink=True)

    def __del__(self):
        self.close()

    def set_data(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
        assert len(X) == len(y)
        assert X.ndim == 2
        assert y.ndim == 1
        assert dtypes_are_equal(X.dtype, np.float64)
        assert dtypes_are_equal(y.dtype, np.float64)
        assert X.dtype.itemsize == 8
        assert y.dtype.itemsize == 8

        size = len(y)
        grow = size > self.capacity
        if grow:
            if self.capacity:
                n_growth = math.ceil(math.log(size / self.capacity, self.growth_rate))
                capacity = int(math.ceil(self.capacity * self.growth_rate ** n_growth))
                shm_id = self.shm_id + 1
            else:
                assert self.shm_X is None
                assert self.shm_y is None
                capacity = size
                shm_id = 0

            row_size = X.shape[1]
            if self.row_size is not None:
                assert row_size == self.row_size
            shm_X = SharedMemory(f'{self.basename_X}_{shm_id}', create=True,
                                 size=capacity * row_size * X.dtype.itemsize, track=False)
            shm_y = SharedMemory(f'{self.basename_y}_{shm_id}', create=True, size=capacity * y.dtype.itemsize,
                                 track=False)

        with self.lock:
            if grow:
                if self.capacity:
                    #  here, before, reallocating we unlink the underlying shared memory without making sure that the
                    #  training loop process has had a chance to close() it first, so this might lead to some warnings
                    #  references:
                    #  - https://stackoverflow.com/a/63004750/2447427
                    #  - https://github.com/python/cpython/issues/84140
                    #  - https://github.com/python/cpython/issues/82300
                    #    - comment provides a fix that turns off tracking:
                    #    https://github.com/python/cpython/issues/82300#issuecomment-2169035092
                    self.close()
                self.shm_X = shm_X
                self.shm_y = shm_y
                self.shm_id = shm_id
            self.size = size
            self.X[...] = X
            self.y[...] = y
