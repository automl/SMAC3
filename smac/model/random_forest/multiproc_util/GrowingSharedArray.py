#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional

import math
from multiprocessing import Lock, shared_memory

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
        self.shm_X: Optional[shared_memory.SharedMemory] = None
        self.shm_y: Optional[shared_memory.SharedMemory] = None

    def open(self, shm_id: int):
        if shm_id != self.shm_id:
            self.close()
            self.shm_X = shared_memory.SharedMemory(f'{self.basename_X}_{shm_id}')
            self.shm_y = shared_memory.SharedMemory(f'{self.basename_y}_{shm_id}')
            self.shm_id = shm_id

    def close(self):
        if self.shm_X is not None:
            self.shm_X.close()
            del self.shm_X
            self.shm_X = None
        if self.shm_y is not None:
            self.shm_y.close()
            del self.shm_y
            self.shm_y = None
        self.shm_id = None

    def __del__(self):
        self.close()

    @property
    def capacity(self) -> Optional[int]:
        if self.shm_y is None:
            return None
        assert self.shm_y.size % np.float64.itemsize == 0
        return self.shm_y.size / np.float64.itemsize

    @property
    def row_size(self) -> Optional[int]:
        if self.shm_X is None:
            return None
        if self.shm_X.size == 0:
            assert self.shm_y.size == 0
            return 0
        assert self.shm_X.size % self.shm_y.size == 0
        return self.shm_X.size // self.shm_y.size

    def np_view(self, size: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        X = np.ndarray(shape=(self.capacity, self.row_size), dtype=np.float64, buffer=self.shm_X.buf)
        y = np.ndarray(shape=(self.capacity,), dtype=np.float64, buffer=self.shm_y.buf)
        return X[:size], y[:size]

    def get_data(self, shm_id: int, size: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        with self.lock:
            self.open(shm_id)
            shared_X, shared_y = self.np_view(size)
            X, y = np.array(shared_X), np.array(shared_y)  # make copies

        return X, y


class GrowingSharedArray(GrowingSharedArrayReaderView):
    def __init__(self):
        self.growth_rate = 1.5
        super().__init__(lock=Lock())

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
                self.shm_id += 1
            else:
                assert self.shm_X is None
                assert self.shm_y is None
                capacity = size
                self.shm_id = 0

            if self.row_size is not None:
                assert X.shape[1] == self.row_size

            shm_X = shared_memory.SharedMemory(f'{self.basename_X}_{self.shm_id}', create=True,
                                               size=capacity * self.row_size * X.dtype.itemsize)
            shm_y = shared_memory.SharedMemory(f'{self.basename_y}_{self.shm_id}', create=True,
                                               size=capacity * y.dtype.itemsize)

        with self.lock:
            if grow:
                if self.capacity:
                    assert self.shm_X is not None
                    self.shm_X.close()
                    self.shm_X.unlink()
                    assert self.shm_y is not None
                    self.shm_y.close()
                    self.shm_y.unlink()
                self.shm_X = shm_X
                self.shm_y = shm_y
            X_buf, y_buf = self.np_view(size)
            X_buf[...] = X
            y_buf[...] = y

