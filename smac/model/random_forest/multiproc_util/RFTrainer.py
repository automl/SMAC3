#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING, Optional

from multiprocessing import Lock, Queue

from numpy import typing as npt
import numpy as np
from pyrfr.regression import binary_rss_forest

if TYPE_CHECKING:
    from pyrfr.regression import binary_rss_forest as BinaryForest, forest_opts as ForestOpts

from .GrowingSharedArray import GrowingSharedArrayReaderView


SHUTDOWN = -1


def training_loop_proc(data_queue: Queue, data_lock: Lock):
    shared_arrs = GrowingSharedArrayReaderView(data_lock)
    while True:
        # discard all but the last msg in the queue
        msg = None
        while True:
            m = data_queue.get(block=False)
            if m is None:
                break
            else:
                msg = m

        if msg == SHUTDOWN:
            break

        shm_id, size = msg

        X, y = shared_arrs.get_data(shm_id, size)

        data = init_data_container(X, y)

        _rf = binary_rss_forest()
        _rf.options = opts

        _rf.fit(data, rng=self._rng)


class RFTrainer:
    def __init__(self):
        self._model: Optional[BinaryForest] = None
        self.model_lock = Lock()
        self.model_queue = Queue(maxsize=1)

        self.opts = None
        self.data_queue = Queue(maxsize=1)

        super().__init__(daemon=True)
        self.start()

    @property
    def model(self):
        # discard all but the last model in the queue
        model = None
        while True:
            m = self.model_queue.get(block=False)
            if m is None:
                break
            else:
                model = m

        with self.model_lock:
            if model is not None:
                self._model = model
            return self._model

    def submit_for_training(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64], opts: ForestOpts):
        # use condition variable to wake up the trainer thread if it's sleeping
        with self.data_cv:
            assert data is not None
            # overwrite with latest training data
            self.data = data
            self.opts = opts
            self.data_cv.notify()

    def run(self) -> None:
        while True:
            # sleep until new data is submitted for training
            with self.data_cv:
                while self.data is None:
                    self.data_cv.wait()
                data = self.data
                self.data = None

            # here we could (conditionally) call self.model_available.clear() in order to make _some_ worker threads
            # wait for training to finish before receiving a new configuration to try, depending on CPU load; we might
            # have to replace the Event by a Condition



            with self.model_lock:
                self._model = _rf

            if not self.model_available.is_set():
                self.model_available.set()
