#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Iterable, Optional, Union

from multiprocessing import Lock, Queue, Process
import queue
import sys

from numpy import typing as npt
import numpy as np
from pyrfr.regression import (binary_rss_forest as BinaryForest, default_random_engine as DefaultRandomEngine,
                              forest_opts as ForestOpts)

from .GrowingSharedArray import GrowingSharedArrayReaderView, GrowingSharedArray
from ..util import get_rf_opts, train

from enum import Enum, auto, unique


@unique
class Concurrency(Enum):
    THREADING = auto()
    THREADING_SYNCED = auto()
    MULTIPROC = auto()
    MULTIPROC_SYNCED = auto()


SHUTDOWN = None


ENABLE_DBG_PRINT = False


def debug_print(*args, file=sys.stdout, **kwargs):
    if ENABLE_DBG_PRINT:
        print(*args, **kwargs, flush=True, file=file)
        file.flush()


# TODO: the type of the value passed for the 'bounds' param below is a tuple of tuples. Might this add some memory
#  dependency between the processes which might mess up the cleanup process?
def rf_training_loop(
        model_queue: Queue, data_queue: Queue, data_lock: Lock,
        # init rf train
        bounds: Iterable[tuple[float, float]], seed: int,
        # rf opts
        n_trees: int, bootstrapping: bool, max_features: int, min_samples_split: int, min_samples_leaf: int,
        max_depth: int, eps_purity: float, max_nodes: int, n_points_per_tree: int
) -> None:
    rf_opts = get_rf_opts(n_trees, bootstrapping, max_features, min_samples_split, min_samples_leaf, max_depth,
                          eps_purity, max_nodes, n_points_per_tree)

    # Cast to `int` incase we get an `np.integer` type
    rng = DefaultRandomEngine(int(seed))
    shared_arrs = GrowingSharedArrayReaderView(data_lock)

    def send_to_optimization_loop_process(msg: Union[BinaryForest, type(SHUTDOWN)]):
        # remove previous models from queue, if any, before pushing the latest model
        while True:
            try:
                _ = model_queue.get(block=False)
            except queue.Empty:
                break
        debug_print(f'TRAINER SENDING {"SHUTDOWN CONFIRM" if msg == SHUTDOWN else "MODEL"}', file=sys.stderr)
        model_queue.put(msg)
        debug_print(f'TRAINER SENDING {"SHUTDOWN CONFIRM" if msg == SHUTDOWN else "MODEL"} DONE', file=sys.stderr)

    while True:
        debug_print('TRAINER WAIT MSG', file=sys.stderr)
        data_msg = data_queue.get()  # if queue is empty, wait for training data or shutdown signal
        debug_print(f'TRAINER GOT MSG: {data_msg}', file=sys.stderr)
        must_shutdown = data_msg == SHUTDOWN
        if must_shutdown:
            debug_print(f'TRAINER GOT SHUTDOWN 1', file=sys.stderr)

        # discard all but the last data_msg in the queue
        while True:
            try:
                data_msg = data_queue.get(block=False)
            except queue.Empty:
                break
            else:
                if data_msg == SHUTDOWN:
                    debug_print(f'TRAINER GOT SHUTDOWN 2', file=sys.stderr)
                must_shutdown = must_shutdown or data_msg == SHUTDOWN
        if must_shutdown:
            shared_arrs.close()
            send_to_optimization_loop_process(SHUTDOWN)
            # don't kill current process until we make sure the queue's underlying pipe is flushed
            model_queue.close()
            model_queue.join_thread()
            break

        shm_id, size = data_msg
        X, y = shared_arrs.get_data(shm_id, size)
        # when shm_id changes, here we should notify main thread it can call unlink the shared memory bc we called
        # close() on it
        # UPDATE: we avoided the warnings by disabling tracking for shared memory

        rf = train(rng, rf_opts, n_points_per_tree, bounds, X, y)

        send_to_optimization_loop_process(rf)
    debug_print(f'TRAINER BYE BYE', file=sys.stderr)


class RFTrainer:
    def __init__(self,
                 # init rf train
                 bounds: Iterable[tuple[float, float]], seed: int,
                 # rf opts
                 n_trees: int, bootstrapping: bool, max_features: int, min_samples_split: int, min_samples_leaf: int,
                 max_depth: int, eps_purity: float, max_nodes: int, n_points_per_tree: int,
                 # process synchronization
                 background_training: Optional[Concurrency] = None) -> None:
        self.background_training = background_training

        self._model: Optional[BinaryForest] = None
        self.shared_arrs: Optional[GrowingSharedArray] = None
        self.model_queue: Optional[Queue] = None
        self.data_queue: Optional[Queue] = None
        self.training_loop_proc: Optional[Process] = None

        # in case we disable training in the background, and we need these objects in the main thread
        self.opts: ForestOpts = get_rf_opts(n_trees, bootstrapping, max_features, min_samples_split, min_samples_leaf,
                                            max_depth, eps_purity, max_nodes, n_points_per_tree)
        self.n_points_per_tree: int = n_points_per_tree
        self.bounds = tuple(bounds)

        # this is NOT used when training in background
        # Cast to `int` incase we get an `np.integer` type
        self.rng = DefaultRandomEngine(int(seed))

        self.open(seed)

        super().__init__()

    def open(self, seed: int) -> None:
        assert self.background_training is None or self.background_training in Concurrency
        if self.background_training is None:
            pass
        elif self.background_training is Concurrency.THREADING:
            raise NotImplementedError
        elif self.background_training is Concurrency.THREADING_SYNCED:
            raise NotImplementedError
        else:
            self.shared_arrs = GrowingSharedArray()
            self.model_queue = Queue(maxsize=1)
            self.data_queue = Queue(maxsize=1)
            self.training_loop_proc = Process(
                target=rf_training_loop,
                daemon=True,
                name='rf_trainer',
                args=(self.model_queue, self.data_queue, self.shared_arrs.lock, self.bounds, seed, self.opts.num_trees,
                      self.opts.do_bootstrapping, self.opts.tree_opts.max_features,
                      self.opts.tree_opts.min_samples_to_split, self.opts.tree_opts.min_samples_in_leaf,
                      self.opts.tree_opts.max_depth, self.opts.tree_opts.epsilon_purity,
                      self.opts.tree_opts.max_num_nodes, self.n_points_per_tree)
            )
            self.training_loop_proc.start()

    def close(self):
        # send kill signal to training process
        if self.data_queue is not None:
            if self.training_loop_proc is not None:
                debug_print('MAIN SEND SHUTDOWN')
                self.send_to_training_loop_proc(SHUTDOWN)
                debug_print('MAIN FINISHED SEND SHUTDOWN')
            # make sure the shutdown message is flush before moving on
            self.data_queue.close()
            self.data_queue.join_thread()
            del self.data_queue
            self.data_queue = None

        # wait till the training process died
        if self.model_queue is not None and self.training_loop_proc is not None and self.training_loop_proc.is_alive():
            # flush the model queue, and store the latest model
            while True:
                debug_print('MAIN WAIT SHUTDOWN CONFIRM')
                msg = self.model_queue.get()
                debug_print(f'MAIN RECEIVED {"SHUTDOWN CONFIRMATION" if msg == SHUTDOWN else "MODEL"}'
                            f' AFTER WAITING FOR SHUTDOWN CONFIRMATION')
                # wait for SHUTDOWN message, because that guarantees that shared_arrs.close() has been called within
                # the training process; this way we make sure we call unlink only after close has had the chance to be
                # called within the child process
                if msg == SHUTDOWN:
                    break
                else:
                    self._model = msg

        if self.training_loop_proc is not None:
            # wait for training to finish
            if self.training_loop_proc.is_alive():
                self.training_loop_proc.join()
            del self.training_loop_proc
            self.training_loop_proc = None

        if self.model_queue is not None:
            del self.model_queue
            self.model_queue = None

        # make sure this is called after SHUTDOWN was received because we want the trainer process to call
        # shared_arrs.close() before we call unlink
        if self.shared_arrs is not None:
            self.shared_arrs.close()
            del self.shared_arrs
            self.shared_arrs = None

    def __del__(self):
        self.close()

    @property
    def model(self) -> BinaryForest:
        if self._model is None:
            if self.model_queue is None:
                raise RuntimeError('rf training loop process has been stopped before being able to train a model')
            # wait until the first training is done
            msg = self.model_queue.get()
            if msg == SHUTDOWN:
                raise RuntimeError("the shutdown message wasn't supposed to end up here")
            else:
                self._model = msg

        if self.model_queue is not None:
            # discard all but the last model in the queue
            while True:
                try:
                    msg = self.model_queue.get(block=False)
                except queue.Empty:
                    break
                else:
                    if msg == SHUTDOWN:
                        raise RuntimeError("the shutdown message wasn't supposed to end up here")
                    else:
                        self._model = msg

        return self._model

    def send_to_training_loop_proc(self, data_info: Union[tuple[int, int], type[SHUTDOWN]]):
        if self.data_queue is None:
            raise RuntimeError('rf training loop process has been stopped, so we cannot submit new training data')

        # empty queue before pushing new data onto it
        while True:
            try:
                old_data = self.data_queue.get(block=False)
            except queue.Empty:
                break
            else:
                assert old_data != SHUTDOWN
        self.data_queue.put(data_info)

    def submit_for_training(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
        if self.background_training is None:
            self._model = train(self.rng, self.opts, self.n_points_per_tree, self.bounds, X, y)
        else:
            if self.background_training in (Concurrency.THREADING, Concurrency.THREADING_SYNCED):
                raise NotImplementedError
            self.shared_arrs.set_data(X, y)
            self.send_to_training_loop_proc((self.shared_arrs.shm_id, len(X)))
            if self.background_training is Concurrency.MULTIPROC_SYNCED:
                self._model = self.model_queue.get()
