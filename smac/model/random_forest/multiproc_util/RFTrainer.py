#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Iterable, Optional, Union

from multiprocessing import Lock, Queue, Process
import queue
import sys

from numpy import typing as npt
import numpy as np
from pyrfr import regression
from pyrfr.regression import binary_rss_forest as BinaryForest

from .GrowingSharedArray import GrowingSharedArrayReaderView, GrowingSharedArray
from ..util import init_data_container


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
    rf_opts = regression.forest_opts()
    rf_opts.num_trees = n_trees
    rf_opts.do_bootstrapping = bootstrapping
    rf_opts.tree_opts.max_features = max_features
    rf_opts.tree_opts.min_samples_to_split = min_samples_split
    rf_opts.tree_opts.min_samples_in_leaf = min_samples_leaf
    rf_opts.tree_opts.max_depth = max_depth
    rf_opts.tree_opts.epsilon_purity = eps_purity
    rf_opts.tree_opts.max_num_nodes = max_nodes
    rf_opts.compute_law_of_total_variance = False
    if n_points_per_tree > 0:
        rf_opts.num_data_points_per_tree = n_points_per_tree

    # Case to `int` incase we get an `np.integer` type
    rng = regression.default_random_engine(int(seed))
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
        data = init_data_container(X, y, bounds)

        if n_points_per_tree <= 0:
            rf_opts.num_data_points_per_tree = len(X)

        rf = BinaryForest()
        rf.options = rf_opts
        debug_print(f'TRAINER STARTS TRAINING', file=sys.stderr)
        rf.fit(data, rng)
        debug_print(f'TRAINER FINISHED TRAINING', file=sys.stderr)
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
                 sync: bool = False) -> None:
        self.sync = sync

        self._model: Optional[BinaryForest] = None
        self.shared_arrs: Optional[GrowingSharedArray] = None
        self.model_queue: Optional[Queue] = None
        self.data_queue: Optional[Queue] = None
        self.training_loop_proc: Optional[Process] = None

        self.open(bounds, seed, n_trees, bootstrapping, max_features, min_samples_split, min_samples_leaf, max_depth,
                  eps_purity, max_nodes, n_points_per_tree)

        super().__init__()

    def open(self,
             # init rf train
             bounds: Iterable[tuple[float, float]], seed: int,
             # rf opts
             n_trees: int, bootstrapping: bool, max_features: int, min_samples_split: int, min_samples_leaf: int,
             max_depth: int, eps_purity: float, max_nodes: int, n_points_per_tree: int) -> None:
        self.shared_arrs = GrowingSharedArray()
        self.model_queue = Queue(maxsize=1)
        self.data_queue = Queue(maxsize=1)
        self.training_loop_proc = Process(
            target=rf_training_loop,
            daemon=True,
            # name='rf_trainer',
            args=(self.model_queue, self.data_queue, self.shared_arrs.lock, tuple(bounds), seed, n_trees, bootstrapping,
                  max_features, min_samples_split, min_samples_leaf, max_depth, eps_purity, max_nodes,
                  n_points_per_tree)
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
        self.shared_arrs.set_data(X, y)
        self.send_to_training_loop_proc((self.shared_arrs.shm_id, len(X)))
        if self.sync:
            self._model = self.model_queue.get()
