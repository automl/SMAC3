#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Iterable, Optional, Union

from multiprocessing import Lock, Queue, Process
import queue

from numpy import typing as npt
import numpy as np
from pyrfr import regression
from pyrfr.regression import binary_rss_forest as BinaryForest

from .GrowingSharedArray import GrowingSharedArrayReaderView, GrowingSharedArray
from ..util import init_data_container


SHUTDOWN = None


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

    while True:
        # print('TRAINER WAIT MSG', flush=True)
        msg = data_queue.get()  # if queue is empty, wait for training data or shutdown signal
        # print(f'TRAINER GOT MSG: {msg}', flush=True)
        must_shutdown = msg == SHUTDOWN
        # if must_shutdown:
        #     print(f'TRAINER GOT SHUTDOWN 1', flush=True)

        # discard all but the last msg in the queue
        while True:
            try:
                msg = data_queue.get(block=False)
            except queue.Empty:
                break
            else:
                # if msg == SHUTDOWN:
                #     print(f'TRAINER GOT SHUTDOWN 2', flush=True)
                must_shutdown = must_shutdown or msg == SHUTDOWN
        if must_shutdown:
            shared_arrs.close()
            # TODO: empty queue before pushing SHUTDOWN
            # print(f'TRAINER SENDS SHUTDOWN CONFIRMATION', flush=True)
            model_queue.put(SHUTDOWN)
            # print(f'TRAINER FINISHED SEND SHUTDOWN CONFIRMATION', flush=True)
            model_queue.close()
            # model_queue.join_thread()  # TODO: enable this again
            # print(f'TRAINER BYE BYE', flush=True)
            break

        shm_id, size = msg
        X, y = shared_arrs.get_data(shm_id, size)
        # when shm_id changes, here we should notify main thread it can call unlink the shared memory bc we called
        # close() on it
        # UPDATE: we avoided the warnings by disabling tracking for shared memory
        data = init_data_container(X, y, bounds)

        if n_points_per_tree <= 0:
            rf_opts.num_data_points_per_tree = len(X)

        rf = BinaryForest()
        rf.options = rf_opts
        rf.fit(data, rng)

        # print(f'TRAINER FINISHED TRAINING', flush=True)

        # remove previous models from queue, if any, before pushing the latest model
        while True:
            try:
                _ = model_queue.get(block=False)
            except queue.Empty:
                break
        # print(f'TRAINER SENDING MODEL', flush=True)
        model_queue.put(rf)
        # print(f'TRAINER SENDING MODEL DONE', flush=True)


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
            target=rf_training_loop, daemon=True,  name='rf_trainer',
            args=(self.model_queue, self.data_queue, self.shared_arrs.lock, tuple(bounds), seed, n_trees, bootstrapping,
                  max_features, min_samples_split, min_samples_leaf, max_depth, eps_purity, max_nodes,
                  n_points_per_tree)
        )
        self.training_loop_proc.start()

    def close(self):
        # send kill signal to training process
        if self.data_queue is not None:
            if self.training_loop_proc is not None:
                # print('MAIN SEND SHUTDOWN', flush=True)
                self.send_to_training_loop_proc(SHUTDOWN)
                # print('MAIN FINISHED SEND SHUTDOWN', flush=True)
            # make sure the shutdown message is flush before moving on
            self.data_queue.close()
            self.data_queue.join_thread()

        # wait till the training process died
        if self.model_queue is not None and self.training_loop_proc is not None and self.training_loop_proc.is_alive():
            # flush the model queue, and store the latest model
            while True:
                # print('MAIN WAIT SHUTDOWN CONFIRM', flush=True)
                msg = self.model_queue.get()
                # print(f'MAIN RECEIVED {"SHUTDOWN CONFIRMATION" if msg == SHUTDOWN else msg} '
                #       f'AFTER WAITING FOR SHUTDOWN CONFIRMATION', flush=True)
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

        # I think this might be redundant, since according to the official docs, close and join_thread are called
        # anyway when garbage-collecting queues, and we don't use JoinableQueues
        if self.data_queue is not None:
            del self.data_queue
            self.data_queue = None

        if self.model_queue is not None:
            # self.model_queue.close()
            # self.model_queue.join_thread()
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

        if self.data_queue is None:
            raise RuntimeError('rf training loop process has been stopped, so we cannot submit new training data')

        self.send_to_training_loop_proc((self.shared_arrs.shm_id, len(X)))

        if self.sync:
            self._model = self.model_queue.get()
