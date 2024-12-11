#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Iterable, Optional

from multiprocessing import Lock, Queue, Process
import queue

from numpy import typing as npt
import numpy as np
from pyrfr import regression
from pyrfr.regression import binary_rss_forest as BinaryForest

from .GrowingSharedArray import GrowingSharedArrayReaderView, GrowingSharedArray
from ..util import init_data_container


SHUTDOWN = -1


def rf_training_loop(
        model_queue: Queue, data_queue: Queue, data_lock: Lock,
        # init rf train
        bounds: Iterable[tuple[float, float]], seed: int,
        # rf opts
        n_trees: int, bootstrapping: bool, max_features: int, min_samples_split: int, min_samples_leaf: int,
        max_depth: int, eps_purity: float, max_nodes: int, n_points_per_tree: int
):
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
        msg = data_queue.get()  # wait for training data or shutdown signal
        # discard all but the last msg in the queue
        while True:
            try:
                msg = data_queue.get(block=False)
            except queue.Empty:
                break

        if msg == SHUTDOWN:
            break

        shm_id, size = msg

        X, y = shared_arrs.get_data(shm_id, size)

        data = init_data_container(X, y, bounds)

        if n_points_per_tree <= 0:
            rf_opts.num_data_points_per_tree = len(X)

        rf = BinaryForest()
        rf.options = rf_opts

        rf.fit(data, rng)

        # remove previous models from pipe, if any
        while True:
            m = model_queue.get(block=False)
            if m is None:
                break
        model_queue.put(rf)


class RFTrainer:
    def __init__(
            # init rf train
            self, bounds: Iterable[tuple[float, float]], seed: int,
            # rf opts
            n_trees: int, bootstrapping: bool, max_features: int, min_samples_split: int, min_samples_leaf: int,
            max_depth: int, eps_purity: float, max_nodes: int, n_points_per_tree: int
    ) -> None:
        self._model: Optional[BinaryForest] = None
        self.model_queue = Queue(maxsize=1)
        self.data_queue = Queue(maxsize=1)
        self.shared_arrs = GrowingSharedArray()

        self.training_loop_proc = Process(daemon=True, target=rf_training_loop, name='rf_trainer', args=(
            self.model_queue, self.data_queue, self.shared_arrs.lock, tuple(bounds), seed, n_trees, bootstrapping,
            max_features, min_samples_split, min_samples_leaf, max_depth, eps_purity, max_nodes, n_points_per_tree
        ))
        self.training_loop_proc.start()

        super().__init__()

    def close(self):
        # I think this might be redundant, since according to the official docs, close and join_thread are called
        # anyway when garbage-collecting queues, and we don't use JoinableQueues
        if self.data_queue is not None:
            if self.training_loop_proc is not None:
                self.data_queue.put(SHUTDOWN)
            self.data_queue.close()
            self.data_queue.join_thread()
            del self.data_queue
            self.data_queue = None

        if self.training_loop_proc is not None:
            # wait for training to finish
            self.training_loop_proc.join()
            del self.training_loop_proc
            self.training_loop_proc = None

        if self.model_queue is not None:
            _ = self.model  # flush the model queue, and return latest model
            self.model_queue.close()
            self.model_queue.join_thread()
            del self.model_queue
            self.model_queue = None

    def __del__(self):
        self.close()

    @property
    def model(self) -> Optional[BinaryForest]:
        # discard all but the last model in the queue
        while True:
            try:
                self._model = self.model_queue.get(block=False)
            except queue.Empty:
                break
        return self._model

    def submit_for_training(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
        self.shared_arrs.set_data(X, y)
        self.data_queue.put((self.shared_arrs.shm_id, len(X)))
