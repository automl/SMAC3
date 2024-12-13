#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import numpy as np
from pyrfr.regression import (default_data_container as DataContainer, forest_opts as ForestOpts,
                              binary_rss_forest as BinaryForest, default_random_engine as DefaultRandomEngine)

if TYPE_CHECKING:
    from numpy import typing as npt


def get_rf_opts(n_trees: int, bootstrapping: bool, max_features: int, min_samples_split: int, min_samples_leaf: int,
                max_depth: int, eps_purity: float, max_nodes: int, n_points_per_tree: int) -> ForestOpts:
    rf_opts = ForestOpts()
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

    return rf_opts


def init_data_container(X: npt.NDArray[np.float64], y: npt.NDArray[np.float64],
                        bounds: Iterable[tuple[float, float]]) -> DataContainer:
    """Fills a pyrfr default data container s.t. the forest knows categoricals and bounds for continous data.

    Parameters
    ----------
    X : np.ndarray [#samples, #hyperparameter + #features]
        Input data points.
    Y : np.ndarray [#samples, #objectives]
        The corresponding target values.

    Returns
    -------
    data : DataContainer
        The filled data container that pyrfr can interpret.
    """
    # Retrieve the types and the bounds from the ConfigSpace
    data = DataContainer(X.shape[1])

    for i, (mn, mx) in enumerate(bounds):
        if np.isnan(mx):
            data.set_type_of_feature(i, mn)
        else:
            data.set_bounds_of_feature(i, mn, mx)

    for row_X, row_y in zip(X, y):
        data.add_data_point(row_X, row_y)

    return data


def train(rng: DefaultRandomEngine, rf_opts: ForestOpts, n_points_per_tree: int, bounds: Iterable[tuple[float, float]],
          X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> BinaryForest:
    data = init_data_container(X, y, bounds)

    if n_points_per_tree <= 0:
        rf_opts.num_data_points_per_tree = len(X)

    rf = BinaryForest()
    rf.options = rf_opts

    rf.fit(data, rng)

    return rf

