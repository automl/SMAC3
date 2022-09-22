import unittest

import numpy as np
import pytest
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.forbidden import (
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    ForbiddenInClause,
)
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from smac.model.utils import check_subspace_points, get_types
from smac.utils.subspaces import ChallengerListLocal, LocalSubspace


def generate_cont_hps():
    hp1 = UniformIntegerHyperparameter("non_log_uniform_int", lower=0, upper=100, log=False)
    hp2 = UniformIntegerHyperparameter("log_uniform_int", lower=1, upper=100, log=True)

    hp3 = UniformFloatHyperparameter("non_log_uniform_float", lower=0.0, upper=100.0, log=False)
    hp4 = UniformFloatHyperparameter("log_uniform_float", lower=1.0, upper=100.0, log=True)
    return [hp1, hp2, hp3, hp4]


def generate_ord_hps():
    hp1 = OrdinalHyperparameter("ord_hp_1", sequence=[1, 2, 3, 4, 5, 6, 7])
    return [hp1]


def generate_cat_hps(num_hps: int = 1):
    hps = []
    for i in range(num_hps):
        hp = CategoricalHyperparameter(f"cat_hp_{i}", choices=["a", "b", "c", "d"])
        hps.append(hp)
    return hps


def generate_ss_bounds(cs):
    bounds_ss_cont = []
    for hp in cs.get_hyperparameters():
        if isinstance(hp, (UniformIntegerHyperparameter, UniformFloatHyperparameter)):
            if hp.log:
                bounds_ss_cont.append((0.1, 0.9))
            else:
                bounds_ss_cont.append((0.05, 0.95))
        if isinstance(hp, OrdinalHyperparameter):
            bounds_ss_cont.append((1, 4))
    return bounds_ss_cont


@unittest.mock.patch.multiple(LocalSubspace, __abstractmethods__=set())
def test_cs_subspace_1():
    cs = ConfigurationSpace()
    hps = generate_cont_hps()
    hps.extend(generate_ord_hps())
    hps.extend(generate_cat_hps())
    cs.add_hyperparameters(hps)

    types, bounds = get_types(cs)
    bounds_ss_cont = np.array(generate_ss_bounds(cs))

    bounds_ss_cat = [(1, 3)]
    subspace = LocalSubspace(
        config_space=cs,
        bounds=bounds,
        hps_types=types,
        bounds_ss_cont=bounds_ss_cont,
        bounds_ss_cat=bounds_ss_cat,
        model_local=None,
    )

    for hp in subspace.cs_local.get_hyperparameters():
        if isinstance(hp, CategoricalHyperparameter):
            # for categorical hps, we set bound as 1, 3, i.e., the 2nd and 4th element should be selected for
            # building a subspace
            assert hp.choices == ("b", "d")

        elif isinstance(hp, OrdinalHyperparameter):
            # for ordinary hps, we set bound as (1,3), i.e., we select the 2nd, 3rd, 4th values of
            # the ordinary sequence
            assert hp.sequence == (2, 3, 4)
        elif isinstance(hp, UniformFloatHyperparameter):
            if hp.log:
                raw_hp_range = (1, 100)
                # we map from [0., 1.] to [0.1, 0.5]
                raw_hp_range_log = np.log(raw_hp_range)
                new_hp_range_log_lower = 0.1 * (raw_hp_range_log[1] - raw_hp_range_log[0]) + raw_hp_range_log[0]
                new_hp_range_log_upper = 0.9 * (raw_hp_range_log[1] - raw_hp_range_log[0]) + raw_hp_range_log[0]

                new_range = np.exp([new_hp_range_log_lower, new_hp_range_log_upper])
                np.testing.assert_almost_equal([hp.lower, hp.upper], new_range)
            else:
                raw_hp_range = (0, 100)
                new_hp_range_lower = 0.05 * (raw_hp_range[1] - raw_hp_range[0]) + raw_hp_range[0]
                new_hp_range_upper = 0.95 * (raw_hp_range[1] - raw_hp_range[0]) + raw_hp_range[0]
                new_range = [new_hp_range_lower, new_hp_range_upper]

                np.testing.assert_almost_equal([hp.lower, hp.upper], new_range)
        elif isinstance(hp, UniformIntegerHyperparameter):
            if hp.log:
                raw_hp_range = (1, 100)
                raw_hp_range_log = np.log(raw_hp_range)
                new_hp_range_log_lower = 0.1 * (raw_hp_range_log[1] - raw_hp_range_log[0]) + raw_hp_range_log[0]
                new_hp_range_log_upper = 0.9 * (raw_hp_range_log[1] - raw_hp_range_log[0]) + raw_hp_range_log[0]
                new_range[0] = np.floor(np.exp(new_hp_range_log_lower))
                new_range[1] = np.ceil(np.exp(new_hp_range_log_upper))
                new_range = np.asarray(new_range, dtype=np.int32)
                np.testing.assert_equal([hp.lower, hp.upper], new_range)
            else:
                raw_hp_range = (0, 100)
                new_hp_range_lower = 0.05 * (raw_hp_range[1] - raw_hp_range[0]) + raw_hp_range[0]
                new_hp_range_upper = 0.95 * (raw_hp_range[1] - raw_hp_range[0]) + raw_hp_range[0]
                new_range[0] = np.floor(new_hp_range_lower)
                new_range[1] = np.ceil(new_hp_range_upper)
                new_range = np.asarray(new_range, dtype=np.int32)
                np.testing.assert_equal([hp.lower, hp.upper], new_range)


@unittest.mock.patch.multiple(LocalSubspace, __abstractmethods__=set())
def test_cs_subspace_2():
    # check act_dims
    cs = ConfigurationSpace()
    hps = generate_cont_hps()
    hps.extend(generate_ord_hps())
    hps.extend(generate_cat_hps(2))

    cs.add_hyperparameters(hps)

    types, bounds = get_types(cs)
    bounds_ss_cont = np.array(generate_ss_bounds(cs))

    bounds_ss_cont = np.array(bounds_ss_cont)

    bounds_ss_cat = [(1, 3), (0, 2)]

    activ_dims = [0, 2, 6]

    subspace = LocalSubspace(
        config_space=cs,
        bounds=bounds,
        hps_types=types,
        bounds_ss_cont=bounds_ss_cont,
        bounds_ss_cat=bounds_ss_cat,
        model_local=None,
        activate_dims=activ_dims,
    )
    np.testing.assert_equal(subspace.activate_dims_cat, [0])
    np.testing.assert_equal(subspace.activate_dims_cont, [0, 4])

    hps_global = cs.get_hyperparameters()
    hps_local = subspace.cs_local.get_hyperparameters()
    for dim_idx, act_dim in enumerate(activ_dims):
        assert (hps_local[dim_idx].__class__ == hps_global[act_dim].__class__)


@unittest.mock.patch.multiple(LocalSubspace, __abstractmethods__=set())
def test_cs_subspace_3():
    # check if bounds is None works correctly
    # check act_dims
    cs = ConfigurationSpace()
    hps = generate_cont_hps()
    hps.extend(generate_ord_hps())
    hps.extend(generate_cat_hps(2))

    cs.add_hyperparameters(hps)

    types, bounds = get_types(cs)
    bounds_ss_cont = np.array(generate_ss_bounds(cs))

    bounds_ss_cont = np.array(bounds_ss_cont)
    bounds_ss_cat = [(1, 3), (0, 2)]
    hps_global = cs.get_hyperparameters()

    subspace = LocalSubspace(
        config_space=cs,
        bounds=bounds,
        hps_types=types,
        bounds_ss_cont=None,
        bounds_ss_cat=bounds_ss_cat,
        model_local=None,
    )
    hps_local = subspace.cs_local.get_hyperparameters()
    for hp_local, hp_global in zip(hps_global, hps_local):
        if isinstance(hp_local, CategoricalHyperparameter):
            assert hp_local != hp_global
        else:
            assert hp_local == hp_global

    subspace = LocalSubspace(
        config_space=cs,
        bounds=bounds,
        hps_types=types,
        bounds_ss_cont=bounds_ss_cont,
        bounds_ss_cat=None,
        model_local=None,
    )
    hps_local = subspace.cs_local.get_hyperparameters()
    for hp_local, hp_global in zip(hps_global, hps_local):
        if isinstance(hp_local, CategoricalHyperparameter):
            assert hp_local == hp_global
        else:
            assert hp_local != hp_global

    subspace = LocalSubspace(
        config_space=cs, bounds=bounds, hps_types=types, bounds_ss_cont=None, bounds_ss_cat=None, model_local=None
    )
    hps_local = subspace.cs_local.get_hyperparameters()
    for hp_local, hp_global in zip(hps_global, hps_local):
        assert hp_local == hp_global


@unittest.mock.patch.multiple(LocalSubspace, __abstractmethods__=set())
def test_ss_normalization():
    cs_global = ConfigurationSpace(1)
    hps = generate_cont_hps()
    hps.extend(generate_cat_hps(1))
    cs_global.add_hyperparameters(hps)

    types, bounds = get_types(cs_global)
    bounds_ss_cont = np.array(generate_ss_bounds(cs_global))

    bounds_ss_cat = [(1, 3)]

    subspace = LocalSubspace(
        config_space=cs_global,
        bounds=bounds,
        hps_types=types,
        bounds_ss_cont=bounds_ss_cont,
        bounds_ss_cat=bounds_ss_cat,
        model_local=None,
    )

    cs_local = subspace.cs_local
    samples_global = cs_global.sample_configuration(20)
    X_samples = np.array([sample.get_array() for sample in samples_global])
    X_normalized = subspace.normalize_input(X_samples)

    ss_indices = check_subspace_points(
        X=X_normalized,
        cont_dims=subspace.activate_dims_cont,
        cat_dims=subspace.activate_dims_cat,
        bounds_cont=subspace.bounds_ss_cont,
        bounds_cat=subspace.bounds_ss_cat,
    )

    ss_indices = np.where(ss_indices)[0]
    for ss_idx in ss_indices:
        x_normalized = X_normalized[ss_idx]

        sample_local = Configuration(cs_local, vector=x_normalized).get_dictionary()
        sample_global = samples_global[ss_idx].get_dictionary()
        for key in sample_local.keys():
            if "int" in key:
                # There is some numerical issues here for int hps
                assert sample_local[key] - sample_global[key] < 3
            else:
                assert sample_local[key] == sample_global[key]


@unittest.mock.patch.multiple(LocalSubspace, __abstractmethods__=set())
def test_add_new_observations():
    cs_global = ConfigurationSpace(1)
    hps = generate_cont_hps()
    hps.extend(generate_cat_hps(1))
    cs_global.add_hyperparameters(hps)

    types, bounds = get_types(cs_global)
    bounds_ss_cont = np.array(generate_ss_bounds(cs_global))

    bounds_ss_cat = [(1, 3)]

    subspace = LocalSubspace(
        config_space=cs_global,
        bounds=bounds,
        hps_types=types,
        bounds_ss_cont=bounds_ss_cont,
        bounds_ss_cat=bounds_ss_cat,
        model_local=None,
    )

    samples_global = cs_global.sample_configuration(20)
    X_samples = np.array([sample.get_array() for sample in samples_global])
    y_samples = np.ones(np.shape(X_samples)[0])

    ss_indices = check_subspace_points(
        X=X_samples,
        cont_dims=subspace.activate_dims_cont,
        cat_dims=subspace.activate_dims_cat,
        bounds_cont=subspace.bounds_ss_cont,
        bounds_cat=subspace.bounds_ss_cat,
    )

    subspace.add_new_observations(X_samples, y_samples)
    assert sum(ss_indices) == len(subspace.ss_x)
    assert sum(ss_indices) == len(subspace.ss_y)

    assert len(X_samples) == len(subspace.model_x)
    assert len(y_samples) == len(subspace.model_y)

    # test if initialization works
    subspace_1 = LocalSubspace(
        config_space=cs_global,
        bounds=bounds,
        hps_types=types,
        bounds_ss_cont=bounds_ss_cont,
        bounds_ss_cat=bounds_ss_cat,
        model_local=None,
        initial_data=(X_samples, y_samples),
    )

    np.testing.assert_allclose(subspace.ss_x, subspace_1.ss_x)
    np.testing.assert_allclose(subspace.ss_y, subspace_1.ss_y)
    np.testing.assert_allclose(subspace.model_x, subspace_1.model_x)
    np.testing.assert_allclose(subspace.model_y, subspace_1.model_y)


@unittest.mock.patch.multiple(LocalSubspace, __abstractmethods__=set())
def test_challenger_list_local_full():
    # check act_dims
    cs_global = ConfigurationSpace(1)
    hps = generate_cont_hps()
    hps.extend(generate_ord_hps())
    hps.extend(generate_cat_hps(2))

    cs_global.add_hyperparameters(hps)

    types, bounds = get_types(cs_global)
    bounds_ss_cont = np.array(generate_ss_bounds(cs_global))

    bounds_ss_cont = np.array(bounds_ss_cont)

    bounds_ss_cat = [(1, 3), (0, 2)]

    subspace = LocalSubspace(
        config_space=cs_global,
        bounds=bounds,
        hps_types=types,
        bounds_ss_cont=bounds_ss_cont,
        bounds_ss_cat=bounds_ss_cat,
        model_local=None,
    )
    cs_local = subspace.cs_local

    num_data = 10

    rs = np.random.RandomState(1)

    challengers = cs_local.sample_configuration(num_data)
    challengers = [(rs.rand(), challenger) for challenger in challengers]

    cl = ChallengerListLocal(cs_local, cs_global, challengers, config_origin="test", incumbent_array=None)

    assert len(cl) == num_data
    new_challenger = next(cl).get_dictionary()
    challenger_local = challengers[0][1].get_dictionary()

    for key in new_challenger.keys():
        if "int" in key:
            # There is some numerical issues here for int hps
            assert new_challenger[key] - challenger_local[key] < 3
        else:
            assert new_challenger[key] == challenger_local[key]

    assert next(cl).origin == "test"

@unittest.mock.patch.multiple(LocalSubspace, __abstractmethods__=set())
def test_challenger_list_local_reduced():
    # check act_dims
    cs_global = ConfigurationSpace(1)
    hps = generate_cont_hps()
    hps.extend(generate_ord_hps())
    hps.extend(generate_cat_hps(2))

    cs_global.add_hyperparameters(hps)

    types, bounds = get_types(cs_global)
    bounds_ss_cont = np.array(generate_ss_bounds(cs_global))

    bounds_ss_cont = np.array(bounds_ss_cont)

    bounds_ss_cat = [(1, 3), (0, 2)]

    activ_dims = [0, 2, 6]

    subspace = LocalSubspace(
        config_space=cs_global,
        bounds=bounds,
        hps_types=types,
        bounds_ss_cont=bounds_ss_cont,
        bounds_ss_cat=bounds_ss_cat,
        model_local=None,
        activate_dims=activ_dims,
    )

    cs_local = subspace.cs_local

    incumbent_array = cs_global.sample_configuration(1).get_array()

    num_data = 10

    rs = np.random.RandomState(1)

    challengers = cs_local.sample_configuration(num_data)
    challengers = [(rs.rand(), challenger) for challenger in challengers]

    cl = ChallengerListLocal(
        cs_local, cs_global, challengers, config_origin="test", incumbent_array=incumbent_array
    )

    new_challenger = next(cl)

    for i, hp in enumerate((cs_global.get_hyperparameters())):
        if i not in activ_dims:
            assert incumbent_array[i] == pytest.approx(new_challenger.get_array()[i], 1e-5)
        else:
            assert new_challenger.get_dictionary()[hp.name] == challengers[0][1].get_dictionary()[hp.name]


def test_exception():
    cs_local = ConfigurationSpace(1)
    hps = generate_cont_hps()
    cs_local.add_hyperparameters(hps)

    cs_global = ConfigurationSpace(1)
    hps.extend(generate_cat_hps())
    cs_global.add_hyperparameters(hps)

    challengers = cs_local.sample_configuration(5)
    challengers = [(0.0, challenger) for challenger in challengers]
    with pytest.raises(ValueError) as excinfo:
        ChallengerListLocal(cs_local, cs_global, challengers, "test")
        assert excinfo == "Incumbent array must be provided if the global configuration space has more " \
                          "hyperparameters then the local configuration space"


def test_add_forbidden_ss():
    f0 = UniformFloatHyperparameter("f0", 0.0, 100.0)
    c0 = CategoricalHyperparameter("c0", [0, 1, 2])
    o0 = OrdinalHyperparameter("o0", [1, 2, 3])

    i0 = UniformIntegerHyperparameter("i0", 0, 100)

    forbid_1 = ForbiddenEqualsClause(c0, 0)
    forbid_2 = ForbiddenInClause(o0, [1, 2])

    forbid_3 = ForbiddenEqualsClause(f0, 0.3)
    forbid_4 = ForbiddenEqualsClause(f0, 59.0)

    forbid_5 = ForbiddenAndConjunction(forbid_2, forbid_3)
    forbid_6 = ForbiddenAndConjunction(forbid_1, forbid_4)
    forbid_7 = ForbiddenEqualsClause(i0, 10)

    cs_local = ConfigurationSpace()
    f0_ss = UniformFloatHyperparameter("f0", 0.0, 50.0)
    c0_ss = CategoricalHyperparameter("c0", [0, 1, 2])
    o0_ss = OrdinalHyperparameter("o0", [1, 2, 3])
    cs_local.add_hyperparameters([f0_ss, c0_ss, o0_ss])

    assert LocalSubspace.fit_forbidden_to_ss(cs_local, forbid_1) is not None
    assert LocalSubspace.fit_forbidden_to_ss(cs_local, forbid_2) is not None

    assert LocalSubspace.fit_forbidden_to_ss(cs_local, forbid_3) is not None
    assert LocalSubspace.fit_forbidden_to_ss(cs_local, forbid_4) is None

    assert LocalSubspace.fit_forbidden_to_ss(cs_local, forbid_5) is not None
    assert LocalSubspace.fit_forbidden_to_ss(cs_local, forbid_6) is None

    assert LocalSubspace.fit_forbidden_to_ss(cs_local, forbid_7) is None
