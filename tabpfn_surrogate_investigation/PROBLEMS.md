# TabPFN-as-surrogate: problems to solve for a v3 reimplementation

Context: `feature/tabpfn_1239` (Daphne's branch) added a `TabPFNModel` (wrapping
`tabpfn.TabPFNRegressor`, the v2 API) plus a custom `RiemannExpectedImprovement`
acquisition function. That branch was cut from an old `main` and was never merged.
`main` has since gone through a preprocessing refactor
(`b1ade2b3d`, "Refactor surrogate models: fully extract preprocessing into
SurrogateTransformer") that changes the contract every `AbstractModel` subclass
must implement. On top of that, we're now targeting TabPFN **v3**, not v2.

Findings below are split into (A) problems that are about SMAC's own
architecture and hold regardless of which TabPFN version/API we bind to, and
(B) problems that are specific to whatever the PFN library actually does
internally and need re-checking against the real v3 API before we write code
against them — I have not verified v3's API, so these are framed as open
questions, not facts.

All line numbers reference current `main` unless stated otherwise.

---

## A. SMAC-side architecture problems (version-agnostic)

### A1. `AbstractModel` contract changed — Daphne's model no longer even instantiates (blocker)

`smac/model/abstract_model.py` now requires every subclass to:
- assign `self.transformer: SurrogateTransformer` in `__init__` (`abstract_model.py:58`)
- implement `build_transformer(normalize_y: bool) -> SurrogateTransformer` (abstract, `abstract_model.py:81`)
- rely on the base `train()`/`predict()` to call
  `self.transformer.fit_transform(X, Y)` / `self.transformer.transform_X(X)`
  (`abstract_model.py:133`, `:189`) — a subclass's `_train`/`_predict` now
  receives already-preprocessed `X`, not raw `X`.

Every existing model builds its transformer explicitly, e.g. RF
(`smac/model/random_forest/abstract_random_forest.py:30`), GP
(`smac/model/gaussian_process/abstract_gaussian_process.py:57`), the plain
`RandomModel` (`smac/model/random_model.py:22`), and `MultiObjectiveModel`
(`smac/model/multi_objective_model.py:53`).

Daphne's `TabPFNModel` predates this entirely: it never sets `self.transformer`
and never implements `build_transformer`. On current `main`, simply
constructing `TabPFNModel(...)` and calling `.train()` would raise
`AttributeError: 'TabPFNModel' object has no attribute 'transformer'`
immediately. **This alone means the branch can't just be rebased — the model
needs a new `_train`/`_predict`/`build_transformer` written against the
current contract.**

### A2. Inactive (conditional) hyperparameters arrive as `NaN`, and someone has to handle them

`RunHistoryEncoder.transform()` builds `X` as an all-`NaN` matrix and only
fills in values for *active* hyperparameters
(`smac/runhistory/encoder/encoder.py:35-37`). Every model is responsible for
turning those `NaN`s into something usable via its own `_impute_inactive`,
which is plugged into its transformer:

- RF: per-hyperparameter-type sentinel — `len(choices)` for categoricals,
  `len(sequence)` for ordinals, `-1` for continuous, `1` for constants
  (`smac/model/random_forest/abstract_random_forest.py:42-59`). The point is
  to give "inactive" its own distinct, learnable value rather than treating it
  as noise.
- GP: a single `-1` for every inactive entry
  (`smac/model/gaussian_process/abstract_gaussian_process.py:153-157`).

Daphne's branch had this at one point — commit `ab11a4dc1` added
`SimpleImputer(strategy="mean")` — but the final "refactor" commit
(`cd0a4f039`) deleted it and nothing replaced it. Her own shipped example
(`examples/4_advanced_optimizer/5_tabPFN_surrogate_model.py`) has *four*
conditional hyperparameters (`degree`, `coef0`, `gamma`, `gamma_value` all
gated by `InCondition`s), so this isn't a corner case — it's exercised on the
very first run.

Whatever v3's actual missing-value story turns out to be (see B1), "inactive
because the parent condition is false" is a **structural** absence, not a
statistical missing-at-random value — RF/GP deliberately encode it as an
extra, distinct, learnable state rather than "missing." A v3 `TabPFNModel`
needs its own `_impute_inactive` that makes a deliberate choice here, not an
implicit one inherited from whatever the library does with `NaN` by default.

### A3. Continuous vs. ordinal vs. categorical encoding reaching the model — answering your question directly

This is entirely SMAC/ConfigSpace's doing, not something any current model
adds on top:

- **Continuous** (`UniformFloatHyperparameter`/`UniformIntegerHyperparameter`):
  ConfigSpace's own array representation already normalizes these to `[0, 1]`
  (`[-1, 1]` if the hyperparameter can be inactive) — see
  `smac/utils/configspace.py:94-105`, comment: *"Are sampled on the unit
  hypercube thus the bounds are always 0.0, 1.0"*. Log-scale hyperparameters
  (`log=True`) are log-transformed **before** this min-max step, so a
  log-uniform HP arrives linearly spaced in `[0,1]` in log-space, not raw
  units.
- **Ordinal** (`OrdinalHyperparameter`): encoded as small raw integers
  `[0, n_categories-1]` (or `n_categories` if it can be inactive) —
  `smac/utils/configspace.py:79-85`. Note this is **not** rescaled to `[0,1]`
  the way Float/Int are — it's on a completely different numeric scale than
  the rest of the columns.
- **Categorical**: small consecutive integer codes `0..k-1` (`k` if it can be
  inactive), never one-hot (`smac/utils/configspace.py:71-77`). No current
  model treats these values as ordered numbers — RF/GP are told the column
  cardinality via the `types` array and branch/kernel on it accordingly.

None of the current models add *further* scaling to hyperparameter columns —
`SurrogateTransformer.transform_configs` only runs `impute_inactive`
(`smac/model/surrogate_transformer.py:112-134`); the `MinMaxScaler`+PCA in
`SurrogateTransformer` only ever touches **instance features**
(`transform_instance_features`, `:136-161`), never the hyperparameter block.

Why this matters for a PFN specifically, more than it does for RF/GP:
- RF is scale-invariant by construction — none of this matters to it.
- GP fits a per-dimension kernel lengthscale at training time, so it can
  absorb a column that's on the "wrong" scale (e.g. raw ordinal `0..4`
  sitting next to `[0,1]` floats) by learning a large lengthscale for it.
- A pretrained PFN is **frozen** — it has no per-dimension parameter it
  fits to your data at inference time the way a GP kernel does. Whatever
  scale-normalization it does is either (a) a fixed embedding scheme baked in
  at pretraining time, or (b) a data-driven per-column transform it
  recomputes on your batch (this is what TabPFN v2's preprocessing pipeline
  did). Either way, feeding it one column on `[0,1]`, one on `[-1,1]`, and one
  on raw `0..4` integers is a real input to reason about, not a solved
  problem inherited from RF/GP's track record.
- Even if the PFN does its own per-column data-driven renormalization, a BO
  inner loop has *very* few rows (SMAC's `min_trials`/initial design is
  typically 5-20 points before the first model fit) — far below the row
  counts typical tabular benchmarks (and TabPFN's own preprocessing choices)
  were tuned against. A data-driven quantile/power-transform fit on 10 points
  is on shakier ground than ConfigSpace's fixed, sample-independent log/
  min-max encoding. There's a real argument that SMAC's existing
  domain-aware encoding (which already puts a log-scale HP where it "should"
  be) is *more* trustworthy input for the PFN than whatever it would infer
  from 10 raw points itself — but that's a design decision to validate
  empirically, not assume.
- The raw-integer-scale ordinal columns are the one clear inconsistency worth
  fixing regardless: either rescale ordinals into `[0,1]` before handing them
  to the PFN for consistency with the rest of the columns, or confirm
  (empirically, against real v3 behavior) that the PFN's own per-column
  normalization makes this a non-issue.

### A4. y-scale consistency once wired into `SurrogateTransformer`

`SurrogateTransformer` optionally z-score-normalizes `y`
(`normalize_y`, fit in `SurrogateTransformer.fit`, applied in `transform_y`,
`smac/model/surrogate_transformer.py:90-95`, `:197-221`). GP opts in
(`normalize_y=True` by default,
`smac/model/gaussian_process/abstract_gaussian_process.py:46`) and explicitly
un-does it on the way out — `self.transformer.untransform_y(mu, var)` in
`smac/model/gaussian_process/gaussian_process.py:244,259,290`. RF opts out
(scale-invariant, doesn't need it).

Any PFN wrapper needs to make the same explicit choice:
- If `normalize_y=False`: fine, nothing more to do, but then the PFN sees
  raw-cost-scale targets — check whether the library's own target
  discretization/binning behaves reasonably across your scenario's actual
  cost range (could be `[0,1]` accuracy-loss or `1e-3..1e6` runtime — very
  different regimes).
  - If `normalize_y=True`: `_predict` **must** call
  `self.transformer.untransform_y(mean, var)` before returning, exactly like
  GP does — otherwise every downstream consumer (incumbent-cost tracking,
  `ConfigSelector._get_x_best`, logging, validation) silently receives
  normalized-scale numbers instead of real cost. Daphne's `_predict` never
  did this (there was no `transformer` to un-transform through on the old
  branch), so this is a new failure mode that only appears once the model is
  correctly wired into the current `SurrogateTransformer` — worth calling out
  explicitly so it isn't missed in a v3 rewrite.
  - If normalization is on, the acquisition function's `eta` (best observed
  cost, arrives in raw units from the runhistory via `ConfigSelector`) must
  also be run through `model.transformer.transform_y(eta)` before being
  compared against anything the model computes internally in normalized
  space — see A5.

### A5. The acquisition function should not reinvent the model's preprocessing

`RiemannExpectedImprovement._compute` (`smac/acquisition/function/tabpfn_acq_fun.py`
on the old branch) bypasses `model.predict()`/`model.predict_marginalized()`
entirely and hand-rolls its own preprocessing
(`model._x_imputer.transform(X)` → `model._x_pt.transform(...)` →
`model._x_scaler.transform(...)`) before reaching into `model._tabpfn`
directly. Two independent problems with this, beyond the fact that those
three attributes no longer exist on the model at all (already covered in the
earlier review):

1. Every standard acquisition function (`ExpectedImprovement`,
   `smac/acquisition/function/expected_improvement.py:140,168,249`) calls
   `self._model.predict_marginalized(X)`, which routes through
   `model.predict()` → `self.transformer.transform_X(X)`
   (`abstract_model.py:189`) — i.e. **the one, single, fitted preprocessing
   pipeline the model actually trained with**. A PFN-specific EI should call
   through the same path (or at minimum `model.transformer.transform_X(X)`
   directly) rather than maintaining a second, independent preprocessing
   implementation that can silently drift out of sync with `_train`/`_predict`
   — which is exactly what happened here.
2. If the model's target space is normalized (A4) and the acquisition
   function needs the *native* distributional output (logits/bins) rather
   than the collapsed `(mean, var)` from `predict()`, then `eta` needs the
   matching `transform_y` treatment before it's compared against anything in
   that native space — otherwise the "is this candidate better than the
   incumbent" comparison at the heart of EI is comparing two different
   scales.

### A6. No facade / `ConfigSelector` integration exists

None of `smac/facade/*.py` were touched by the tabpfn branch — every facade's
`get_model()`/`get_acquisition_function()`/`get_acquisition_maximizer()`
(abstract, `smac/facade/abstract_facade.py:363-379`) still only ever returns
RF/GP + standard EI. TabPFN is usable only by manually constructing
`HyperparameterOptimizationFacade(..., model=TabPFNModel(...),
acquisition_function=RiemannExpectedImprovement())` by hand, as in the
example. That's an acceptable starting point for evaluation, but note:

- `ConfigSelector._get_x_best()` calls `model.predict_marginalized(X)`
  (`smac/main/config_selector.py`, `_get_x_best`) — this comes for free once
  `build_transformer`/`predict` are implemented correctly, but it's never
  been exercised with a PFN in practice.
- `ConfigSelector.__init__`'s `retrain_after=8` default retrains the model
  from scratch on the *entire* runhistory every 8 proposed configs
  (`smac/main/config_selector.py` docstring + `_check_for_retrain`). RF/GP
  retraining cost grows sub-linearly-ish with history size; a PFN's forward
  pass over the whole context is a different cost profile entirely and was
  never benchmarked against SMAC's retrain cadence — worth explicitly tuning
  `retrain_after`/`retrain_wallclock_ratio` for whatever the real per-call
  cost turns out to be, rather than inheriting RF/GP's defaults.

### A7. `categorical_features_indices` / column-order assumption is implicit and untested

Daphne's model computes categorical column indices once, in `__init__`, from
`enumerate(configspace.values())`, and assumes that ordering exactly matches
the column order of the `X` arrays it's later trained/predicted on. That
happens to be true today (the encoder builds columns in `configspace.values()`
order), but:
- it's never asserted or tested anywhere,
- it only flags `CategoricalHyperparameter` — `OrdinalHyperparameter` and
  `Constant` are excluded, which is *directionally* consistent with
  `get_types()`'s own convention of treating ordinals as numeric
  (`smac/utils/configspace.py:79-85`), but the two pieces of logic
  (`get_types()` and the model's own categorical-index list) are maintained
  completely independently and could silently drift apart.

A v3 rewrite should derive categorical indices from the same source of truth
`get_types()` already uses (or reuse `get_types()`'s output directly) rather
than recomputing the same classification a second, separate way.

### A8. Multi-objective and multi-fidelity paths are architecturally fine but completely unvalidated

`MultiObjectiveModel` wraps arbitrary `AbstractModel` subclasses by
composition — each objective gets its own independent model instance, trained
and queried through the normal `.train()`/`.predict()` interface
(`smac/model/multi_objective_model.py:82-99`). Once a `TabPFNModel`
correctly implements `build_transformer`, this should work for free — but it
has never been run, and multi-fidelity (`ConfigSelector._collect_data`
filtering by budget, `smac/main/config_selector.py`) has never been exercised
with a PFN retraining repeatedly at low budgets either.

### A9. Reproducibility: `seed`/`random_state` plumbing was already broken once

On the v2 branch, `TabPFNModel.__init__` stores `self.random_state = seed`
but `_get_tabpfn()` never actually passes it into the constructed regressor —
so SMAC's `seed=` was silently disconnected from the model's own randomness.
Whatever the v3 API's actual seeding parameter is called, make sure it's
wired through and covered by a reproducibility test (two runs, same seed,
same result) — this class of bug is easy to introduce and easy to miss
because nothing crashes, results just aren't actually reproducible.

### A10. Operational/scalability unknowns, never measured

- No device (CPU/GPU) management is exposed on the model at all. SMAC's
  acquisition-function maximizers (`differential_evolution.py`,
  `local_and_random_search.py`) call the acquisition function on batches of
  candidate points many times per SMBO iteration — for a PFN, each of those
  calls is a transformer forward pass, not a cheap tree lookup or closed-form
  kernel eval. This was never benchmarked on the old branch and directly
  determines whether TabPFN-as-surrogate is practical wall-clock-wise for any
  given `n_trials`.
- No incremental fit — every retrain reprocesses the full context from
  scratch. Combined with A6's retrain cadence, this sets a real, currently
  unknown ceiling on how many trials/retrains are practical before the PFN
  forward-pass cost dominates the actual target-function evaluation cost it's
  supposed to be optimizing around.

### A11. Testing gaps that let the above go unnoticed on the old branch

- `tests/test_model/test_tabpfn.py` exercises the model in isolation but
  never touches the acquisition function — the `AttributeError` in
  `RiemannExpectedImprovement` shipped without any test catching it.
- No test uses a configspace with real `Condition`s (i.e. exercising actual
  `NaN` inactive values end-to-end), despite the shipped example having four
  of them.
- No end-to-end test drives a full `facade.optimize()` loop with
  `TabPFNModel` + a PFN-aware acquisition function through `ConfigSelector`.
- No multi-objective test, no reproducibility (same-seed) test.

A v3 rewrite should add at least: one conditional-configspace test through
the acquisition function (not just the model), one seeded-reproducibility
test, and one short `facade.optimize()` smoke test.

---

## B. Open questions specific to the actual PFN library/API (need re-verification against v3 — not asserted here)

I did not verify these against TabPFN v3 specifically (only v2's API surface
is visible in Daphne's branch, and you've said v2 is not the target) — treat
every item below as "go check the v3 docs/source," not as an established
fact:

- **Missing-value semantics**: does v3 accept `NaN` natively, and if so, is
  its learned notion of "missing" a good semantic match for "structurally
  inactive because a parent condition is false" (A2), or should SMAC still
  impute an explicit sentinel value the way RF/GP do?
- **Built-in preprocessing**: does v3 still auto-apply a per-column
  data-driven transform (quantile/power-transform-style) before its encoder,
  and if so, is it beneficial or redundant/harmful on top of ConfigSpace's
  already-normalized `[0,1]`/log-scale input (A3)? Can/should it be disabled
  in favor of trusting SMAC's own encoding?
- **Categorical handling**: does v3 keep a `categorical_features_indices`-style
  API, and does it still treat those columns as unordered embeddings (matching
  ConfigSpace's `CategoricalHyperparameter` semantics) rather than ordinal
  magnitudes?
- **Seeding**: what is v3's actual reproducibility knob (`random_state` or
  otherwise), and does setting it actually make repeated `.fit()`/`.predict()`
  calls with identical inputs deterministic?
- **Output distribution API**: does v3 expose an equivalent to v2's
  `output_type="full"` logits/`criterion` object, or a different mechanism,
  for building a distribution-aware acquisition function (as opposed to
  collapsing to a Gaussian `(mean, var)` for standard EI)? This determines
  whether a "Riemann EI"-style acquisition function is even the right shape
  for v3, or whether v3's native uncertainty representation calls for a
  different acquisition function design entirely.
- **Model loading / checkpoint parameter**: v2's `model_path=''` (passed
  explicitly in Daphne's code) is suspicious as a sentinel — check what v3
  expects for "use the default pretrained checkpoint" vs. a real path, since
  passing the wrong sentinel could silently force a re-download or break
  offline/CI usage.
- **Context-length / row-count limits and per-call latency**: these directly
  determine whether A6/A10's retrain cadence and BO-scale row counts are
  workable at all with v3 — worth an early, deliberate benchmark rather than
  discovering it mid-integration.

---

## Suggested order of attack

1. Rewrite the model against the current `AbstractModel`/`SurrogateTransformer`
   contract (A1) — nothing else can be evaluated until this compiles and runs.
2. Make an explicit, tested decision on inactive-hyperparameter handling (A2)
   and y-normalization (A4), since the shipped example already needs the
   former on the very first run.
3. Resolve the B-list open questions against the actual v3 API before writing
   the acquisition function — several A-list decisions (A3, A5) depend on
   what v3 actually does internally.
4. Rewrite the acquisition function to route through `model.transformer`
   instead of duplicating preprocessing (A5).
5. Add the missing tests (A11) as you go, not after — the acquisition-function
   gap is exactly how the v2 branch's breakage went unnoticed.
6. Only once the above is solid, consider facade/`ConfigSelector` wiring for
   out-of-the-box selectability (A6) and multi-objective/multi-fidelity
   validation (A8).
