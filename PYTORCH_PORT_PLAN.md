# MaskInterpreter → PyTorch Conversion Plan

Scope agreed with user:
- **Port:** core method + evaluation pipeline (not GUI, not figure scripts, not reference predictor nets).
- **Predictors:** model-agnostic — the port wraps any *frozen `nn.Module`* the user brings. No TF-weight loading, no numerical-parity requirement against the old `.h5`/SavedModel checkpoints.
- **Variants:** all four — 2D & 3D image-to-image, 2D classification, 2D regression.
- **Companion repo:** investigated (see below); reuse its primitives, build the port here.

---

## 0. Key findings that shape the port

### 0.1 What's TF-coupled (needs porting)
- `models/MaskInterpreter.py`, `MaskInterpreterCLF.py`, `MaskInterpreterRegression.py` — `keras.Model` subclasses with custom `train_step`/`compile`/`fit`, `tf.GradientTape`, `tf.random.normal`, heavy `tf.float64` casting.
- `models/UNETO.py` — `get_unet` dynamic 2D/3D U-Net builder (Keras functional), plus a `UNET` training wrapper.
- `utils/metrics.py` — `tf_pearson_corr` (+ segmentation-weighted variant).
- `utils/callbacks.py` — `SaveModelCallback` (Keras callback).
- `utils/utils.py` — `resize_image` (`tf.image.resize`), `predict` (`tf.keras.backend.clear_session`, **OOM-recursion on `batch_size-1`, requires patch-count divisible by batch_size**). The patch helpers `get_weights`/`collect_patchs`/`assemble_image`/`slice_image` are **pure numpy**. ⚠️ **`preprocess_image` is NOT pure numpy** (review A/B): it calls `ImageUtils.imread/image_to_ndarray/get_channel/normalize_std` + `dataset.df.get_item` and pads Z<32 via `np.pad(...,'edge')` with `ceil((32-Z)/2)` both sides — must be rewritten under §7.2.
- `dataset.py` — `DataGen(keras.utils.Sequence)`; ~all numpy + `cell_imaging_utils`, the only TF is the base class + `resize_image`.
- `mg_analyzer.py` — `analyze_th`, `calc_unet_pcc`, `find_noise_scale`: `keras.models.load_model`, `tf.random.normal`, `tf.where`, `tf.cast`, patch inference via `utils.predict`.
- `train.py` / `test.py` — driver scripts.

### 0.2 What needs NO porting (framework-agnostic)
- All of `create_data/` (quilt3 download, aicssegmentation, tiff/csv metadata).
- ~15 of 25 `figures/` scripts (pure numpy/pandas/scipy/matplotlib).
- The numpy patch-assembly helpers `get_weights`/`collect_patchs`/`assemble_image`/`slice_image` in `utils/utils.py` (but NOT `preprocess_image` — see §0.1).

### 0.3 Out of agreed scope (not porting now)
- `gui/gui_logic.py` (grad-cam / guided-backprop / saliency via `tf.custom_gradient`, `GradientTape`, model surgery) + `gui/gui.py` (PySimpleGUI).
- `figures/` scripts that call TF (0,1,2,3,4,7,10,11,12,13, `plot_predictions_from_input_image`) — become trivial once core+eval is ported; revisit later.
- Reference predictor nets: `models/regressor_cellcycle.py` (ResNet18), `models/clf-cifar10.py` (VGG19). User brings their own PyTorch predictors; torchvision has equivalents if reference impls are wanted later.

### 0.4 Companion repo `zaritskylab/Interpretability` (reuse, don't depend on)
- Provides working PyTorch: `MaskGenerator_No_Context.py` / `_CELTIC.py`, `UNETO_CELTIC.py` (`UNet3D`), `metrics.py` (`pearson_corr_torch`, numpy + weighted), torch `SaveModelCallback` + `EarlyStopping`, `dataset_no_context.py` (torch `Dataset`, tifffile), `transforms.py`, config-driven train harness, and a frozen-predictor `Wrapper` pattern.
- **Gaps vs. our scope:** 3D single-cell only (no 2D, no classification, no regression); predictor API hardwired to CELTIC `celtic_wrapper(signal, m)` with a cell-mask multiply; **no `mg_analyzer`-equivalent eval suite** (threshold sweep / mask-size / organelle-context / noise-scale selection / FOV patch inference).
- **Deviations from paper's TF method** (decide deliberately):
  - Generator preproc uses a fixed box filter `F.conv3d(weight=ones)` instead of the TF **learned** `Conv3D(32, relu)` on image & prediction.
  - Loss reweighted: `(1-w)*unet + w*mask + clamp(target-pcc,0)²*10` vs. TF `w_sim*MSE + w_mask*MSE(mask,0) + w_pcc*|target-min(pcc,target)|`.
  - `+1e-8` denominator in pearson; numpy pearson default-masks `a != 1e-4`.
- **License:** repo is `NOASSERTION`; this repo is CC BY-NC 4.0. Both Zaritsky-lab. **Confirm reuse is OK** before copying code in (likely fine given same lab / user affiliation).

### 0.5 Environment reality
- No `torch`, no `tensorflow`, no `cell_imaging_utils` installed in any listed conda env; no pretrained checkpoints on disk.
- Companion repo **ships example single-cell Nuclear-envelope data** (`data/Nuclear-envelope/cells/*.tiff` + metadata CSVs) and pretrained models on Zenodo (DOI 10.5281/zenodo.20522083) — usable to sanity-check the 3D single-cell path end-to-end.

---

## 1. Target design

**New package `mask_interpreter/` (PyTorch), NCHW/NCDHW throughout.** Keep old TF modules in place initially (delete at the end) so behavior can be compared.

```
mask_interpreter/
├── predictors.py        # freeze(module); Predictor protocol; thin wrappers
├── unet.py              # UNet(nn.Module): 2D+3D, dynamic depth (FIRST-AXIS-driven, §1.1), configurable final activation
├── metrics.py           # pearson_corr (torch + numpy, seg-weighting); unbiased=False, eps policy (§2.1)
├── interpreter/
│   ├── base.py          # MaskInterpreterBase: mask→adapt-with-noise→loss; per-variant hooks (§2)
│   ├── image2image.py   # 3D; TWO learned Conv3D(32,relu) on x & pred → concat(64ch) → adaptor; one-sided pcc clamp
│   ├── classification.py# 2D; grad-of-max-prob aug channel → ONE learned Conv2D(32,relu) → adaptor; two-sided pcc
│   └── regression.py    # 2D; grad-of-sum-output aug channel → adaptor DIRECTLY (NO conv preproc); two-sided pcc
├── data/
│   ├── single_cell.py   # torch Dataset (tiff triplets) — adapt companion's DataGen
│   ├── fov.py           # FOV/patch path — port DataGen numpy logic (drop Sequence); keep dilate + predictors kwargs
│   └── transforms.py    # normalize (std/minmax), rot90 augment (NCDHW axes), pad-to-32 (edge, ceil both sides)
├── train.py             # Trainer: plain loop + AMP + checkpoint + early-stop (monitors per-variant `val_stop`)
├── analyze.py           # port of mg_analyzer: analyze_th / calc_unet_pcc / find_noise_scale (tifffile+pandas I/O)
├── config.py            # dataclass/yaml config (replaces global_vars mutation); per-variant defaults (§2.2)
└── __init__.py
```

**Design principles**
- **Model-agnostic predictor:** MaskInterpreter takes any frozen `nn.Module` (`pred(x)->y`). Provide `freeze(m)` = `.eval()` + `requires_grad_(False)`. *Do not* wrap the predictor in `no_grad` during loss — gradients must flow predictor→adapted_image→mask→generator (only predictor **params** are frozen). **Compute the reference prediction `pred(x)` under `torch.no_grad()`/`.detach()`** — TF computes `unet_target` *outside* the GradientTape (`MaskInterpreter.py:96`), so it's a constant; matching this saves memory and matches semantics. Note `.eval()` disables the predictor's BatchNorm updates/dropout (correct, deterministic) but won't bit-match a TF predictor that had dropout active under `fit`; document per predictor.
- **Per-variant generator front-ends are NOT uniform** (review A/B/D — verified): image2image applies two learned `Conv3D(32,relu)` (to image and to prediction) then concats; classification applies one learned `Conv2D(32,relu)` to the grad-augmented input; regression applies **no** conv preproc (adaptor consumes the 2-channel augmented input directly). `base.py` must expose this as a per-variant hook, not a shared front-end.
- **NCHW/NCDHW native.** Any NHWC input converted at the data boundary. `concat` axis `-1`→`1`; loss spatial reductions remapped. rot90 augment must use NCDHW spatial axes (TF used `np.rot90(axes=(2,3))` on NDHWC).
- **2D image2image is a NEW variant, not a port** (review A#5): the TF `MaskInterpreter` is 3D-only (`Conv3D` hardcoded, `patch_size=(32,128,128,1)`). A 2D image2image path (requested in scope) has no TF reference — build it by analogy to the 3D form (Conv2D preproc, 2D adaptor) and validate on synthetic data; flag any behavioral choice that has no TF precedent.
- **float32** default (not TF's float64) — faster on GPU; acceptable since no parity requirement. Make dtype configurable.
- **Config object** replaces mutating `global_vars` at import time. `global_vars.py` is removed once `config.py` covers its fields (or kept as a thin deprecation shim if unported scripts still import it — see §7.6).

### 1.1 U-Net dynamic depth — FIRST-AXIS-driven (blocking correction)
`get_unet` (`UNETO.py:26,61`) drives depth **solely by the first spatial axis** (`while layer_dim[0] > 4`) while the strided `k=4,s=2` conv halves **all** axes uniformly. So for `(32,128,128)` it produces **3 levels**, bottleneck `(4,16,16)` — NOT `(4,4,4)`. The port's loop must key on `dim[0]` only (uniform anisotropic halving), or skip topology / channel counts diverge. Do **not** use the companion `UNet3D` as the structural template — it is fixed-depth (3 downsamples), hardcodes `reduce_channels 256→128` (only valid for `base_filters=16`), has a dead `UpBottleneck.up`, a hardcoded sigmoid head, and attaches skips post-downsample (TF attaches pre-downsample). Port `get_unet` directly; borrow only the padding arithmetic (stride-1 k3→pad1; strided k4/s2→pad1; transposed k4/s2/pad1→×2).

---

## 2. Faithful loss — per-variant (the TF variants differ; do NOT unify)

Common skeleton, but **three details differ per variant** (review A#1, B#4/#6, D#1/#3/#4 — all verified against source):
```
adapted = mask*x + noise*(1-mask),  noise ~ N(0, noise_scale)
sim   = MSE(pred(x).detach(), pred(adapted))              # "similarity"
total = w_sim*sim + w_mask*size + w_pcc*pcc
```

| | image2image (3D) | classification (2D) | regression (2D) |
|---|---|---|---|
| generator front-end | 2× learned `Conv3D(32,relu)` on x & pred → concat | 1× learned `Conv2D(32,relu)` on aug input | **none** (adaptor on 2-ch aug input) |
| input augmentation | concat with predictor output | grad of **max(pred)** wrt x, min-max norm, `.detach()`, as extra channel | grad of **sum(pred)** wrt x, same treatment |
| PCC domain | spatial pixels, optional seg-weight | class-score vectors | output vectors |
| `size` term | `MSE(mask, 0)` | `L1(mask)` | `L1(mask)` |
| `pcc` term | **one-sided:** `|target − min(pcc, target)|` (stops at target) | **two-sided:** `|target − pcc|` (penalizes over-correlation too) | **two-sided:** `|target − pcc|` |
| `importance_mask_size` metric | `mean(mask)` | `1 − mean(mask)` | `1 − mean(mask)` |
| `stop` metric (drives checkpoint/early-stop via `val_stop`) | `pcc_loss + MSE(mask,0)` | `pcc_loss + mean(mask)` | `pcc_loss + mean(mask)` |

- **grad-aug takes the raw predictor output** — `max`/`sum` of the bare output, **no added softmax** (`CLF.py:215`, `Regression.py:249`). Compute via `torch.autograd.grad`, per-sample min-max normalize, `.detach()`, concat.
- **Companion's reweighted loss** (`(1-w)*sim + w*mask + 10*relu(target−pcc)²`) only ever applied to 3D image2image; offer it **only** for image2image, under a **separate namespaced config** (its `mask_loss_weight` is a convex-mix that also sets `w_sim=1−w`, and its pcc weight is hardcoded ×10 and squared — it must NOT share `w_sim/w_mask/w_pcc` names with the faithful form, or a mode switch silently reinterprets weights; review B#8).

### 2.1 PCC numerics (review B#3, A#10, D metric-parity)
> **Update (post-implementation):** PCC now delegates to **`cubic.metrics.pcc`**
> (github.com/alxndrkalinin/cubic) for the numpy eval path, and the torch training loss
> mirrors cubic's formula in a differentiable form (`cubic.pcc` returns a float and
> detaches torch inputs). cubic uses standard Pearson *r* (ratio-invariant to the
> Bessel correction below), returns **NaN** on zero-variance (denom `< 1e-12`) and
> **clips** to `[-1,1]`. Weighted PCC uses a boolean mask (`seg > 0`) with an
> empty→full-array fallback. The population-std discussion below is retained for context.

- `tf_pearson_corr` uses **population** std (`reduce_std`, ÷N) and cov (`reduce_mean`, ÷N); `torch.std` defaults to Bessel ÷(N-1). Port must use `torch.std(..., unbiased=False)` (or manual ÷N) or every PCC is scaled ≈(N-1)/N — non-negligible for small-N classification prob vectors and weighted-PCC subsets.
- TF has **no eps** in the denominator → constant input yields NaN. Decide policy explicitly (guard with small eps vs. preserve NaN) and document; do **not** silently inherit companion's `+1e-8` and its `a!=1e-4` default mask (both absent from this repo's TF).
- **Weighted PCC** (`weighted_pcc=True`, image2image only): faithful semantics = plain PCC computed over only the pixels where the seg weight ∈ {1.0, 255.0} (accept both 0/1 and 0/255 encodings), all others weighted 0; empty-region → fall back to full arrays (TF `tf.cond`). Reimplement from `metrics.py:22-36`; do **not** copy the repo's numpy weighted path (`metrics.py:41` has a latent `tf.logical_and`-inside-`np.where` bug) or the companion's formula.

### 2.2 Per-variant default hyperparameters (review D#3)
No single "faithful default." Seed configs from the TF sources:
- image2image: `noise_scale=1.5, target_loss_weight=10.0, mask_loss_weight=1.0, similarity_loss_weight=1.0, pcc_target=0.9` (`MaskInterpreter.py:62`, `train.py`).
- classification: `noise_scale=0.5, target_loss_weight=1.75, pcc_target=0.95` (`CLF.py:__main__`).
- regression: `noise_scale=0.5, target_loss_weight=2.5, pcc_target=0.95` (`Regression.py:__main__`).

---

## 3. Phased work breakdown

### Phase 0 — Env + skeleton (0.5 day)
- New conda/uv env: `torch`(+CUDA), `torchvision`, numpy, pandas, scipy, scikit-image, scikit-learn, matplotlib, tifffile, tqdm, pyyaml. **No `cell_imaging_utils`** (replaced by tifffile+pandas per §7.2). New `pyproject.toml`; drop `tensorflow*`.
- Create `mask_interpreter/` skeleton. Pull companion example NE data + a small predictor for smoke-testing.

### Phase 1 — Core primitives + image2image (2–3 days)
- `metrics.pearson_corr` (torch + numpy, seg-weighted) — reimplement from TF `metrics.py`, **not** companion: population std (`unbiased=False`), explicit eps policy, weighted-selection semantics per §2.1. No `a!=1e-4` default mask.
- `unet.UNet` — port `get_unet` directly (NOT companion `UNet3D`; see §1.1): **first-axis-driven** dynamic depth (`while dim[0] > 4`, uniform anisotropic halving), `Conv{2,3}d`, optional BatchNorm (default on, per `gv.batch_norm`), skip connections attached **pre-downsample** (TF order), transpose-conv upsampling, **configurable** final activation (not hardcoded sigmoid). `'same'` mapping: stride-1 k3→pad1; strided k4/s2→pad1; transposed k4/s2/pad1→×2. Validate output shapes on 2D + 3D + anisotropic `(32,128,128)` toy inputs.
- `predictors.freeze` + protocol; `interpreter/base.py` (adapt-with-noise, per-variant hooks for front-end/pcc-clamp/metrics, `__call__`→mask).
- `interpreter/image2image.py` (two learned `Conv3D(32)` on x & pred, concat, adaptor; one-sided pcc clamp; `mean(mask)` size metric).
- `train.Trainer` (plain loop, AMP optional, checkpoint via ported `SaveModelCallback` — drop its silent `except:save_weights` fallback; early stopping on per-variant `val_stop`).
- **Test:** toy predictor where only a known sub-region matters → assert mask concentrates there, loss ↓, mask∈[0,1], predictor params unchanged after training; `pearson_corr` matches `scipy.stats.pearsonr` (population) within tol; UNet output shape == input spatial shape for 2D/3D/anisotropic patches.

### Phase 2 — Classification + regression variants (1–2 days)
- `_augment_input_with_gradients`: `torch.autograd.grad` of (**max(pred)** for clf | **sum(pred)** for reg) w.r.t. input on the **raw predictor output** (no softmax), per-sample min-max normalize (guard `max−min+eps`), `.detach()`, concat as extra channel. Verify no second-order graph leak.
- `interpreter/classification.py`: **one** learned `Conv2D(32,relu)` before the adaptor; two-sided pcc; `1−mean(mask)` size metric; `stop = pcc_loss + mean(mask)`.
- `interpreter/regression.py`: **no** conv preproc (adaptor consumes 2-ch aug input directly); two-sided pcc; same metric conventions as clf.
- **Test:** synthetic classifier/regressor with a known informative pixel block; mask localizes it; confirm clf has the Conv2D preproc param and regression does not; grad-aug channel ∈ [0,1] and detached (no grad to predictor from it).

### Phase 3 — Data + transforms (2 days)
- `data/single_cell.py`: adapt companion torch `Dataset` (tiff signal/target/mask triplets + transforms). Clean, low-risk.
- `data/fov.py`: port `DataGen` FOV/patch sampling logic (numpy: patch sampling, augment rot90, SSD cache, norm) — drop `keras.utils.Sequence`, expose as `torch.utils.data.Dataset` for `DataLoader`. **In-scope kwargs to keep** (review A#9): `dilate` (weighted-PCC training sets `gv.target="structure_seg", dilate=True`) and `predictors` (2-channel in-silico concat, mirrored in `test.py`). **Explicitly out of scope** (other model types): `pairs`/`masking_pair`/`for_clf`/`crop_edge`/`mask`/`input_as_y`/`output_as_x` — document exclusion, don't silently drop. rot90 must use NCDHW spatial axes (TF used `np.rot90(axes=(2,3))` on NDHWC).
- `data/transforms.py`: `normalize` (std/minmax), replace `tf.image.resize` (`utils.resize_image`, NEAREST) with `F.interpolate(mode="nearest")`; rot90 augment; pad-to-32 replicating TF's `ceil((32-Z)/2)` both-sides edge pad exactly (over-shoots odd deficits — match for patch-count parity).
- **Test:** dataset yields correct shapes/dtypes/normalization on companion NE example data; rot90 lands on the right axes in NCDHW; pad-to-32 reproduces TF output shape for odd Z.

### Phase 4 — Eval pipeline (`analyze.py`) (3–4 days — bumped for I/O + fidelity fixes)
- Port `analyze_th` (modes `agg`/`loo`/`mask`/`regular`), `calc_unet_pcc`, `find_noise_scale`:
  - `keras.models.load_model`→ instantiate + `load_state_dict`; `tf.where/cast`→`torch.where`.
  - **⚠ Blocking — noise drawn ONCE per image, reused across all thresholds** (`mg_analyzer.py:512` is outside the `for th` loop at :519, reused at :559). Draw `noise = torch.randn(...)*σ` once per image; do NOT redraw inside the threshold loop, or every value in `pcc_results.csv`/`context_results.csv` shifts.
  - **Two models per image** (review A#6): `analyze_th` runs the frozen predictor (`model.unet`) AND the interpreter (`model(x)`→mask). `analyze.py` must expose both `interpreter.predictor` and `interpreter(x)->mask`.
  - **`predict` port** (`utils.py:132`): batched inference on device under `torch.no_grad()`. Original reshapes to `(-1, batch_size, *patch)` → **requires patch-count divisible by batch_size**; handle non-divisible counts (pad/last-partial batch) explicitly. Replace the OOM path cleanly: catch `torch.cuda.OutOfMemoryError`, `empty_cache()`, retry smaller, **raise a real exception** on exhaustion (TF had bare `except:` + `raise("str")` → `TypeError`; do not preserve).
  - **`preprocess_image` rewrite** (§0.1 / §7.2): tifffile + numpy replacement that reproduces the 6-channel FOV contract — columns `[input_col, target_col, "structure_seg", "channel_dna", "channel_membrane", "membrane_seg"]`, per-channel `normalize=[T,T,F,T,T,F]`, 6-tuple return `(input,target,target_seg,nuc,mem,mem_seg)`, and the Z<32 edge-pad (`ceil((32-Z)/2)` both sides). `DatasetMetadataSCV` CSV access → `pandas`.
  - **Remove the broad `except`→`-1` sentinel** (`mg_analyzer.py:589`, review B#5): raise on failure; handle the specific zero-intersection case (`c = 1/(intersection/size)` divides by zero when `intersection==0`) explicitly rather than poisoning the row with `-1`.
  - Reuse pure-numpy `collect_patchs`/`assemble_image`/`get_weights`.
  - I/O via `tifffile` + `pandas`. **Clean output names** (§7.4): per-image tiffs `input_/target_/unet_prediction_/mask_/noisy_*`; CSVs `pcc_results.csv` / `mask_size_results.csv` / `context_results.csv`.
- **Test:** run `analyze_th(mode="regular")` and `mode="agg"` on a tiny FOV with a trained toy interpreter; assert CSVs + tiffs produced, PCC∈[-1,1]; assert a fixed seed → identical CSV across a re-run (noise-reuse determinism); assert a zero-intersection input raises rather than writing `-1`.

### Phase 5 — Validation, cleanup, docs (1–2 days)
- End-to-end on companion NE single-cell example data (+ their Zenodo pretrained 3D predictor if licensing OK) as a realistic sanity check for the 3D image2image path.
- Integration tests wired into a suite; run full suite.
- Update `README.md` / `md/*.md` to the PyTorch API; rewrite `example.ipynb`.
- **Cleanup without breaking retained code** (review C#1/#2): the plan retains `figures/` and `gui/`, which import `dataset`/`mg_analyzer`/`utils`/`global_vars`. Do **not** delete those TF modules outright — either (a) keep them until the dependent scripts are ported/removed, or (b) move them under a `legacy_tf/` subpackage and update the retained imports. `global_vars.py` is replaced by `config.py`; keep a thin shim if unported scripts still import it. Grep for dangling imports after any deletion and fix or gate them.
- Remove `tensorflow-addons`, `cell-imaging-utils`, and the `numpy==1.20.3` pin from `pyproject.toml` (there is **no** direct `tensorflow` entry to remove — review A#12); reassess `patchify`/`quilt3`/`opencv-python` if their only consumers were dropped.

**Rough total: ~10–14 focused days**, front-loaded on Phases 1 & 4 (highest risk).

---

## 3a. Execution model (parallelism, verification, iteration)

**Phase ordering is mostly sequential** (each phase depends on the prior): Phase 1 must land before 2/3/4. Within phases, these units are **independent and can be dispatched to parallel subagents** (no shared files, no data dependency):
- Phase 1: `metrics.py` ∥ `unet.py` (independent modules; `interpreter/base.py` + `image2image.py` + `train.py` depend on both, so they run after the barrier).
- Phase 2: `classification.py` ∥ `regression.py` (independent; both depend only on Phase-1 base).
- Phase 3: `single_cell.py` ∥ `fov.py` ∥ `transforms.py` (independent; `fov` uses `transforms`, so order those two or duplicate the small helper).
- Phase 4: the three analyzer functions (`analyze_th`, `calc_unet_pcc`, `find_noise_scale`) share `preprocess_image`/`predict` — build those shared pieces first (barrier), then the three can be finished in parallel.

**Verification is on me, never "trust the subagent."** For every unit above, the concrete check is:
- `metrics.py` → run its unit test vs `scipy.stats.pearsonr` (population std) + weighted-selection cases; read the diff.
- `unet.py` → run the shape test on 2D/3D/anisotropic `(32,128,128)` patches; assert output spatial == input and skip-concat channel counts; read the module.
- `interpreter/*` → run the toy-overfit test (mask localizes the known-informative region, loss ↓, mask∈[0,1], predictor params unchanged); confirm per-variant front-end/pcc-clamp/metrics by reading the class.
- `data/*` → run the shape/normalization/rot90-axis/pad-to-32 tests on companion NE data.
- `analyze.py` → run `analyze_th` in `regular`+`agg`, check CSV/tiff outputs, seed-determinism (noise-reuse), and zero-intersection raises.
- Whole suite green before Phase 5 cleanup; re-grep for dangling imports after any module move/delete.

**Iterate autonomously to green.** Execution runs each phase, runs its tests, and **keeps fixing and re-running until the phase's success criteria pass — no pausing to ask between steps.** Stop and ask the user ONLY for: (1) a genuine ambiguity with no source precedent (e.g., a design choice for the new 2D-image2image variant that materially changes results), (2) an irreversible/destructive action (deleting the TF modules in Phase 5), or (3) a true blocker (missing pretrained model / data / GPU that no reasonable default resolves). The one open licensing question (§7.5) is the only pre-known ask.

---

## 4. Testing / validation strategy
- Per user convention: integration tests import & run the real code; for each bug/feature write the failing test first.
- No local TF baseline ⇒ validate against **algorithmic expectations** (toy problems with known-important regions) + **companion example data** for the 3D path.
- Numeric checks: `pearson_corr` vs `scipy.stats.pearsonr`; mask range; frozen-predictor invariance; gradient-flow assertions.

## 5. Dependencies / environment
- Add: torch(+CUDA), torchvision, tifffile, pyyaml (+ existing numpy/pandas/scipy/scikit-image/scikit-learn/matplotlib/tqdm).
- Remove: `tensorflow-addons`, `cell_imaging_utils` (→ `tifffile`+`pandas`), and the `numpy==1.20.3` pin (incompatible with modern torch — bump).
- `imagecodecs` optional (tifffile compression); keep if the TIFFs need it.

## 6. Risks
- **U-Net depth / `'same'`-padding / transpose-conv shape mismatches** (2D & 3D, anisotropic) — main architectural risk; first-axis-driven depth (§1.1); unit-test output shapes early on `(32,128,128)`.
- **Per-variant fidelity drift** — the three variants differ in front-end conv, pcc clamp (one- vs two-sided), and `stop`/`mask_size` metrics (§2). A "unify the variants" shortcut silently changes clf/reg training and checkpoint selection. Encode differences as explicit per-variant hooks + tests.
- **PCC numeric divergence** — Bessel correction (`unbiased=False`), eps/NaN policy, weighted-selection semantics (§2.1). Affects both the training `pcc_target` term and eval CSVs.
- **Noise-reuse in `analyze_th`** — must draw once per image, reuse across thresholds (§Phase 4); a per-threshold redraw silently changes every CSV value.
- **Gradient-flow correctness** — freeze predictor *params* while keeping the graph for input-gradients; detach the *reference* prediction; grad-aug channel detached. Easy to get wrong with `no_grad`/`eval`.
- **FOV `DataGen`/`analyze_th`/`preprocess_image` are large & intricate** (SSD cache, threading, 6-channel FOV contract, patch bookkeeping, `predict` divisibility) — port incrementally, test on small inputs.
- **Silent-failure inheritance** — the TF `analyze_th` broad-except→`-1` and `predict` bare-except/`raise(str)` must be replaced with real exceptions, not preserved (project convention: raise, no silent fallback).
- **dtype divergence** (float64→float32) may slightly shift PCC — acceptable per scope; document.

## 7. Decisions — RESOLVED
1. **Loss form:** ✅ Implement **both**, select via config; **faithful-TF is the default** (matches paper + eval).
2. **FOV I/O:** ✅ **Replace `cell_imaging_utils` with `tifffile` + `pandas`.** Drop the dependency; rewrite the analyzer's TIFF/CSV I/O (`ImageUtils.imread/imsave/image_to_ndarray/get_channel/normalize*` → `tifffile` + numpy; `DatasetMetadataSCV` → `pandas`). Larger `analyze.py` port, budget +0.5–1 day in Phase 4.
3. **Trainer:** ✅ **Plain PyTorch loop** (AMP optional) + ported `SaveModelCallback`/`EarlyStopping`. No Lightning.
4. **Fix typos:** ✅ **Use clean public names** (`min_percentage`, `max_percentage`, `delete_cache`, `similarity_loss_weight`, `pcc_results.csv`, `mask_size_results.csv`, `context_results.csv`). Downstream figure scripts that read these CSVs get matching edits when/if they're ported.

### Still open
5. **Companion code reuse** given its `NOASSERTION` license — confirm OK to copy primitives in (same lab; likely fine). If not, reimplement from the TF originals + this plan (no blocker, slightly more work in Phase 1).
6. **`global_vars.py` disposition** — replaced by `config.py`; decide keep-as-shim vs remove based on how many retained (unported) `figures/`/`gui/` scripts still import it (§Phase 5).

---

## 8. Review log (stress-tested against the codebase)

Four fresh-context review passes (correctness/completeness, fidelity/edge-cases, merge-compat, companion cross-reference) ran against the real source. Changes folded in:

**Blocking (fixed):**
- Noise is drawn **once per image and reused across thresholds** in `analyze_th` (`mg_analyzer.py:512` outside the `:519` loop) → Phase 4 + §6.
- U-Net depth is **first-spatial-axis-driven with uniform anisotropic halving**, not "halve until ≤4"; `(32,128,128)`→3 levels, bottleneck `(4,16,16)` → §1.1, Phase 1. Companion `UNet3D` rejected as the structural template (fixed-depth, hardcoded `reduce_channels`, dead `UpBottleneck.up`, sigmoid-only head, post-downsample skips).

**Important (fixed):**
- PCC loss is **not shared**: image2image one-sided clamp vs clf/reg two-sided (§2 table).
- Generator front-end differs per variant: 2×Conv3D vs 1×Conv2D vs none (§1, §2, Phase 2).
- `preprocess_image` is cell_imaging_utils/DataGen-coupled (not numpy) with a 6-channel FOV contract + Z<32 edge-pad → §0.1, Phase 4.
- `analyze_th` runs **two** models; `predict` needs divisible batch counts + clean OOM handling → Phase 4.
- Per-variant `stop`/`importance_mask_size` metrics drive checkpointing → §2 table.
- PCC numerics: `unbiased=False`, eps/NaN policy, weighted-selection semantics → §2.1.
- Loss-`both` weight-name collision (companion is a convex mix, ×10², hardcoded) → §2, namespaced config.
- Broad-except→`-1` and `raise(str)` replaced with real exceptions → Phase 4, §6.

**Nitpicks (fixed):** grad-aug on raw output (no softmax); detach reference prediction; per-variant default hyperparameters (§2.2); rot90 NCDHW axes + exact pad-to-32; keep `dilate`/`predictors` DataGen kwargs, exclude the rest explicitly; state predictor is trained externally (`UNET` wrapper/`test.py` UNET branch not ported); `pyproject` has no direct `tensorflow` entry; Phase-5 delete would leave dangling imports in retained `figures/`/`gui/` → move to `legacy_tf/` or defer deletion; `global_vars.py` disposition (§7.6); drop `SaveModelCallback`'s silent `except:save_weights`.

**Validated as correct (no change):** analyze_th modes `agg/loo/mask/regular`; image2image `concat(Conv3D(32)(x),Conv3D(32)(pred))`; clf grad-of-max-prob, reg grad-of-sum-output; companion coverage gaps (3D-only, no clf/reg, no analyzer) accurate; model-agnostic interface is the right call (companion hardwires `celtic_wrapper(signal,m)` + box-filter preproc); gradient-flow design (freeze params, keep graph) sound; merge surface clean (single quiescent `main`, greenfield package).
