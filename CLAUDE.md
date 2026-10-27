# Project: quantization_101

> "You can throw away three quarters of every number and barely notice."

An educational **101** portfolio project that builds model quantization **from scratch on
CPU**, for a data-science audience. It proves two things, exactly and by measurement:
1. A model's weights carry far more numeric precision than its behavior needs — mapping fp32 →
   int8 (and often int4) is nearly free, done with a few lines of rounding arithmetic.
2. Quantization does **not** fail gracefully everywhere. A single outlier weight stretches the
   scale and collapses the ordinary weights into a few buckets. Per-channel scales and outlier
   handling are the whole art of modern quantization.

This is the quantization half of QLoRA (see sibling project `lora_101`). The outlier-breaks-the-
scale failure is the same phenomenon as an outlier wrecking a feature scaler (`scaling_arena`);
the "how few bits do you actually need" framing comes from `information_theory_101`.

---

## Identity

- **Format:** GitHub repo — 6 Jupyter notebooks + a `src/` library + a lightweight Streamlit app.
- **Audience:** data scientists/engineers who run quantized models via a flag (`load_in_4bit=True`)
  and want to understand the operation, pick a bit-width, and anticipate what breaks.
- **Runs entirely in a 2-CPU / 8 GB GitHub Codespace. No GPU, ever.**
- **Pedagogical arc** (each notebook builds on the last):
  1. `01_precision_you_dont_need` — precision is reducible (motivation).
  2. `02_quantize_from_scratch` — the operation, built (**the showpiece**).
  3. `03_the_outlier_problem` — where it breaks (**the honest crux**).
  4. `04_granularity_and_schemes` — the fixes, built.
  5. `05_end_to_end_on_a_model` — the real thing, full trade-off.
  6. `06_decision_framework` — the guide.

---

## Stack & constraints

- **Python 3.11+**, managed with **`uv`**. No `pip`, no `conda`, no GPU.
- `torch` (CPU build) + `transformers` — the small real model and its weight matrices.
- `numpy` — **the from-scratch quantizers are written in numpy** so the arithmetic is explicit.
- `onnxruntime` — **optional**, only for the reference int8 path (`[onnx]` extra). Code must
  degrade gracefully / skip when it is absent.
- Viz: `matplotlib` + `plotly`. App: `streamlit`. Config: `pydantic` + `pydantic-settings` + `pyyaml`.
- **Budgets:** each notebook executes in **< 6 minutes** on 2 CPUs; a small fp32 model plus its
  quantized copies must fit in 8 GB.

---

## Repository structure

```
config/settings.yaml          Bit-widths, granularity, clip thresholds, seeds
src/
  config.py                   pydantic-settings loader for config/settings.yaml
  quant/
    absmax.py                 Symmetric absmax quantization (from scratch)
    zeropoint.py              Asymmetric zero-point quantization (from scratch)
    granularity.py            Per-tensor vs per-channel / per-row scales
    outliers.py               Clipping / outlier isolation
    dequant.py                Reconstruct floats from integers + scale
  bits/widths.py              int8 / int4 / lower packing + the size math
  models/
    weights.py                Load a small real model's weight matrices
    small_lm.py               A small HF model for end-to-end quality
  evaluation/
    reconstruction.py         Quantization error vs bit-width
    downstream.py             Quality after quantization: fake_quantize_linear_weights
                              (nn.Linear) + fake_quantize_2d_weights (any 2-D weight,
                              incl. GPT-2 Conv1D; `skip` list) + output_drift
    size_speed.py             Memory & speed vs bit-width
    comparison.py             Arena (QuantConfig) across bit-widths, granularity, schemes
  reference/compare.py        Validate from-scratch against torch's int8 quantizer
  artifacts.py                save_figure() -> PNG, save_json() -> JSON (notebook outputs)
  visualisation.py            Shared plotting helpers
app/streamlit_app.py          4 tabs: quantize-a-matrix, outlier breaker, bit-width sweep, scheme compare
notebooks/01..06.ipynb        Built + executed; each saves figures + a results JSON
outputs/figures/*.png         Every notebook visual
outputs/data/NN_results.json  Every notebook's textual/numerical results (nb 06 reads 01-05)
```

Imports are `from src.quant.absmax import ...` (the project is run from the repo root; it is **not**
installed as a package — `tool.uv.package = false`).

---

## Mental model (the concepts the code must make visible)

- An fp32 float stores ~7 decimal digits. **int8 = 256 buckets, int4 = 16 buckets.** Quantizing =
  find the range, carve it into evenly spaced buckets, round each value to the nearest, store the
  integer index + one scale (and a zero-point for asymmetric).
- **Size math is computed, never asserted by hand:** int8 is **1/4** of fp32, int4 is **1/8**.
- **Dequantization** reconstructs floats from `index * scale` (+ zero-point). The round-trip error
  is the core, clean, deterministic signal — it carries the claims that downstream quality only
  demonstrates.
- **The outlier problem** is the crux: one weight 10× the rest stretches `[min, max]`, so the
  ordinary weights squeeze into 2–3 buckets and lose every distinction. This is a *scale* problem,
  **not** a *too-few-bits* problem. Name it as the same failure as a feature scaler in `scaling_arena`.
- **The fixes:** per-channel / per-row scales (an outlier only harms its own row) + clipping extreme
  values before quantizing. This granularity choice is the single most important lever for quality.
- **The knobs:** bit-width (8 vs 4 vs lower) · granularity (per-tensor vs per-channel) · scheme
  (symmetric absmax vs asymmetric zero-point) · target (weight-only vs activations — activations are
  harder because their outliers are worse, which is why weight-only is the common default).
- **Always measure both** reconstruction error *and* downstream quality: a small reconstruction
  error can still move behavior.

---

## Data

- **Real weight matrices** from **`distilgpt2`** (768-wide, authentic heavy-tailed channels;
  `sshleifer/tiny-gpt2` was too degenerate — hidden dim 2). `AutoModel` names have no
  `transformer.` prefix; pick layers by suffix (`endswith("mlp.c_fc.weight")`).
- **distilgpt2 end-to-end** — real logits/perplexity, so downstream effect is measured, not assumed.
  int8 can quantize *everything*; int4 must **spare the tied embedding/`lm_head`** (int4 embeddings
  are catastrophic) — a real "*what* you quantize matters" lesson (see `fake_quantize_2d_weights(skip=...)`).
- **Synthetic matrices with planted outliers** (`planted_outlier_matrix` returns a truth `mask`) —
  so outlier-collapse-and-recovery is *asserted* against known structure. Tests use a seeded
  `TinyMLP` so they never need the network. All seeded for reproducibility.

## Notebooks & artifacts

- Every notebook saves visuals via `src.artifacts.save_figure(fig, "NN_name")` → `outputs/figures/NN_name.png`
  and results via `save_json(dict, "NN_results")` → `outputs/data/NN_results.json`. Notebook 06
  reads 01-05's JSON, so facts are cited, never re-asserted from memory.
- Notebooks are built from `nbformat` cell lists and executed in place; a bootstrap cell adds the
  repo root to `sys.path` so `from src... import ...` works regardless of run directory.
- `make notebooks` executes all six in place (`--inplace`), ~85s total on 2 CPUs.

---

## Commands (always via the Makefile / `uv`)

| Command | What it does |
|---|---|
| `make setup` | Install `uv` if needed, `uv sync --all-extras`, register the Jupyter kernel |
| `make sync` | `uv sync --all-extras` |
| `make lint` | `format` + `check` + `typecheck` (ruff format, ruff check --fix, ty) |
| `make test` | `uv run pytest` |
| `make test-cov` | pytest with coverage |
| `make notebooks` | Execute every `notebooks/0*.ipynb` in place (360s timeout each, ~85s total) |
| `make run` | Streamlit app on port 8501 |
| `make lab` | JupyterLab on port 8888 |
| `make ci` | `sync` + `lint` + `test` (what CI runs) |
| `make dev` | `lint` + `test` (fast local loop) |

Run Python only through `uv run ...`. Never call bare `python`/`pytest`/`ruff`.

---

## Conventions & guardrails

- **Build from scratch.** The quantizers (absmax, zero-point, granularity, outliers, dequant) are
  written explicitly in numpy — scale and rounding visible. Do **not** reach for a library
  quantizer (`bitsandbytes`, `torch.quantization`, etc.) for the core teaching code; those exist
  only behind `src/reference/` as something to validate *against*.
- **Measure, don't assert prose.** Size ratios and round-trip error are computed from the code; the
  README/notebook claims must match what the functions return.
- **What is asserted vs demonstrated** (keep this split — see §8 of the PRD):
  - *Asserted in tests:* round-trip within the bit-width's resolution; int8 = 1/4 & int4 = 1/8;
    per-tensor collapses under a planted outlier; per-channel recovers it.
  - *Demonstrated in notebooks:* small int8 reconstruction error on real weights; int8 downstream
    quality close to fp32; clipping reduces damage; int4-with-granularity close vs naive degraded.
- **Reproducibility:** seed every synthetic matrix; read seeds/bit-widths/thresholds from
  `config/settings.yaml` via `src/config.py` (pydantic-settings) — do not hard-code them in notebooks.
- **Notebooks** tell the arc above, keep cells cheap (< 6 min total), and save figures to
  `outputs/figures/`. Prefer calling `src/` functions over inlining logic.
- **Tone:** precise and two-sided — "most precision is wasted, **but** outliers will destroy a naive
  quantizer." Avoid both hype ("4-bit is free!") and FUD ("quantization is risky").
- **Lint clean:** ruff rules `E,F,W,I,UP,N,B,A,SIM,PTH` (line-length 99, `E501` ignored); `ty` clean
  on `src/`. `make lint` and `make test` must be green before anything is considered done.
- **Known `ty`/warning caveats (don't "fix" them away):** `TinyMLP.forward` carries
  `# ty: ignore[invalid-mutable-override]` (torch stub friction); use `np.max/np.min` free functions,
  not `.max()/.min()` methods, in `src/` (ty picks the wrong numpy overload otherwise). The reference
  check calls `torch.quantize_per_tensor`, which prints a deprecation warning — expected; it still
  validates the from-scratch int8 path.

---

## Token rules / workflow

- **Caveman mode: ON** (see `.claude/skills/caveman.md` if present) — terse, code-first replies.
- `/compact` at notebook boundaries.
- When in doubt, re-read the PRD at `.llm/PRD_quantization_101.md` — it is the single source of truth
  for scope, structure, success criteria (§8), and the thesis.
