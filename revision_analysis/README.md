# Reproducible revision analysis

This folder contains the analyses added for the IJRR-26-7175 working revision. None of the source CSV files in the two code projects are modified.

## Quantitative audit

Run with the standard library only:

```bash
python3 revision_analysis/quantitative_validation.py
```

Outputs:

- `quantitative_validation.json`: complete numerical results and provenance.
- `quantitative_metrics.tex`: generated table/macros. Copy this file beside the
  manuscript when the archived metrics are regenerated.

The script reports full and objectively defined active intervals. It also exposes the manual crop used by the prior Case 6 plotting script as a separate audit item. A percentage of successful samples is never presented as a repeated-trial success probability.

## Affine-versus-interval figure

This script reuses the scientific Python environment bundled with the workspace code:

```bash
"/mnt/e/CUHK/Research/Reconfigurable surgical robot document/Workspace_eMNS-main/Workspace_eMNS-main/.venv/Scripts/python.exe" \
  revision_analysis/regenerate_aa_ia_figure.py
```

Outputs:

- `AA IA comparison revised.pdf` (vector figure)
- `AA IA comparison revised.png` (raster preview)
- `aa_ia_figure_metrics.json` (box counts, iterations, and runtime)

The plot uses `model-in`, `model-out`, and `unresolved` because the archived custom affine arithmetic is not outward-rounded throughout. The mathematical certificates in the manuscript require an inclusion-preserving backend.

## Algebraic regression checks

`quantitative_validation.py` also checks:

- equivalence of the full symmetric trace-free gradient force equation and its five-component matrix form;
- the support function of a Cartesian product of Euclidean balls;
- dyadic resolution depths and monotonicity of the archived certified-inner percentages;
- the correct $2^d$ row-sign-system count in the interval strong-feasibility test.

## Dependency-preserving region certificates

Run the matched 2-D ten-coil benchmark with:

```bash
python3 revision_analysis/benchmark_tight_region_check.py
```

Outputs:

- `tight_region_benchmark.json`: area fractions, box counts, certificate counts,
  iterations, runtime, and the complete benchmark protocol;
- `tight_region_benchmark.pdf`: vector figure;
- `tight_region_benchmark.png`: raster figure.

The production implementation is `tight_region_certificates.py`, with focused
tests under `tests/test_tight_region_certificates.py`. The hybrid
cascade retains common affine noise symbols in a constructive right-inverse
inside certificate and jointly optimizes redundant currents and uncertainty
reserve. A fixed-normal support-function test supplies a complementary outside
certificate.

This script explicitly reconstructs the anisotropic stress case now used in
Fig. 12: $B_x\in[-0.12,0.12]$ T and $B_y\in[-0.04,0.04]$ T. At 2-mm
resolution, the standard-interval/original cascade leaves 86.108% of the planar
search area unresolved. Replacing only the enclosure by affine arithmetic
leaves 59.009%. Holding that affine enclosure fixed and adding the recourse and
dual-separation certificates leaves 36.304%, a further 38.5% relative
reduction, while certifying 23.145% as model-out. These values remain nominal
because the archived custom affine backend is not outward-rounded throughout.
The certificate logic is rigorous when supplied with inclusion-preserving
affine forms or interval bounds.

The companion `tight_region_benchmark_1mm.json` repeats the complete three-way
ablation at 1-mm resolution. The affine/original cascade leaves 46.094%
unresolved, whereas the recourse/dual cascade leaves 13.538% unresolved (a
70.6% relative reduction) and classifies 44.135% model-in and 42.328%
model-out. Both JSON files include pointwise regression checks for returned
model-in and model-out terminal boxes; these finite checks are not substituted
for the continuous certificates.
