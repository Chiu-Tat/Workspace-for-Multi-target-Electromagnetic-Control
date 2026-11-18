# `hybrid_region`

Reusable library form of the dependency-preserving certified-region method
evaluated in [`../revision_analysis/`](../revision_analysis/).

For a box `Q` of microrobot configurations, the method decides whether *every*
configuration in `Q` admits an admissible coil current for *every* requested
field in the task box:

```
h = c(q) + G(q) i,    q in Q,    i in [i_lower, i_upper]
```

It layers two dependency-preserving certificates on top of the original
Rohn/contractor cascade: a **constructive right-inverse recourse** test that
retains shared affine noise symbols, and a **fixed-normal dual separation**
test.

## Why this package exists

The method previously existed in two disconnected halves: the certificate
mathematics now in `emns/certificates/tight_region.py`, and the
cascade that orders those tests buried inside
`revision_analysis/benchmark_tight_region_check.py` — a one-shot script that
also owns argument parsing, plotting, and JSON export. This package extracts
the method itself so it can be imported, tested, and pointed at a different
problem without editing a benchmark script.

## Quick start

```python
from hybrid_region import RegionProblem, certify_region

problem = RegionProblem(
    initial_box=((-0.05, 0.05), (-0.05, 0.05)),   # metres
    z_offset=0.075,                                # coil-model plane, metres
    task_box=((-0.12, 0.12), (-0.04, 0.04)),       # tesla
    current_bounds=(-15.0, 15.0),                  # amperes
    n_coils=10,
)

report = certify_region(problem, mode="affine_hybrid", eps=0.002)
print(report.unresolved_fraction)     # 0.363037...
print(report.certificate_counts)      # which test closed how many boxes
print(report.summary())               # JSON-serialisable
```

The defaults are exactly the anisotropic Fig. 12 stress case, so
`RegionProblem()` with no arguments reproduces the manuscript's configuration.

To run the full three-way ablation:

```python
from hybrid_region import run_ablation

result = run_ablation(eps=0.002)      # dict, ready for json.dump
print(result["relative_unresolved_reduction"]["affine_to_hybrid"])
```

## The three cascades

Consecutive modes differ in exactly one component, which is what makes the
enclosure and certificate contributions separately attributable.

| Mode | Enclosure | Tests |
|---|---|---|
| `interval_original` | standard interval | Rohn, contractor |
| `affine_original` | affine | Rohn, contractor |
| `affine_hybrid` | affine (identical) | Rohn, **affine right inverse**, **right inverse**, contractor, **dual separation** |

Order within `affine_hybrid` (see `cascade.py`):

```
RobustFeasible                        -> inside  / rohn
affine_constructive_inside_certificate-> inside  / affine_right_inverse
constructive_inside_certificate       -> inside  / right_inverse
Out                                   -> outside / contractor
dual_outside_certificate              -> outside / dual_separation
otherwise                             -> undetermined, bisect
```

The added tests are consulted only after the original ones decline, so
`affine_hybrid` can never reach a different verdict than `affine_original` on a
box the baseline already closed. `test_hybrid_never_weaker_than_affine_baseline`
asserts this.

Note that `right_inverse` (the interval-bound fallback) does not fire on the
Fig. 12 case at either archived resolution — the affine form subsumes it. It is
retained for enclosures that carry no affine forms.

## Module layout

| File | Contents |
|---|---|
| `__init__.py` | public API |
| `_backend.py` | exposes shared backends; **synchronises the coil model** |
| `model.py` | `RegionProblem`, interval/affine actuation-matrix enclosures |
| `certificates.py` | re-export layer over `emns.certificates.tight_region` |
| `cascade.py` | `classify_box` — the three decision cascades |
| `search.py` | `certify_region` — branch-and-bound driver, `RegionReport` |
| `ablation.py` | `run_ablation` — the three-way comparison |
| `test_hybrid_region.py` | self-tests |

Nothing here forks the numerics. The interval and affine arithmetic, magnetic
field model, and certificate proofs are imported from `emns` and remain the
single source of truth.

## The coil-model synchronisation

`_backend.py` performs this assignment at import:

```python
WS_lib_Interval_fast.params_list = WS_lib_Affine.params_list
```

**This is required, not cosmetic.** The two backend modules ship different coil
parameter tables whose z-coordinates differ by exactly 0.085 m, because they
were written against different coordinate origins. Without the assignment, the
interval and affine enclosures describe two physically different machines.

The failure is silent rather than conservative: at `eps=0.01` the
unsynchronised `interval_original` cascade terminates after 21 boxes having
*certified the entire search region as infeasible* (16 contractor proofs,
0% unresolved). The correct synchronised result at the same tolerance leaves
100% unresolved, and at `eps=0.002` leaves 86.108%. A wrong coil model
therefore produces confident proofs about the wrong machine, which is why this
is asserted by `test_coil_models_are_synchronised` rather than left to
convention.

`analysis/affine/ten_coil.py` performs the same assignment at module scope. The
archived
benchmark is correct only because it imports that module — the dependency is
invisible at the call site. This package repeats the assignment explicitly so
it does not rely on an import side effect, and
`test_coil_models_are_synchronised` guards it.

## Tests

```bash
python -m hybrid_region.test_hybrid_region
```

18 tests, runs in about a minute. The file is also `pytest`-compatible
(`pytest hybrid_region/test_hybrid_region.py`) if pytest is installed — it is
not currently present in the bundled virtual environment.

Certificate-level proofs are tested separately in
`tests/test_tight_region_certificates.py`.

## Equivalence with the archived benchmark

`certify_region` reproduces `revision_analysis/tight_region_benchmark.json`
exactly at `eps=0.002` — identical iteration counts, box counts, certificate
counts, and area fractions for all three modes.

| Mode | Unresolved area, 2 mm | 1 mm |
|---|---|---|
| `interval_original` | 86.108% | 69.568% |
| `affine_original` | 59.009% | 46.094% |
| `affine_hybrid` | 36.304% | 13.538% |

## Caveats

All certificates are **one-sided**: `certified=True` is a proof, `certified=False`
means only that the test was inconclusive.

The guarantees require inclusion-preserving enclosures. The archived custom
affine backend is not outward-rounded throughout, so the percentages above are
nominal. The certificate logic is rigorous when supplied with
inclusion-preserving affine forms or interval bounds.
