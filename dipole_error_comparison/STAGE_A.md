# Decision 1, Stage A: compact multipole model — numerical results

Date: 25 October 2026. Numerical study only; no hardware data. All results are
for the illustrative Case Study 1 cylinders (moving 9 × 5 mm, 0.12 A m²; fixed
11 × 5.5 mm, 0.145 A m²), assumed uniformly and axially magnetized.

## Files

| File | Content |
|---|---|
| `multipole_model.py` | Axial multipole interaction, orders L = 1, 3, 5, 7; outputs body-frame [b_y, b_z, f_y, f_z] |
| `charge_reference.py` | Exact surface-charge model of both cylinders, Gauss quadrature (reference) |
| `truncation_bound.py` | Explicit truncation bound tau_{L,r}(s), closed form |
| `stageA_compare.py` | Matched comparison and pointwise TFW test |
| `results/stageA_compare.json` | 14 304 admissible poses: 61 × 61 grid, headings 0°, 45°, 90°, 135° |
| `results/stageA_bound.json` | Bound thresholds and coverage |

## Checks

- L = 1 reproduces the Case 6 point-dipole offset (max difference 4.8 × 10⁻⁷ in SI units).
- Closed-form Q₃ and Q₅ agree with direct integration of the end-face charges (Q₅ = m(3L⁴ − 30L²R² + 30R⁴)/48).
- The charge reference converges: refining 8 × 32 × 8 → 16 × 64 × 16 changes outputs by < 10⁻⁶ mT or mN, also at 11.5 mm separation.
- The closed-form bound matches brute-force summation to 12 digits.

## Accuracy against the exact reference

Maximum absolute error over the grid and headings (force in mN, field in mT):

| Separation | Samples | L = 1 f / b | L = 3 f / b | L = 5 f / b | L = 7 f / b | Old point-element ref. 4×16×5, f |
|---|---|---|---|---|---|---|
| 11.3–15 mm | 416 | 1030 / 17.2 | 205 / 2.7 | 100 / 0.89 | 53 / 0.47 | 157 |
| 15–20 mm | 752 | 184 / 4.1 | 25 / 0.41 | 4.6 / 0.05 | 2.3 / 0.02 | 20 |
| 20–30 mm | 2272 | 28 / 0.86 | 2.1 / 0.05 | 0.13 / 0.002 | 0.05 / 0.001 | 2.2 |
| ≥ 30 mm | 10 864 | 2.1 / 0.10 | 0.07 / 0.002 | < 0.01 | < 0.01 | 0.13 |

Pointwise TFW classification disagreements with the reference (Case Study 1 task: b_y = 0, b_z = 20 mT, f_y = 0, f_z in [−0.1, 0.1] N; ±15 A):

| Separation | L = 1 | L = 3 | L = 5 | L = 7 |
|---|---|---|---|---|
| 11.3–15 mm (416) | 53 | 18 | 1 | 0 |
| 15–20 mm (752) | 61 | 7 | 0 | 0 |
| ≥ 20 mm | 0 | 0 | 0 | 0 |

Mean cost per pose: all four multipole orders together 3.0 ms; point-element
reference 4 × 16 × 5: 9.9 ms; charge reference 8 × 32 × 8: 38.6 ms.

Observations:

1. The multipole series converges as predicted; each added order reduces the error, most strongly beyond 15 mm.
2. Close to the convergence radius (11.3–13 mm), partial sums can oscillate: at single poses and single components, L = 3 can be worse than L = 1. Below about 15 mm, no low order is accurate.
3. The point-element quadrature used for the current Table 6 and Fig. 15 (4 × 16 × 5) has errors up to 157 mN below 15 mm. The close-range finite-body numbers in the manuscript must be recomputed with the exact reference.

## Explicit truncation bound

Derivation (charge picture). Each cylinder is end-face charge with total
absolute charge 2m/L_c inside its bounding sphere (radius a). The bihomogeneous
part of degree (l, l') of 1/|d + xi − eta| is bounded by C(l+l', l) a_s^l a_r^l' / s^(l+l'+1),
because the n-th derivative tensor of 1/r has norm n!/s^(n+1) and, for symmetric
multilinear forms on a Hilbert space, multilinear and polynomial norms agree
(Banach). Even-degree parts vanish by the inversion antisymmetry of the charge.
Summing the dropped parts gives a closed form that is finite exactly for
s > a_s + a_r and decreases with s. It bounds the Euclidean norm of the in-plane
field and force vectors, hence every body-frame component.

Coverage on the Stage A grid: no violation for any order (0 of 14 304 poses).
Largest observed error / bound: L = 1: 0.47; L = 3: 0.17; L = 5: 0.033; L = 7: 0.10.

Separation above which the bound is at most a given tolerance:

| Order | force ≤ 10 mN | force ≤ 5 mN | force ≤ 1 mN | field ≤ 1 mT |
|---|---|---|---|---|
| L = 1 | 26.9 mm | 29.9 mm | 38.5 mm | 25.4 mm |
| L = 3 | 21.6 mm | 23.2 mm | 27.6 mm | 20.3 mm |
| L = 5 | 18.9 mm | 19.9 mm | 22.7 mm | 17.9 mm |
| L = 7 | 17.3 mm | 18.0 mm | 20.0 mm | 16.6 mm |

Scope of the bound: uniform axial magnetization, exact geometry and moments,
truncation only. Coefficient, metrology, pose, and coil-model uncertainty remain
in rho_r of Lemma 1.

## Proposed Decision A

- Adopt L = 5 as the operational interaction model. Its error is below 5 mN beyond 15 mm and causes one TFW disagreement in 1168 samples below 20 mm.
- Certify with tau_{5,r} from 18.9 mm (force bound ≤ 10 mN, 10 % of the task half-width). Between 11.3 mm and about 19 mm, the bound is too loose to be useful: there the model-robust TFW stays unresolved unless a reference with its own error bound replaces the truncated model.
- The Case Study 1 planned path keeps at least 22 mm separation, where tau_{5,f} ≈ 1.5 mN, so the path stays within the certified range.
- State the near-contact limit explicitly: below about 19 mm, the method gives model results, not certificates.

## Still open in Stage A

- Recompute Table 6 and the Fig. 15 discrepancy statistics with the exact reference.
- Torque outputs (needed only for tasks with torque components) and the coil-field derivatives that L = 5 needs in G_L.
- Affine enclosures of the L = 5 terms for the region test, and the recomputed 3-D TFWs (Stage C).
