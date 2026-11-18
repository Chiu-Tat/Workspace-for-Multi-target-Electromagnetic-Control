# Case 7: two aligned magnetic microspheres in a branching vessel

For two separate vessels with close approaches, see [VESSEL_VARIANTS.md](VESSEL_VARIANTS.md).
The field/force formulation below also applies there, with each sphere confined
to its assigned lumen and with separate flow fields.
For tapered, connected side branches and secondary branch targets, see
[BRANCHED_NETWORKS.md](BRANCHED_NETWORKS.md); each sphere is confined to its
assigned arterial or venous network.

This case defines a geometry and a control formulation, not a demonstrated
controller. The task is to route two spheres from a common inlet to different
daughter vessels. Their mechanical orientation is not a task variable. Their
magnetic moments are assumed to equilibrate along the **local external field**.
Motion direction means the direction of translational velocity, including flow.

## 1. Anatomical model and coordinates

The carotid bifurcation gives a common carotid artery (CCA) inlet and internal
(ICA) and external (ECA) carotid outlets. Real carotid geometry is variable;
centerlines, curvature and lumen cross sections can be obtained from imaging
[1,2]. This example is an idealized phantom inspired by that anatomy. None of
its numerical dimensions are asserted to be patient measurements or fitted
population means.

`geometry.json` gives four Bezier control points per branch, radii, a bulb term,
and rigid registration into the EMNS coordinate frame. Input lengths are mm;
the Python geometry and magnetic APIs use metres. The default bifurcation
origin is (0,0,0.085) m with identity rotation. This places the phantom near the
existing examples; it does **not** establish that the whole vessel is actuable.

| Branch | Local endpoints (mm) | Base radius, start → end (mm) |
|---|---|---|
| CCA | (0,-28,0) → (0,0,0) | 3.2 → 3.2 |
| ICA | (0,0,0) → (-10,30,4) | 3.2 → 2.3 |
| ECA | (0,0,0) → (17,24,-3) | 2.4 → 1.8 |

The ICA has an additional Gaussian bulb of amplitude 0.7 mm near s=0.22.
The branches leave the common junction in distinct directions; their overlap
forms the connected lumen, including the junction, without an internal wall.

For branch e and 0≤s≤1, with control points P_e,k,

\[
c_e(s)=t_0+R\sum_{k=0}^3 {3\choose k}(1-s)^{3-k}s^k P_{e,k},
\qquad
r_e(s)=(1-s)r_{e,0}+sr_{e,1}
       +A_e\exp[-((s-s_{b,e})/w_e)^2].
\]

s is **not arc length**: dℓ/ds=||c'_e(s)||. The exact mathematical lumen is
the union of swept balls,

\[
\Omega=\bigcup_e\bigcup_{s\in[0,1]}
\{x:\|x-c_e(s)\|\le r_e(s)\}.
\]

This explicitly relates x,y,z by an inequality, which is appropriate for a
3D volume. A single equality such as z=f(x,y) describes a surface, not a lumen.
For any regular centerline point, let (t,n,b) be an orthonormal frame with
t=c'/||c'||. Convenient interior cross-section coordinates are

\[
p_e(s,\rho,\theta)=c_e(s)+\rho r_e(s)
 [n_e(s)\cos\theta+b_e(s)\sin\theta],\quad 0\le\rho\le1.
\]

These disk coordinates give an interior subset and are not a unique global
coordinate system at the bifurcation. The sphere center is free to move across
the cross section; using only c(s) is a path-planning reduction, not a physical
constraint. The frame implementation chooses a stable reference axis locally;
use parallel transport for globally smooth frame-dependent trajectories.

For a sphere radius a and wall margin δ, the physical center domain is the
erosion Ω⊖B_(a+δ). The implemented sampled clearance is

\[
h_N(p)=\max_{e,k}\{r_e(s_k)-\|p-c_e(s_k)\|\}.
\]

h_N≥a+δ is a **sufficient** interior certificate: the complete sphere and margin
fit inside one generating ball. It can be conservative in the junction.
h_N is not exact signed wall distance. h_N<0 alone does not prove that a point
is outside the continuous lumen. Sampling error and mesh extraction error
must not be confused with a continuous geometric certificate.

The exported mesh is the zero level set of h_N on a 0.65-mm grid using 201
centerline samples per branch. It has rounded closed endpoint caps for viewing;
these are not physiological closures. A future CFD model must cut inlet/outlet
planes, label boundary patches, smooth/validate the junction and refine its mesh.
No blood-flow solution is supplied here.

### Replacing the phantom with a scan

Segment a CTA/MRA/3D-ultrasound lumen, retain its surface, extract branching
centerlines and radius information (for example with VMTK [4]), and register
the surface to the coil frame using measured landmarks. Fit piecewise splines
or Bezier segments to the centerlines and radius samples. Preserve the branch
graph and endpoint correspondence. Validate fit and registration error against
the original surface; circular/swept-ball reconstructions can lose noncircular
cross sections and bulb detail. For final collision constraints, use the
original segmented lumen with a validated distance/containment computation.
`geometry.json` currently accepts the analytic three-branch model; an automatic
DICOM segmentation/import pipeline is not implemented.

## 2. The ten-coil field and aligned-moment force

Let I∈R^10 denote the shared coil currents, and let H(p)∈R^(3×10) contain
the field per ampere of each coil. In the initial dilute approximation,

\[
B_i=H(p_i)I,\qquad
G_i=\nabla B(p_i),\quad [G_i]_{ab}=\partial B_a/\partial p_b.
\]

`magnetics.py` reuses `params_list` from `emns.workspace.fast`. For each
existing coil dipole moment per ampere q and displacement r=p-r_coil,

\[
B/I=10^{-7}\left(3r(q\cdot r)/\|r\|^5-q/\|r\|^3\right),
\]
\[
\frac{\partial (B_a/I)}{\partial p_b}=10^{-7}
\left[\frac{3(\delta_{ab}q\cdot r+r_aq_b+q_ar_b)}{\|r\|^5}
-\frac{15r_ar_b(q\cdot r)}{\|r\|^7}\right].
\]

The new module differentiates this field directly and checks it by finite
differences. It does not call the legacy selective-gradient routine: its
`By_dy` expression currently has a single 3*y*m_y term where this derivative
has 6*y*m_y. Existing case implementations have not been changed by this case.

In a source-free region, G is symmetric and trace-free. For a dipole m,
F=Gᵀm. The alignment assumption is

\[
m_i=\mathcal M_i(\|B_i\|)\widehat B_i,
\qquad F_i=\mathcal M_i(\|B_i\|)G_i^T\widehat B_i
=\mathcal M_i(\|B_i\|)\nabla\|B_i\|.
\]

Thus a field direction is not a commanded motion direction. A spatially
uniform field produces alignment but no magnetic translation. The field
gradient, the moment law and the flow determine translation.

Two possible material models must be kept distinct [3]:

* **Constant-magnitude aligned moment:** M_i=m_s,i. Then
  F_i=m_s,i G_iᵀB_i/||B_i||. Require a nonzero field and alignment relaxation
  much faster than the control/flow timescale. A freely rotating permanent
  magnetic sphere can be approximated this way if that assumption holds.
* **Linear induced moment:** m_i=κ_i B_i, where κ_i is measured effective
  polarizability in A m²/T (including particle/medium contrast and the sphere's
  demagnetizing response). Then F_i=κ_i G_iᵀB_i=κ_i/2 ∇||B_i||². This model is
  appropriate only below saturation. Do not assign a bulk susceptibility
  directly without the corresponding constitutive conversion.

More generally the equilibrated magnetic potential is
U_i=-∫_0^||B_i|| M_i(b)db. For an induced dipole it is -κ_i||B_i||²/2;
differentiating -m(B)·B as though m were fixed would incorrectly double the
induced force. There is no orientation target or independent torque command.
Near zero field a constant-moment direction is undefined. A physical bias-field
threshold must be chosen from relaxation/noise data, not from the numerical
1-nT guard in the helper function.

### Mutual interaction

If interaction is retained, "local external field" excludes the particle's
own field but includes the other sphere:

\[
B_i^{ext}=H_iI+D(p_i-p_j)m_j,\quad
D(r)=\frac{\mu_0}{4\pi\|r\|^3}(3\hat r\hat r^T-I_3),
\quad m_i=\mathcal M_i(\|B_i^{ext}\|)\widehat B_i^{ext}.
\]

This is a coupled algebraic alignment problem. Use
F_i=(∇B_i^ext)ᵀm_i, taking the spatial derivative with the source moment m_j
held fixed; do not differentiate a re-equilibrated m_j as an extra force term.
For linear induction the six moment components satisfy a coupled 6×6 linear
system at fixed positions. Check nonsingularity and physical validity before
using it. Nonlinear saturation/alignment can require a stable equilibrium
solve. Interaction is not a current-independent affine offset here because
the moments themselves depend on current.

The supplied force primitives are **coil-only**. Dropping interaction requires
checking its field and force against the external field and force budget over
the intended separation range; enforce ||p1-p2||≥a1+a2+δ12 and reject close
approaches where the point-dipole/dilute model fails. Field-induced chaining is
an additional reason to include separation in the future task constraints.

## 3. Translational dynamics and blood flow

For small particle Reynolds number and rapid inertial relaxation, start with

\[
\dot p_i=u(p_i,t)+\frac{F_i^{mag}+F_i^{gb}+F_i^{other}}{\zeta_i},
\quad \zeta_i=6\pi\eta a_i,\quad
F_i^{gb}=(\rho_{p,i}-\rho_f)\frac{4\pi a_i^3}{3}g.
\]

Here u is blood velocity, not particle velocity, and g is a vector in the
registered EMNS frame. The basic drag model follows creeping-flow
magnetophoresis [3]. Validate Re_p=2ρ_f a_i||ṗ_i-u||/η≪1 and
τ_p=2ρ_p a_i²/(9η) small relative to actuation times. Wall effects, red-cell
interactions, shear-dependent viscosity and particle-particle hydrodynamics
can invalidate the scalar drag approximation; then use a calibrated mobility
matrix and disturbance bounds. Small particle Re does not imply creeping
flow for the whole carotid artery.

Away from the junction, a provisional circular-tube profile is
u=2Q_e(t)/(πr_e²)[1-(r_perp/r_e)²]t_e, with Q_CCA=Q_ICA+Q_ECA. This is only a
branchwise approximation: at a curved bifurcation, obtain a consistent 3D
pulsatile flow from CFD or measurements. Do not blend overlapping branch
profiles and claim incompressibility. The example deliberately leaves viscosity,
flow, density and magnetization uncalibrated rather than inventing control results.

Given a requested velocity v_i*, the required magnetic force is
F_i*=ζ_i(v_i*-u_i)-F_i^gb-F_i^other. If only direction d_i is specified, use
v_i*=v_i d_i with v_i≥v_min>0 optimized or bounded: otherwise zero motion can
trivially satisfy a direction constraint. A tolerance cone can be expressed as
|| (I_3-d_id_iᵀ)v_i^actual ||≤tan(θ_max)d_iᵀv_i^actual and
d_iᵀv_i^actual≥v_min. Blood advection must be included when assessing routing
authority; useful lateral steering need not imply upstream swimming capability.

## 4. Simultaneous current allocation and feasibility

For the induced, coil-only model define D_i,a=∂H(p_i)/∂p_a. Then

\[
F_{i,a}=I^TQ_{i,a}I,\quad
Q_{i,a}=\frac{\kappa_i}{2}(H_i^TD_{i,a}+D_{i,a}^TH_i).
\]

The supplied `induced_force_matrices` implements these matrices. The six-force
map f(I)=[F1;F2] is quadratic, not the fixed-orientation linear map used by
earlier cases. For either aligned law, f(-I)=f(I): simply reversing all currents
does not reverse the forces. For identical nearby spheres, their field and
gradient maps approach each other, reducing independent routing authority.

A future allocator can minimize

\[
\min_{I,v_1,v_2}\sum_i\|W_i(F_i(I)-F_i^*)\|^2
 +\lambda\|I\|^2+\lambda_\Delta\|I-I_{prev}\|^2
\]

subject to measured current, thermal power and slew limits; any required bias
field bound; velocity cones/speed bounds; wall and inter-sphere clearance; and
predicted motion constraints. The ±15-A setting used in existing examples can
be an initial comparison bound, but hardware ratings and total thermal limits
must be verified. A local SQP/sequential-convex step is a candidate method;
check its solution in the original nonlinear equations and report residuals.
Local optimization failure is not a proof of infeasibility.

For induced moments the local force Jacobian has row 2(Q_i,a I_0)ᵀ. Full row
rank six is necessary for local unrestricted six-force correction at a bias
current, but is not sufficient for global reachability or constrained control.
At I_0=0 it vanishes. Current saturation, shared fields and poor conditioning
can still prevent the requested pair of directions even with ten coils.
For centerline-reduced tasks, assess the Jacobian of the actual velocity
objectives and lateral constraints, rather than counting coils versus particles.

Define a vessel-restricted task-feasible set at time t as

\[
\mathcal W(t)=\{(p_1,p_2)\in\Omega_{a_1,\delta}\times\Omega_{a_2,\delta}:
\text{separation holds and }\exists I\in\mathcal I,
\ \dot p_i(I,t)\in\mathcal V_i\ (i=1,2)\}.
\]

For uncertain flow/material/registration ξ, distinguish adjustable recourse
(∀ξ ∃I(ξ)) from a single robust command (∃I ∀ξ); they answer different control
questions. The attainable force set is generally not a zonotope. Existing
linear affine-hull certificates cannot be applied unchanged. A validated future
approach needs nonlinear enclosures or quadratic relaxations with correct
one-sided guarantees. A local Jacobian approximation is not a global certificate.

## 5. What is ready, and what comes next

Ready: parameterized connected vessel; world registration; interior point and
conservative clearance queries; visual mesh and centerline export; two aligned
force laws using the ten-coil parameters; checks of geometry and magnetic
derivatives. No trajectories, flow solve, current optimization, time integration
or hardware experiment have been performed.

Before simulation, select/calibrate sphere radius and M(B), blood/phantom fluid
properties and flow boundary conditions, current/thermal limits, field alignment
threshold, tracking noise and clearance margins. The displayed 50-µm particle
radii are illustrative placeholders. First assess pointwise force/velocity
authority and close-pair interaction; then plan separate outlet routes and
implement feedback. This will establish whether this anatomical-scale phantom
is suitable for the available coil setup or should be resized for a bench study.

## Sources

1. [Geometry of the human common carotid artery. A vessel cast study of 86 specimens](https://pubmed.ncbi.nlm.nih.gov/12389998/).
2. [Three-dimensional geometry of the human carotid artery](https://pubmed.ncbi.nlm.nih.gov/22757506/).
3. Jin et al., *Differences in magnetically induced motion of diamagnetic,
   paramagnetic, and superparamagnetic microparticles detected by Cell Tracking
   Velocimetry*, Analyst 133, 1767–1775 (2008), DOI: 10.1039/b802113a.
   [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC2801409/).
4. Vascular Modelling Toolkit, [Computing Centerlines](https://vmtk.github.io/tutorials/Centerlines.html).

The geometry choices, multi-particle allocation formulation and repository
integration above are proposed here; the sources do not validate this ten-coil
configuration for this vascular task.
