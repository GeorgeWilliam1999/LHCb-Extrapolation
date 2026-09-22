This is the first of three method pages. Protocol 1 is where the particles come from and what a reference track is; protocol 2 derives the discrete-time construction and shows why it needs no labels; protocol 3 is the network itself. Protocols 4 to 6 (the loss, the training protocol and the gates) and 7 to 10 (the grid, the stopping rule, the comparators and the scoring) are the next two pages.
**A note on units, once.** $`t_x`$ and $`t_y`$ are slopes, $`dx/dz`$ and $`dy/dz`$, built from a simulated hit's exit minus entry displacement divided by its own $`\Delta z`$; no arctangent is applied anywhere in the pipeline. Every label reading "mrad" in the figures and in the underlying result files is therefore a slope difference multiplied by $`10^3`$, which this write-up writes as $`\times 10^{-3}`$ (slope). For the accepted tracks, pseudorapidity between 2 and 5 and so $`|t_x|`$ below about 0.27, that number reads as milliradians to within 7 per cent at the edge of the acceptance and within 1 per cent for most tracks. Positions are in micrometres unless the text says millimetres.
# Protocol 1: the tracks
1. **The sample.** The training population is harvested from the official simulated sample: TestFileDB entry `expected_2024_minbias_xdigi`, production 00212966, 200 events, magnet up, conditions tag `sim-20231017-vc-mu100`. These are centrally produced events, used instead of self-generated ones on George's instruction of 21 July 2026, so that a configuration error of ours cannot be silently baked into the physics.
2. **The selection.** The forward cross-magnet particles are selected by the previous study's cut cascade, unchanged:
<table fit-page-width="true" header-row="true">
<tr>
<td>cut</td>
<td>rows in</td>
<td>removed</td>
<td>rows out</td>
<td>particles out</td>
</tr>
<tr>
<td>cross-magnet rows in the harvested table, both directions</td>
<td>41,398</td>
<td>0</td>
<td>41,398</td>
<td>21,932</td>
</tr>
<tr>
<td>the pre-magnet plane is an Upstream Tracker plane</td>
<td>41,398</td>
<td>2,476</td>
<td>38,922</td>
<td>20,655</td>
</tr>
<tr>
<td>both directions present for the particle</td>
<td>38,922</td>
<td>2,388</td>
<td>36,534</td>
<td>18,267</td>
</tr>
<tr>
<td>pseudorapidity between 2 and 5</td>
<td>36,534</td>
<td>1,240</td>
<td>35,294</td>
<td>17,647</td>
</tr>
<tr>
<td>momentum between 1 and 200 GeV</td>
<td>35,294</td>
<td>0</td>
<td>35,294</td>
<td>17,647</td>
</tr>
<tr>
<td>not an electron</td>
<td>35,294</td>
<td>4,780</td>
<td>30,514</td>
<td>15,257</td>
</tr>
<tr>
<td>forward rows only, one per particle</td>
<td>30,514</td>
<td>15,257</td>
<td>15,257</td>
<td>15,257</td>
</tr>
<tr>
<td>both frozen planes within 60 mm of the particle's own</td>
<td>15,257</td>
<td>775</td>
<td>14,482</td>
<td>14,482</td>
</tr>
<tr>
<td>fiducial: the reference trajectory stays inside the field map</td>
<td>14,482</td>
<td>0</td>
<td>14,482</td>
<td>14,482</td>
</tr>
</table>
1. **The start state.** Each particle's real state on its own last Upstream Tracker plane, which lies between 2,593 and 2,663 mm, is moved to $`z_0`$ with RK6 at 0.1 mm. That state is what every chain starts from.
2. **The reference track.** RK6 marches from there across the crossing and its state is stored on the 257 planes $`z_0 + k\,L/256`$, $`k = 0 \ldots 256`$. Because 256 is divisible by 2, 64, 128 and 256, every plane of every step grid in this study lies on that one grid.
3. **The real end state.** The reference is carried on from $`z_1`$ to the particle's own first fibre-tracker plane, between 7,819 and 7,833 mm, and the particle's real state there is stored beside it. The difference is the material floor.
4. **The splits,** by particle, from the training set's own assignment: **training 11,567, validation 1,463, test 1,452**. Unlike the previous study the training split is not capped, because here one network trains on the whole crossing rather than one network per step.
5. **The gate.** The validation and test particles must be the previous study's particles, in the same order, with bit-identical start states, real states, planes and momenta, and the reference states on the 129 shared planes must agree. They do, to at most $`1.5 \times 10^{-4}`$ µm in position and $`6.3 \times 10^{-8}`$ in slope $`\times 10^3`$, with $`q/p`$ identical. RK6 restarts its 0.1 mm steps at every stored plane and this study stores twice as many planes, so differences of that size are expected.
The equation the reference integrates, and which the loss of protocol 4 enforces, is the Lorentz force with $`z`$ as the independent variable:
$$
\frac{dx}{dz} = t_x, \qquad \frac{dy}{dz} = t_y, \qquad \frac{d(q/p)}{dz} = 0,
$$
$$
\frac{dt_x}{dz} = \kappa\,\frac{q}{p}\,\mathcal{N}\,\big[t_x t_y B_x - (1 + t_x^2) B_y + t_y B_z\big], \qquad \frac{dt_y}{dz} = \kappa\,\frac{q}{p}\,\mathcal{N}\,\big[(1 + t_y^2) B_x - t_x t_y B_y - t_x B_z\big],
$$
where $`\mathcal{N} = \sqrt{1 + t_x^2 + t_y^2}`$ is the path length per unit $`z`$, $`(B_x, B_y, B_z)`$ is the measured field at the particle's position, and $`q/p`$ is in the convention of the LHCb first-level trigger software, $`q/p = 0.299792458\,q/p\,[\mathrm{GeV}^{-1}]`$, with $`\kappa = 10^{-3}`$ converting metres to the millimetres in which $`z`$ is measured, so that a 10 GeV particle has $`|q/p| = 0.03`$. The fifth equation is the one the scheme exploits: $`q/p`$ is a constant of the motion, an input carried through unchanged, never an output. The field is the v8r1 map of the LHCb magnet, magnet-up polarity to match the sample, read by trilinear interpolation on a 100 mm grid from a file whose md5 is `9e49ddc4313b589f273540e7e0bb513b`.
![The crossing. Top: the magnitude of the measured field on the beam axis against z, from the last Upstream Tracker plane at 2,648.2 mm to the first fibre-tracker plane at 7,826.0 mm. Bottom: the four step grids of this study, 2, 64, 128 and 256 steps of 2,588.9, 80.9, 40.5 and 20.2 mm.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/Block_E_single_network_chain/E0_Track_dataset/figures/tracks_overview.png)
The momentum bands used throughout, and how many of the 1,452 test tracks fall in each: 1-2 GeV, 5 tracks; 2-3 GeV, 159; 3-5 GeV, 341; 5-7 GeV, 233; 7-10 GeV, 197; 10-15 GeV, 208; 15-20 GeV, 109; 20-30 GeV, 115; 30-50 GeV, 55; 50-100 GeV, 24; 100-200 GeV, 6. Coarsely, 500 tracks below 5 GeV, 430 between 5 and 10 GeV, 383 between 10 and 25 GeV and 134 above 25 GeV.
# Protocol 2: the discrete-time construction, and why it needs no labels
A Runge-Kutta method with $`q`$ **stages** advances a state $`S`$ from $`z_{\rm s}`$ to $`z_{\rm s} + \Delta z`$ through $`q`$ intermediate **stage states** $`Y_j`$, each living on the plane $`z_{\rm s} + c_j \Delta z`$:
$$
Y_j = S + \Delta z \sum_{k=1}^{q} a_{jk}\, f(Y_k,\, z_{\rm s} + c_k \Delta z), \quad j = 1 \ldots q, \qquad S_{\rm end} = S + \Delta z \sum_{j=1}^{q} b_j\, f(Y_j,\, z_{\rm s} + c_j \Delta z),
$$
where $`f`$ is the right-hand side of protocol 1 and the triple $`(c, A, b)`$ is the method's *tableau*. When $`A`$ is a full matrix the stage equations are **implicit**: every $`Y_j`$ depends on every other, and a classical solver has to iterate them to convergence, which is expensive.
The **Gauss-Legendre** family puts the nodes $`c_j`$ at the roots of the shifted Legendre polynomial of degree $`q`$ and chooses $`A`$ and $`b`$ so that the scheme is exactly the polynomial of degree $`q`$ that satisfies the differential equation at those $`q`$ points. That is what **collocation** means: the approximation is a polynomial made to obey the equation at chosen points. The method then has order $`2q`$, so its truncation error falls as $`\Delta z^{2q+1}`$ per step, and it is stable at any step length. The tableaux are built for every $`q`$ and verified on each call against the row-sum, quadrature, collocation and symplecticity conditions to $`10^{-15}`$.
**A worked example at **$`q = 2`$**.** The two nodes are $`c = (0.211324865405,\ 0.788675134595)`$, that is $`\tfrac{1}{2} \mp \tfrac{\sqrt{3}}{6}`$; the matrix and weights are
$$
A = \begin{pmatrix} 1/4 & 1/4 - \sqrt{3}/6 \\ 1/4 + \sqrt{3}/6 & 1/4 \end{pmatrix} = \begin{pmatrix} 0.25 & -0.038675134595 \\ 0.538675134595 & 0.25 \end{pmatrix}, \qquad b = (1/2,\ 1/2).
$$
So a two-stage step of 80.9 mm starting at $`z_{\rm s}`$ asks for the state on the planes $`z_{\rm s} + 17.1`$ mm and $`z_{\rm s} + 63.8`$ mm and for the end state at $`z_{\rm s} + 80.9`$ mm: three states, coupled by the two lines above, which a solver would have to iterate.
**The construction replaces the solver with a network.** Given the start state $`S`$, a network emits all of $`Y_1 \ldots Y_q`$ and $`S_{\rm end}`$ at once, $`4(q+1)`$ numbers. The stage equations are then rearranged so that each of the $`q+1`$ outputs **reconstructs the input**:
$$
\hat S_j = Y_j - \Delta z \sum_{k} a_{jk}\, f(Y_k, z_k), \quad j = 1 \ldots q, \qquad \hat S_{q+1} = S_{\rm end} - \Delta z \sum_{j} b_j\, f(Y_j, z_j),
$$
and the training objective is the mismatch between every reconstruction and the actual input. **No label appears anywhere.** The only ingredients are the input state, the tableau, and the equation of motion evaluated at the network's own proposed positions. If the objective were zero the outputs would be exactly the stage states and end state of the collocation scheme, which is why the exact scheme is the ceiling in every table, and why feeding the exactly solved states to the objective is a test the machinery has to pass (gate 2).
# Protocol 3: the network
1. **Inputs, six numbers.** The state $`(x, y, t_x, t_y, q/p)`$, each component divided by its spread over the round-1 training states (the standard deviation, one fixed number per component for the whole run; at 64 steps these are 446.3 mm, 365.8 mm, 0.1655, 0.06948 and 0.06259), plus the $`z`$ of the plane the step starts on, mapped linearly onto the interval from minus one to one across the start planes $`z_0`$ to $`z_1 - \Delta z`$.
2. **Why the sixth input.** The same state at $`z = 2.65`$ m and at $`z = 4.7`$ m crosses about 0.21 T and about 1.05 T over the next step, so the correct output differs. Knowing where it is is what lets one network serve every step. George settled this on 16 September 2026, choosing the start plane's $`z`$ over the alternative of feeding the network field values sampled along the step.
3. **Outputs, as a deviation from the straight line.** Across a 40 mm step the magnet's correction to a straight line is of order $`10^{-2}`$ mm while the state itself moves by tens of millimetres, so a last layer asked for the absolute state is being asked for six or seven significant figures. Instead the raw output $`r_j`$ is turned into a state by
$$
Y_j = \mathrm{straight}_j + \mathrm{scale} \odot r_j, \qquad \mathrm{straight}_j = \big(x + t_x (z_j - z_{\rm s}),\ y + t_y (z_j - z_{\rm s}),\ t_x,\ t_y\big),
$$
with $`\odot`$ the component-wise product and the scale computed from the input state and the field map alone, by a 16-point midpoint rule along the straight line through the input.
1. **The scale in **$`x`$** and **$`t_x`$**.** To first order the slope change over a step is $`\kappa |q/p| \int |B|\,dz`$ and the position change is that integrated again, so $`\mathrm{slope}_x = \kappa\,|q/p| \int |B|\,dz`$ and $`\mathrm{pos}_x = \mathrm{slope}_x \cdot |\Delta z| / 2`$, floored at $`10^{-12}`$ and $`10^{-9}`$ mm, floors that never bind.
2. **A separate scale in **$`y`$** and **$`t_y`$**, new in this study.** The dominant field component is $`B_y`$, so the bend is mainly in $`x`$ and the same scale is far too coarse for $`y`$. The $`y`$ scale bounds the $`y`$ rate of the equation of motion term by term, so that it cannot vanish through cancellation:
$$
I_y = \int \sqrt{1 + t_x^2 + t_y^2}\,\Big[(1 + t_y^2)|B_x| + |t_x t_y B_y| + |t_x B_z|\Big]\,dz, \qquad \mathrm{slope}_y = \max\!\big(\kappa |q/p| I_y,\ 10^{-3}\,\mathrm{slope}_x\big),
$$
with $`\mathrm{pos}_y = \mathrm{slope}_y \cdot |\Delta z| / 2`$. The floor at one thousandth of the $`x`$ scale is active for 7.5 to 10.7 per cent of states. Under the single scale of the previous study the true $`y`$ deviation divided by its scale sat at 0.004, about 240 times smaller than the quantity it was meant to describe; under the new scale it sits at 0.50 to 0.67 (gate 3).
1. **Size.** Two hidden layers of 128 units with hyperbolic-tangent activations, double precision, one seed. The parameter count grows with $`q`$ only through the last layer: 18,956 at $`q = 2`$, 19,988 at $`q = 4`$, 22,052 at $`q = 8`$ and 26,180 at $`q = 16`$.
2. **Chaining.** The routine `carry` applies the **same weight tensor** $`N`$ times: each step's end state, with $`q/p`$ copied through unchanged, is the next step's input, and only the start-plane input advances by $`\Delta z`$. Each run folder holds exactly one weight file. This was checked directly on 20 September 2026: loading the single weight file of the 64-step, two-stage run and applying it 64 times by hand to the 1,452 test tracks reproduces the stored chain states with a maximum difference of exactly zero, and $`q/p`$ is unchanged along the chain.
