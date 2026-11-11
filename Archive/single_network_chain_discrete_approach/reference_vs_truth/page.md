<callout icon="🧭">
	This page answers a question asked on 22 September 2026: how close is the reference we score our networks against to what the simulation says actually happened? It assumes no prior knowledge. Every number on it was written from a results file in the repository, and the file is named beside it. **The figures are hosted from the public GitHub repository and will stay blank until George commits and pushes this folder**, which is untracked today. Trust is **Provisional** until George confirms it.
</callout>
<table_of_contents/>

# 1. Introduction

## 1.1 What this page is about

LHCb is a forward spectrometer at the LHC. A charged particle made in a proton collision leaves hits in the vertex detector, then in the four planes of the Upstream Tracker, then crosses a dipole magnet of about one tesla over five metres, and finally leaves hits in the twelve layers of the scintillating-fibre tracker. The magnet bends the particle by an amount inversely proportional to its momentum, and that bend is how LHCb measures momentum.

Reconstructing a track means fitting a trajectory through those hits. The fit works with a **track state** on a plane of constant $`z`$: five numbers $`(x, y, t_x, t_y, q/p)`$, the two transverse positions in millimetres, the two slopes $`t_x = dx/dz`$ and $`t_y = dy/dz`$ (dimensionless, not angles), and the signed inverse momentum. Carrying a state from one plane to another through the field is the job of the **track extrapolator**, and it is the most expensive routine in the fit. Our programme asks whether a small neural network can replace it across the magnet, trained **without labels**: not shown examples of where particles went, but required to satisfy the equations of an implicit Runge-Kutta scheme.

Everywhere else in this programme we have scored those networks against a sixth-order Runge-Kutta integration of the equation of motion, which we call the **reference**. On 22 September George asked the obvious and overdue question: the reference is a solution of an equation, but is the equation what the simulated particle actually did? This page measures the answer on all 14,482 magnet crossings in our set, takes the gap apart into its physical pieces, says which pieces we can remove and which are a floor no deterministic reference can cross, and says what follows for how the networks should be scored.

The short version, which the rest of the page establishes: the networks reproduce the reference to a median of 17 to 96 micrometres depending on the momentum band, while the reference itself misses the simulated truth by a median of 1,736 micrometres over the whole set. None of that 1,736 micrometres is integration error. It is made of a momentum defect we put into the data ourselves and can remove, plus energy loss and multiple scattering, which the equation was never given.

## 1.2 A roadmap

Section 2 states the four questions. Section 3 defines the three objects being compared, walks the data pipeline link by link and says which physical effects enter where, then derives the two tests that separate a bending systematic from random scattering. Section 4 gives the measurements. Section 5 says what is fixable, what is a floor, and what we would like George to decide.

This study lives in the repository folder `single_network_chain_discrete_approach/reference_vs_truth/`, and it reads results produced by the neighbouring folders `Block_E_single_network_chain/` and `Block_F_reweighted_loss/`; those folder names appear here and in the provenance only, never as names for the networks themselves, which we name by their step count, stage count and step length.

## 1.3 The three objects

Three different states exist on the first fibre-tracker plane for one and the same particle, and the whole page is about the differences between them.

**(1) The Geant4-true state.** Geant4 is the simulation toolkit that transports each particle through a model of the detector, including all its material. Every time a particle passes through a sensitive volume, Geant4 records a **Monte Carlo hit**: the point where the particle entered the sensor, the point where it left, the energy it deposited, and the identity of the particle. We build a state from that hit by taking the midpoint of the entry and exit points for $`(x, y, z)`$, and the entry-to-exit displacement divided by its own $`\Delta z`$ for the slopes $`(t_x, t_y)`$. The inverse momentum attached to the state comes not from the hit but from the particle record, which is the momentum the particle had **when it was produced**. This matters a great deal later.

What is in this state: everything Geant4 did to the particle up to that hit, including multiple scattering, ionisation energy loss, and any interaction it survived. What is not in it: any detector effect whatsoever. There is no digitisation, no measurement resolution, no pattern recognition. Reconstructed track states are not used anywhere in this study; every state called "true" here is a Monte Carlo hit.

**(2) The field-only reference.** We take the particle's real state on its last Upstream Tracker plane, carry it to the fixed plane $`z_0 = 2{,}648.2`$ mm, and then integrate the LHCb equation of motion across the magnet to $`z_1 = 7{,}826.0`$ mm and on to the particle's own first fibre-tracker plane. The equation, written in full, is

$$
\frac{dx}{dz} = t_x, \qquad \frac{dy}{dz} = t_y, \qquad \frac{d(q/p)}{dz} = 0,
$$

$$
\frac{dt_x}{dz} = \kappa\,\frac{q}{p}\,\mathcal{N}\left[t_x t_y B_x - (1 + t_x^2) B_y + t_y B_z\right],
$$

$$
\frac{dt_y}{dz} = \kappa\,\frac{q}{p}\,\mathcal{N}\left[(1 + t_y^2) B_x - t_x t_y B_y - t_x B_z\right],
$$

where $`\mathcal{N} = \sqrt{1 + t_x^2 + t_y^2}`$ is the path length per unit of $`z`$; $`(B_x, B_y, B_z)`$ is the magnetic field read from the LHCb field map, version v8r1, upward polarity; $`\kappa = 10^{-3}`$ and $`q/p = 0.299792458\,q/p_{\text{GeV}}`$ are the unit conventions used by Allen, LHCb's first-level trigger. The integrator is Butcher's seven-stage explicit Runge-Kutta method of order six, at a fixed step of 0.1 mm, in double precision throughout. We call this the **reference** or the **field-only reference** below.

Notice the third equation: $`d(q/p)/dz = 0`$. The momentum is held constant along the whole crossing. The equation contains the magnetic field and nothing else. No energy loss, no scattering, no material of any kind.

**(3) The network endpoint.** A small network, two hidden layers of 128 units, applied $`N`$ times in succession from the same real Upstream Tracker state. It was trained label-free on the residual of the equation above, evaluated at $`q`$ Gauss-Legendre collocation points inside each step. A **collocation point** is simply a place inside the step where we insist the equation holds; **Gauss-Legendre** says where those places are put so that the resulting scheme is as accurate as possible for the number of points used. Because the training loss is built only from that equation, the network can learn exactly what the equation says and no more. Whatever the equation omits, the network omits too, by construction.

## 1.4 Why we did not see this earlier

In July 2026 we wrote the label-generation metadata for this programme, and one field in it reads `"material_effects": "excluded by design (master-extrapolator responsibility)"`. That was a deliberate decision: LHCb's production extrapolator handles material in a separate layer, so our labels model the field alone. We then gated the labels against the simulation, and the gate looked reassuring. Gate G2 compared each label with the particle's next real hit over 100,660 short plane-to-plane legs and found a median disagreement of 11.9 micrometres, falling to 3.8 micrometres for particles above 5 GeV.

Those legs are short. The magnet crossing is not. The difference is the **lever arm**: a small error in the slope at one point grows linearly with the distance the particle then travels. A slope error of $`10^{-4}`$ (one tenth of a milliradian) displaces the track by 10 micrometres over a 10 cm gap between two tracker planes, and by 500 micrometres over the 5,178 mm of the magnet crossing. The same physics that was invisible on a short leg is 50 times larger across the magnet. That is the entire reason this page exists.

# 2. Aims

We set out to answer four questions, in order.

1. **How far is the network from the reference?** This is the number every companion write-up quotes, and we restate it here so the comparison is on one page.
2. **How far is the reference from the Geant4 truth?** Measured on all 14,482 crossings, per momentum band, with both a signed and an unsigned metric.
3. **What is that gap made of?** We want it separated into integration error, field-map error, momentum error and multiple scattering, with a test that distinguishes them rather than an assertion.
4. **What can be fixed, what is a floor, and what should the networks be scored against?** The last is a decision for George, and we lay out the options rather than take it.

# 3. Method

This section has four parts: the data pipeline and where each physical effect enters it; how the three-way comparison is made; the two tests that take the gap apart; and what we already knew about the reference's own accuracy.

## 3.1 The data pipeline, link by link

The chain from a simulated collision to a trained network has eight links.

1. **The sample.** Official central production from the LHCb test-file database, entry `expected_2024_minbias_xdigi`, production 00212966, 200 events, read from the local CVMFS mirror. Conditions `dddb-20231017` and `sim-20231017-vc-mu100`, data type 2024, magnet up. These are not events we generated ourselves; George ruled that out in July 2026 as a provenance source.
2. **The truth dump.** A Gaudi job walks the packed Monte Carlo truth of each event and writes three comma-separated files: one row per particle, one per vertex, one per Monte Carlo hit. The hit row carries the entry point, the exit point, the deposited energy, the time, the path length and the sensor identifier. It does **not** carry the particle's momentum at the hit. The method `MCHit::p()` exists on every hit and returns exactly that, but our dump never asked for it. This omission is the origin of the systematic we measure in section 4.2.
3. **The states.** A harvesting script turns each tracker hit into a state: midpoint for position, entry-to-exit displacement over its own $`\Delta z`$ for the slopes, and then joins on the particle table for the charge and the momentum. The momentum column it joins is computed from the particle's production momentum components. Every state a particle has, anywhere in the detector, therefore carries the momentum the particle was born with.
4. **The cut cascade.** From 41,398 cross-magnet rows we require that the pre-magnet plane really is an Upstream Tracker plane (2,476 rows removed), that both directions of the particle survive (2,388 removed), that the pseudorapidity is between 2 and 5 (1,240 removed), that the momentum is between 1 and 200 GeV (0 removed, it is already the set's own domain), and that the particle is not an electron (4,780 removed, because electrons radiate). Keeping one forward row per particle and requiring that its planes sit within 60 mm of $`z_0`$ and $`z_1`$ removes a further 775 and leaves **14,482 crossings**, every one of which stays inside the field map along its whole path.
5. **The reference track.** Each start state is carried to $`z_0`$, then integrated to $`z_1`$, and stored on 257 planes $`z_0 + kL/256`$ along the way, then carried on to the particle's own fibre-tracker plane $`z_{\text{post}}`$. Everything here is the sixth-order integrator at 0.1 mm.
6. **The split.** By particle, inherited from the training set: 11,567 training, 1,463 validation, 1,452 test. This page uses all three concatenated, 14,482 crossings, except where it quotes the network numbers, which are on the 1,452 test crossings only.
7. **The training states.** The networks are trained on states drawn from those reference tracks, with the loss built from the equation of motion alone.
8. **The network.** Applied $`N`$ times from the real Upstream Tracker state to reach $`z_1`$.

The following table says which physical effect is present at which link. It is the heart of the method section: everything in section 4 follows from reading it.

<table header-row="true">
<tr>
<td>physical effect</td>
<td>in the Geant4-true state?</td>
<td>in the field-only reference?</td>
<td>in the network's training loss?</td>
</tr>
<tr>
<td>magnetic deflection</td>
<td>yes, from the simulation's own field</td>
<td>yes, from the v8r1 up map</td>
<td>yes, this is the whole loss</td>
</tr>
<tr>
<td>multiple Coulomb scattering</td>
<td>yes</td>
<td>no</td>
<td>no</td>
</tr>
<tr>
<td>ionisation energy loss along the crossing</td>
<td>yes</td>
<td>no, q/p is held constant</td>
<td>no</td>
</tr>
<tr>
<td>energy already lost before the start plane</td>
<td>yes, the particle really is slower</td>
<td>no, the start state carries the production momentum</td>
<td>no</td>
</tr>
<tr>
<td>hadronic interactions and decays</td>
<td>yes, where the particle survives to the plane</td>
<td>no</td>
<td>no</td>
</tr>
<tr>
<td>bremsstrahlung</td>
<td>yes, but electrons are cut from the set</td>
<td>no</td>
<td>no</td>
</tr>
<tr>
<td>digitisation and measurement resolution</td>
<td>no, these are Monte Carlo hits</td>
<td>no</td>
<td>no</td>
</tr>
<tr>
<td>numerical integration error</td>
<td>Geant4's own stepper, not measured here</td>
<td>yes, and it is tiny: see section 3.4</td>
<td>yes, plus the optimiser's own error</td>
</tr>
</table>

Rows two to five are present on one side of the comparison and absent on the other. They are the gap.

## 3.2 How the comparison is made

Each particle's own first fibre-tracker plane $`z_{\text{post}}`$ is not exactly $`z_1`$, because the tracker is not a single flat plane. In our set $`|z_{\text{post}} - z_1|`$ is at most 7.45 mm, well inside the 60 mm selection window. We therefore carry **both** the reference state at $`z_1`$ and the network's state at $`z_1`$ from $`z_1`$ to that particle's own $`z_{\text{post}}`$, with the same sixth-order integrator, and compare each with the true state there. This is the comparison every training record already stores, so the network numbers in section 4.1 are read from those records and not recomputed.

Two metrics appear below and we name both every time.

- The **max metric**, $`\max(|\Delta x|, |\Delta y|)`$, quoted in micrometres. This is what the momentum-band tables use.
- The **radial metric**, $`\sqrt{\Delta x^2 + \Delta y^2}`$, also in micrometres. This is what the comparator table in section 4.6 uses.

We also distinguish two ways of applying a network, and say which one every number is.

- **Endpoint error**: the network applied $`N`$ times from the particle's real Upstream Tracker state, all the way across the magnet, compared at the fibre-tracker plane. Every network number on this page is an endpoint error.
- **Single-step error**: one application of the network from a state taken off the reference track. None are quoted here.

**A note on the slope units.** $`t_x`$ and $`t_y`$ are dimensionless slopes, taken from the Monte Carlo hit's exit minus entry displacement divided by its own $`\Delta z`$. No arctangent is applied anywhere. Our analysis scripts and figure labels write "mrad" for a slope difference multiplied by $`10^{3}`$, which is not quite the same thing: the angle difference is $`\Delta\theta \approx \Delta t_x / (1 + t_x^2)`$, so for tracks inside our acceptance the two agree to within 7 per cent at the very edge and within 1 per cent for most tracks. In the text below we write "$`\times 10^{-3}`$ (slope)" and mean the dimensionless quantity.

## 3.3 The decomposition: two tests

We write $`\Delta = S_{\text{true}} - S_{\text{ref}}`$ for the true state minus the reference state at $`z_{\text{post}}`$, and $`\Delta x`$ for its first component. The whole decomposition rests on how $`\Delta x`$ behaves under a flip of the particle's charge, and on dividing it by how much the track bent.

**The bend.** For each track we define

$$
b = x_{\text{ref}}(z_{\text{post}}) - \left[x_0 + t_{x,0}\,(z_{\text{post}} - z_0)\right],
$$

the horizontal distance between where the reference track ends up and where a straight line through the start state would have put it. This is the magnet's whole effect on that track. Its median magnitude over our set is 453 mm, and 14,476 of the 14,482 tracks have $`|b| > 20`$ mm, which is the cut we use whenever we divide by it.

**Test one: the sign under a charge flip.** Look at the equation of motion again. The charge enters only through $`q/p`$, and it enters the two slope equations linearly. If we use the wrong momentum, so that $`q/p`$ is too large by a relative amount $`\epsilon = \Delta p / p`$, every slope derivative is scaled by $`(1 + \epsilon)`$, and to first order the whole bend is scaled by $`(1 + \epsilon)`$ as well. A positive particle bends one way and a negative particle the other, so

$$
\Delta x \approx \epsilon \, b \quad\text{flips sign with the charge, while}\quad \frac{\Delta x}{b} \approx \epsilon \quad\text{does not.}
$$

Multiple scattering has no such structure. A scattered particle is as likely to go left as right whatever its charge, so its contribution to the **median** of $`\Delta x`$ is zero for both charges, and it shows up only in the **width** of the distribution, which is charge-blind.

This gives us a clean separation. We split the set by charge and by momentum band and look at two things: the signed median of $`\Delta x`$, which isolates the systematic, and the width of $`\Delta x`$ about that median, which isolates the random part. A worked micro-example: a 5 GeV track that bends by 500 mm and whose reference misses by 2,500 micrometres in $`x`$ has $`\Delta x / b = 5 \times 10^{-3}`$, which reads as a momentum defect of $`5 \times 10^{-3} \times 5\ \text{GeV} = 25`$ MeV.

**Reading $`\Delta x / b`$ as a momentum.** Because $`\Delta x / b \approx \Delta p / p`$, multiplying the band's median ratio by the band's median momentum gives the momentum defect in absolute units. If the defect is a **fixed number of MeV** independent of the momentum, that is characteristic of energy loss: a particle of any momentum loses roughly the same absolute amount crossing the same material. If instead the ratio were **constant** across momentum, that would point at the field rather than the momentum: a field map scaled wrongly by a fixed fraction produces a fixed relative over-bend at every momentum, charge-blind in $`\Delta x / b`$.

**Test two: the width against Highland.** For the random part we compare the measured width with what multiple Coulomb scattering should give. The standard parameterisation is Highland's,

$$
\theta_0 = \frac{13.6\ \text{MeV}}{\beta c p}\, z \sqrt{\frac{x}{X_0}} \left[1 + 0.038 \ln\!\left(\frac{x}{X_0}\right)\right],
$$

where $`\theta_0`$ is the root-mean-square scattering angle projected onto one plane, $`p`$ is the momentum, $`\beta c`$ the velocity (which is $`c`$ to an excellent approximation for every track here, the slowest being 1.5 GeV), $`z = 1`$ the charge in units of the elementary charge, and $`x/X_0`$ the thickness of material traversed in radiation lengths. A particle that scatters continuously over a length $`L`$ ends up displaced sideways by

$$
\sigma_{\text{displacement}} \approx \frac{\theta_0 L}{\sqrt{3}},
$$

the $`\sqrt{3}`$ arising because deflections earned near the end of the path have almost no lever arm left to act over.

The only material between our two planes that we can account for from first principles is air. Taking the radiation length of air as 304 m and the crossing length as 5,177.8 mm gives $`x/X_0 = 0.01703`$, that is 1.70 per cent of a radiation length. We compute the Highland prediction for each band's median momentum, compare it with the measured width, and then invert the formula numerically for the $`x/X_0`$ that the measured width implies. That inverted number is a statement about how much material the simulation actually puts between the planes, which we have not looked up independently.

**Which width.** Because the two charges have systematically different centres, pooling them without care would let the systematic inflate the width. We therefore move each charge onto its own median first, then pool, then take the 68 per cent half-width, that is the 68th percentile of the absolute deviation from the centre. This is the statistic quoted as "68 per cent half-width" throughout section 4.

## 3.4 What we already knew about the reference's own accuracy

Before attributing anything to physics, we have to be sure the integration itself is not the problem. Four independent numbers say it is not.

- **Closure (gate G1).** Integrating a state forward and then back returns it to where it started with a median error of $`2.3 \times 10^{-6}`$ micrometres over 2,000 tracks, worst case 22.4 micrometres.
- **Step convergence (gate G3).** Halving and fifthing the step changes the answer by a median of $`2.8 \times 10^{-5}`$ micrometres, that is 28 picometres, worst case 2.8 micrometres, over 200 tracks.
- **The exact collocation scheme.** Solving the same collocation equations exactly, with no network, and chaining them across the magnet gives an endpoint radial median of 0.147 micrometres at 64 steps with 2 stages, 0.0115 micrometres at 128 steps with 8 stages, and 0.00105 micrometres at 256 steps with 16 stages, against the reference on the 1,452 test tracks.
- **Gate G2 on short legs**, already quoted in section 1.4: 11.9 micrometres median, 3.8 above 5 GeV, over 100,660 legs.

Sub-micrometre against the same equation, and millimetres against the simulation. The reference solves its equation essentially exactly. The equation is what is incomplete.

# 4. Results

## 4.1 The three references side by side

We first put all three objects on one plot, using the three reweighted-loss networks that are currently trained. All numbers are the median of the max metric at the particle's own first fibre-tracker plane, on the 1,452 test crossings, endpoint error, in micrometres.

<table header-row="true">
<tr>
<td>momentum band</td>
<td>n</td>
<td>network vs reference</td>
<td>network vs Geant4 truth</td>
<td>reference vs Geant4 truth</td>
</tr>
<tr>
<td>all</td>
<td>1,452</td>
<td>81 / 96 / 84</td>
<td>1,698 / 1,649 / 1,699</td>
<td>1,694</td>
</tr>
<tr>
<td>1 to 2 GeV</td>
<td>5</td>
<td>2,214 / 2,419 / 1,577</td>
<td>13,750 / 15,091 / 17,098</td>
<td>17,034</td>
</tr>
<tr>
<td>2 to 5 GeV</td>
<td>500</td>
<td>335 / 383 / 316</td>
<td>5,217 / 5,251 / 5,130</td>
<td>5,219</td>
</tr>
<tr>
<td>5 to 10 GeV</td>
<td>430</td>
<td>77 / 97 / 79</td>
<td>1,686 / 1,675 / 1,688</td>
<td>1,630</td>
</tr>
<tr>
<td>10 to 25 GeV</td>
<td>383</td>
<td>24 / 28 / 36</td>
<td>623 / 628 / 610</td>
<td>613</td>
</tr>
<tr>
<td>25 to 200 GeV</td>
<td>134</td>
<td>17 / 20 / 23</td>
<td>276 / 270 / 274</td>
<td>255</td>
</tr>
</table>

The three entries in each network cell are, in order, the network of 64 steps with 2 stages and an 81 mm step, the network of 128 steps with 8 stages and a 40 mm step, and the network of 256 steps with 16 stages and a 20 mm step.

Two statements follow, and we give the measured ranges rather than round ones. Leaving aside the 1 to 2 GeV band, which holds 5 test tracks and where no statement is meaningful, the network-versus-truth median differs from the reference-versus-truth median by between $`-2.6`$ and $`+8.4`$ per cent across all bands and all three networks. Against the truth, in other words, the networks and the reference are the same object. Meanwhile the reference-versus-truth error is between 11 and 25 times larger than the network-versus-reference error. Including the 1 to 2 GeV band widens these to $`-19.3`$ to $`+8.4`$ per cent and 7 to 25 times.

(An earlier internal note quoted "0 to 4 per cent" and "6 to 20 times" for these two ranges. Recomputing them from the same file gives the ranges above; the qualitative statement is unchanged, but the numbers here are the ones to use.)

![Three references for one magnet crossing, endpoint error on the 1,452 test crossings: the median of max(dx, dy) at the particle's own first fibre-tracker plane, against momentum band. Solid black is the field-only reference against the Geant4-true state; the coloured squares are each network against the Geant4-true state, sitting on top of it; the coloured dashed triangles are each network against the field-only reference, an order of magnitude below. Networks are named by step count, stage count and step length.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/reference_vs_truth/figures/three_references.png)

## 4.2 The decomposition, band by band and charge by charge

Now the main measurement, on all 14,482 crossings. For each momentum band we give the number of tracks, the signed median of $`\Delta x`$ for each charge, the median of $`\Delta x / b`$ for each charge, the momentum defect that ratio implies for the band, and the charge-corrected 68 per cent half-width of $`\Delta x`$. Positions are in micrometres.

<table header-row="true">
<tr>
<td>band</td>
<td>n (positive / negative)</td>
<td>median dx, positive / negative</td>
<td>median dx / bend, positive / negative</td>
<td>implied momentum defect</td>
<td>68 per cent half-width of dx</td>
</tr>
<tr>
<td>1 to 2 GeV</td>
<td>89 (63 / 26)</td>
<td>-14,450 / +14,832</td>
<td>+8.80e-3 / +8.50e-3</td>
<td>+16.0 MeV</td>
<td>9,999</td>
</tr>
<tr>
<td>2 to 5 GeV</td>
<td>5,117 (2,649 / 2,468)</td>
<td>-4,826 / +4,509</td>
<td>+5.39e-3 / +4.97e-3</td>
<td>+17.9 MeV</td>
<td>4,372</td>
</tr>
<tr>
<td>5 to 10 GeV</td>
<td>4,286 (2,216 / 2,070)</td>
<td>-1,100 / +973</td>
<td>+2.51e-3 / +2.24e-3</td>
<td>+16.4 MeV</td>
<td>1,521</td>
</tr>
<tr>
<td>10 to 25 GeV</td>
<td>3,697 (1,870 / 1,827)</td>
<td>-118 / +139</td>
<td>+6.11e-4 / +6.91e-4</td>
<td>+9.4 MeV</td>
<td>647</td>
</tr>
<tr>
<td>25 to 50 GeV</td>
<td>1,016 (496 / 520)</td>
<td>+24 / -20</td>
<td>-2.70e-4 / -2.48e-4</td>
<td>-8.1 MeV</td>
<td>312</td>
</tr>
<tr>
<td>50 to 100 GeV</td>
<td>240 (126 / 114)</td>
<td>+52 / -58</td>
<td>-1.19e-3 / -1.29e-3</td>
<td>-77.5 MeV</td>
<td>167</td>
</tr>
<tr>
<td>100 to 200 GeV</td>
<td>37 (19 / 18)</td>
<td>+51 / -58</td>
<td>-1.73e-3 / -2.14e-3</td>
<td>-240.4 MeV</td>
<td>86</td>
</tr>
<tr>
<td>all momenta</td>
<td>14,482 (7,439 / 7,043)</td>
<td>-1,032 / +873</td>
<td>+2.65e-3 / +2.37e-3</td>
<td>+17.0 MeV</td>
<td>2,262</td>
</tr>
</table>

Read the table with test one in mind and it says three things at once.

First, **the signed median flips sign with the charge and the ratio to the bend does not.** At 2 to 5 GeV a positive particle is found 4,826 micrometres to one side of the reference and a negative particle 4,509 micrometres to the other, while both give $`\Delta x / b \approx +5 \times 10^{-3}`$. That is the signature of a momentum error and of nothing else. A field-scale error would be charge-blind in $`\Delta x / b`$ too, so this test alone does not separate the two; the next paragraph does.

Second, **below 25 GeV the implied defect is a constant number of MeV, not a constant fraction.** It is 16.0, 17.9, 16.4 and 9.4 MeV in the four lowest bands, while the relative ratio falls by more than an order of magnitude across the same range. A fixed absolute momentum loss is what ionisation energy loss looks like and is not what a field-scale error looks like. The simulated particle really is slower than the number we handed the integrator, by of order 16 MeV, because the start state carries the particle's **production** momentum: the harvesting script takes it from the particle table, and the hit dump never wrote `MCHit::p()`. What the 16 MeV contains is the energy the particle lost in the vertex detector and the Upstream Tracker before reaching the start plane, plus the energy it loses in the crossing itself, which the integrator also does not model.

Third, **above 25 GeV the sign reverses and the ratio grows.** The implied defect becomes $`-8.1`$, $`-77.5`$ and $`-240.4`$ MeV in the three highest bands, which is a relative under-bend of 0.025, 0.12 and 0.20 per cent. A defect that grows with momentum is not energy loss. We state this as unexplained and name the candidates without choosing between them: a difference between the field map we integrate in, v8r1 upward polarity, and the map and current the simulation was run with under the condition tag `sim-20231017-vc-mu100`, a scale difference of a tenth of a per cent being exactly the right size; or the accuracy of Geant4's own field stepper. The three bands hold 1,016, 240 and 37 tracks, so the highest band in particular carries little weight. Section 5 lists the measurement that would settle it.

![Left: the median of the true state minus the field-only reference in x, by momentum band, separately for positive and negative tracks, on all 14,482 crossings. The two curves are mirror images, which is the signature of a bending systematic. Right: the same quantity divided by how far the track bent. The two curves now lie on top of each other, and the common value is the relative over-bend, positive below 25 GeV and negative above it.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/reference_vs_truth/figures/signed_centre_by_charge.png)

## 4.3 The random part against Highland

The last column of the table above is charge-blind and falls roughly as $`1/p`$, which is what scattering does. Here it is against the Highland prediction for air alone.

<table header-row="true">
<tr>
<td>band</td>
<td>median momentum in GeV</td>
<td>air-only prediction in micrometres</td>
<td>measured 68 per cent half-width in micrometres</td>
<td>ratio</td>
<td>thickness the width implies, in radiation lengths</td>
</tr>
<tr>
<td>1 to 2 GeV</td>
<td>1.86</td>
<td>2,407</td>
<td>9,999</td>
<td>4.15</td>
<td>0.235</td>
</tr>
<tr>
<td>2 to 5 GeV</td>
<td>3.44</td>
<td>1,304</td>
<td>4,372</td>
<td>3.35</td>
<td>0.158</td>
</tr>
<tr>
<td>5 to 10 GeV</td>
<td>6.88</td>
<td>652</td>
<td>1,521</td>
<td>2.33</td>
<td>0.081</td>
</tr>
<tr>
<td>10 to 25 GeV</td>
<td>14.58</td>
<td>308</td>
<td>647</td>
<td>2.10</td>
<td>0.067</td>
</tr>
<tr>
<td>25 to 50 GeV</td>
<td>32.06</td>
<td>140</td>
<td>312</td>
<td>2.23</td>
<td>0.074</td>
</tr>
<tr>
<td>50 to 100 GeV</td>
<td>62.50</td>
<td>72</td>
<td>167</td>
<td>2.32</td>
<td>0.080</td>
</tr>
<tr>
<td>100 to 200 GeV</td>
<td>120.25</td>
<td>37</td>
<td>86</td>
<td>2.30</td>
<td>0.079</td>
</tr>
</table>

Above 5 GeV the measured width is a stable 2.1 to 2.3 times the air-only prediction, and the implied thickness is a stable 6.7 to 8.1 per cent of a radiation length, against the 1.70 per cent that air alone provides. In other words, the simulation puts about four times more scattering material between the two planes than the air does, which is entirely plausible: the magnet region is not empty, and the last Upstream Tracker and first fibre-tracker stations themselves have structure. We have not independently looked up the LHCb material budget for this region, so we quote the implied thickness as a measurement of the simulation and not as a confirmation of anything.

Below 5 GeV the ratio climbs to 3.4 and 4.2. Highland's parameterisation is a Gaussian core description and degrades in exactly this regime, where the non-Gaussian single-scattering tail is large and where the 68 per cent half-width starts to sample it; we do not read the low-momentum rows as evidence of more material.

![The charge-corrected 68 per cent half-width of the true state minus the field-only reference in x, against the band's median momentum, on all 14,482 crossings, both axes logarithmic. Black circles are the measurement. The dashed line is Highland's prediction for the air between the planes alone, 1.70 per cent of a radiation length. The dash-dotted line is Highland for 6.7 per cent of a radiation length, the thickness the 10 to 25 GeV width implies.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/reference_vs_truth/figures/width_vs_momentum.png)

## 4.4 By particle type

If the systematic really is energy loss, then at the same momentum a heavier particle should lose less, because ionisation loss depends on velocity and a heavier particle at a given momentum is slower only in the sense that it is further from the relativistic rise. The prediction is that the over-bend should order itself by mass. Measured at 5 to 25 GeV with the bend cut applied:

<table header-row="true">
<tr>
<td>particle</td>
<td>n</td>
<td>median momentum in GeV</td>
<td>median dx / bend</td>
<td>median absolute dx in micrometres</td>
<td>implied defect in MeV</td>
</tr>
<tr>
<td>pion</td>
<td>6,139</td>
<td>9.25</td>
<td>+1.65e-3</td>
<td>811</td>
<td>+15.3</td>
</tr>
<tr>
<td>kaon</td>
<td>909</td>
<td>10.44</td>
<td>+1.48e-3</td>
<td>699</td>
<td>+15.4</td>
</tr>
<tr>
<td>proton</td>
<td>893</td>
<td>9.82</td>
<td>+1.23e-3</td>
<td>395</td>
<td>+12.1</td>
</tr>
<tr>
<td>muon</td>
<td>42</td>
<td>7.40</td>
<td>+1.11e-3</td>
<td>811</td>
<td>+8.2</td>
</tr>
</table>

The ordering pion, kaon, proton is the predicted one, and the muons, with 42 tracks, sit lowest but carry no statistical weight. The effect is modest, about 25 per cent in the ratio between pion and proton. We record the ordering as consistent with an energy-loss explanation and do not quantify it further here; the rebuilt dataset of step 1 below will measure the loss per track directly and make this test sharp.

## 4.5 The slopes tell the same story

Everything above is about position. The slopes behave the same way, which is what one would expect if a single physical cause is producing both. Median absolute residuals on all 14,482 crossings, both charges pooled, positions in micrometres and slopes as a dimensionless difference multiplied by $`10^{3}`$:

<table header-row="true">
<tr>
<td>band</td>
<td>n</td>
<td>median abs dx in micrometres</td>
<td>median abs dtx, slope difference x 1e-3</td>
<td>median abs dy in micrometres</td>
<td>median abs dty, slope difference x 1e-3</td>
</tr>
<tr>
<td>1 to 2 GeV</td>
<td>89</td>
<td>14,723</td>
<td>6.44</td>
<td>3,763</td>
<td>1.02</td>
</tr>
<tr>
<td>2 to 5 GeV</td>
<td>5,117</td>
<td>4,826</td>
<td>1.69</td>
<td>1,713</td>
<td>0.518</td>
</tr>
<tr>
<td>5 to 10 GeV</td>
<td>4,286</td>
<td>1,268</td>
<td>0.356</td>
<td>818</td>
<td>0.232</td>
</tr>
<tr>
<td>10 to 25 GeV</td>
<td>3,697</td>
<td>409</td>
<td>0.107</td>
<td>371</td>
<td>0.108</td>
</tr>
<tr>
<td>25 to 50 GeV</td>
<td>1,016</td>
<td>178</td>
<td>0.0561</td>
<td>195</td>
<td>0.0513</td>
</tr>
<tr>
<td>50 to 100 GeV</td>
<td>240</td>
<td>120</td>
<td>0.0339</td>
<td>72</td>
<td>0.0225</td>
</tr>
<tr>
<td>100 to 200 GeV</td>
<td>37</td>
<td>76</td>
<td>0.0215</td>
<td>30</td>
<td>0.0123</td>
</tr>
<tr>
<td>all momenta</td>
<td>14,482</td>
<td>1,286</td>
<td>0.368</td>
<td>776</td>
<td>0.220</td>
</tr>
</table>

Two features are worth naming. At low momentum the horizontal residual is three to four times the vertical one, and the horizontal slope residual six times the vertical one: the magnet bends in $`x`$, so the momentum systematic appears there and not in $`y`$. Above 10 GeV, where the systematic has largely gone, the two components converge, as a purely random cause requires. The pooled max metric over the whole set is a median of 1,736 micrometres and a 95th percentile of 12,628 micrometres.

## 4.6 What the reference does agree with

For completeness, the comparators we have accumulated, all endpoint radial medians on the 1,452 test crossings except where noted:

<table header-row="true">
<tr>
<td>quantity</td>
<td>value</td>
<td>n</td>
</tr>
<tr>
<td>closure of the integrator, forward then back, median</td>
<td>2.3e-6 micrometres</td>
<td>2,000</td>
</tr>
<tr>
<td>closure of the integrator, worst case</td>
<td>22.4 micrometres</td>
<td>2,000</td>
</tr>
<tr>
<td>step convergence, 5 mm against 1 mm, median</td>
<td>2.8e-5 micrometres</td>
<td>200</td>
</tr>
<tr>
<td>step convergence, worst case</td>
<td>2.8 micrometres</td>
<td>200</td>
</tr>
<tr>
<td>label against the particle's next hit, short legs, median</td>
<td>11.9 micrometres</td>
<td>100,660</td>
</tr>
<tr>
<td>the same, above 5 GeV</td>
<td>3.8 micrometres</td>
<td>100,660</td>
</tr>
<tr>
<td>the same, below 2 GeV</td>
<td>38.8 micrometres</td>
<td>100,660</td>
</tr>
<tr>
<td>exact collocation scheme, 64 steps, 2 stages, 81 mm step</td>
<td>0.147 micrometres</td>
<td>1,452</td>
</tr>
<tr>
<td>exact collocation scheme, 128 steps, 8 stages, 40 mm step</td>
<td>0.0115 micrometres</td>
<td>1,452</td>
</tr>
<tr>
<td>exact collocation scheme, 256 steps, 16 stages, 20 mm step</td>
<td>0.00105 micrometres</td>
<td>1,452</td>
</tr>
<tr>
<td>straight line, no magnet at all</td>
<td>450,542 micrometres</td>
<td>1,452</td>
</tr>
<tr>
<td>material floor, true state against the reference track</td>
<td>1,896 micrometres</td>
<td>1,452</td>
</tr>
</table>

And one band deserves singling out. At 100 to 200 GeV the median absolute horizontal residual against the Geant4 truth is **76 micrometres**, with the stale start momentum still in place. That is the scale at which a field-only reference and the simulation agree once scattering has become small, and it is a useful reminder of what is achievable: the gap is not an intrinsic property of the method, it is a property of the momenta involved.

# 5. Conclusion and next steps

## 5.1 What we now know

The reference is correct for the equation it solves. Sub-micrometre closure, picometre step convergence and sub-micrometre agreement with the exact collocation scheme leave no room for integration error at the millimetre scale. The 1,736 micrometre median gap to the simulation is the equation's incompleteness, and it has three distinct parts.

1. **The momentum at the start plane is the production momentum, not the momentum the particle actually had at the Upstream Tracker.** This is a defect in our data pipeline, not physics: `MCHit::p()` exists on every hit and our dump never wrote it. It shows up as a charge-antisymmetric over-bend worth a roughly constant 16 MeV below 25 GeV. **This is fixable**, and cheaply.
2. **Energy loss along the crossing itself.** The equation holds $`q/p`$ constant. A real particle loses energy continuously as it goes. This is deterministic and therefore **addable to the equation**, which is what LHCb's production extrapolator does in its own material layer. Part of the 16 MeV above belongs here rather than to the start plane, and only the rebuilt dataset will say how the 16 MeV splits between the two.
3. **Multiple Coulomb scattering.** Random, charge-blind, falling as $`1/p`$, measured here at 647 micrometres of half-width at 10 to 25 GeV and about 2.2 times the air-only Highland estimate. This is **stochastic and therefore irreducible** for any deterministic reference. A deterministic extrapolator can predict its variance and propagate it as a covariance, which is exactly what a Kalman filter does with it, but it can never predict the individual deflection. No amount of extra Runge-Kutta order, extra collocation stages or extra network capacity can touch it.
4. Separately from these three, a **momentum-growing residual above 25 GeV** of $`-0.025`$ to $`-0.20`$ per cent, which we have not explained.

## 5.2 What this changes and what it does not

**Nothing in the network studies moves.** The networks and the reference share the same equation and the same $`q/p`$, so every network-against-reference number in the companion write-ups is unaffected by everything on this page. What moves is only the reference-against-truth column.

The honest sentence for a supervisor is therefore: our networks reproduce the field-only extrapolation to a median of 17 to 96 micrometres depending on the momentum band, endpoint error, max metric; and the field-only extrapolation itself differs from the Geant4 truth by material effects, 647 micrometres of random half-width at 10 to 25 GeV, plus a momentum systematic we have now identified and can remove.

## 5.3 The question behind the question

A reference is only well posed once we say what we want to model. There are three different targets and the programme should choose one deliberately.

- **Field-only propagation.** The equation as it stands. Its correctness is checkable to a micrometre against an independent integrator, but never against a physics sample, because the sample contains scattering. The material effects are then quoted beside every result as a floor.
- **The expected trajectory including mean energy loss.** The equation plus a $`dE/dx`$ term, which is what LHCb's master extrapolator provides. This removes parts one and two above and leaves scattering. The label-free construction handles this without difficulty: the loss is built from a residual, and a residual of a richer equation is still a residual.
- **The full distribution.** Predicting the individual scattered trajectory. Not possible for any deterministic method, and not a sensible target.

## 5.4 The plan, with costs

Five steps, in order. The first is a fix and needs no decision; the rest do.

1. **Re-dump the truth with the local momentum.** Add `MCHit::p()`, the magnitude of the momentum at the hit's entry point in MeV, as a column in a copy of the dump script, run it over the same 200 events into a new output directory, about ten minutes. Rebuild the states with $`q/p`$ from the hit's own momentum, keeping the production momentum as a second column, and rebuild the crossing set as a **new version** of the file. The current set stays untouched, so every existing network number remains exactly reproducible. Cost: under an hour.
2. **Decompose the gap again on the rebuilt set.** With $`q/p`$ taken at the Upstream Tracker hit, the charge-antisymmetric part should fall to what the loss along the crossing alone gives, which is now directly measurable per track as the momentum at the Upstream Tracker hit minus the momentum at the fibre-tracker hit; and to zero against a reference that loses that momentum along the way. Quote the remaining width per band as the stochastic floor. Cost: one day.
3. **Establish the field map's identity.** Determine which map and which current the simulation used under `sim-20231017-vc-mu100`, and compare it with our `field.v8r1.up.bin` at sampled points through a small Gaudi job. A scale difference of a tenth of a per cent is exactly the size of the high-momentum residual in section 4.2. Cost: half a day.
4. **The micrometre statement, where it can hold.** Run a local particle gun with multiple scattering and ionisation switched off in Geant4: muons at 5, 20 and 100 GeV, both charges, about a thousand tracks each. Integrate from the Upstream Tracker hit with its own $`q/p`$ and compare with the fibre-tracker hit. Target: a median below one micrometre. What then limits it is Geant4's own field stepping, and if the residual sits above a micrometre we tighten its accuracy parameters and repeat. Alongside it, integrate the same equation at 0.1 mm and at 0.02 mm and with an independent adaptive integrator at a relative tolerance of $`10^{-13}`$; we expect agreement below 0.01 micrometres. Cost: one day of setup, minutes of running.
5. **Two decisions for George.** (i) Should the reference, and therefore the training target, include energy loss along the crossing, as LHCb's master extrapolator does, or stay field-only with the material effects stated as the floor? (ii) What should the networks be scored against in the write-ups: the field-only reference, which is what they were trained towards, with the material floor quoted beside it; or the corrected reference?

Step 1 is a fix regardless of the answers to step 5, and we recommend it be run first.

# 6. Provenance

Repository [github.com/GeorgeWilliam1999/LHCb-Extrapolation](https://github.com/GeorgeWilliam1999/LHCb-Extrapolation). Working tree **uncommitted**: the folder `single_network_chain_discrete_approach/` is untracked in its entirety, local HEAD 1040ee9, remote `main` at a062bd8. The three figures on this page are pinned to `main` and **will render only after George commits and pushes this folder**; no commit hash has been invented for them.

**This study.** `single_network_chain_discrete_approach/reference_vs_truth/decompose.py`, run with `PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python decompose.py` on 22 September 2026, producing `results/decomposition_by_band.csv` (sections 4.2, 4.5), `results/decomposition_by_pid.csv` (section 4.4), `results/highland_air.csv` (section 4.3), `results/three_references.csv` (section 4.1), `results/rk6_self_consistency.csv` (sections 3.4, 4.6), `results/decompose.log`, and `figures/signed_centre_by_charge.png`, `figures/width_vs_momentum.png`, `figures/three_references.png`. The folder README maps each script to each output.

**Input data.** `single_network_chain_discrete_approach/Block_E_single_network_chain/E0_Track_dataset/results/tracks.npz`, 116 MB, built by `E0_Track_dataset/build_tracks.py`, holding 14,482 magnet crossings split 11,567 training, 1,463 validation, 1,452 test; all three splits are concatenated for every number in sections 4.2 to 4.5. Its metadata file `results/tracks_meta.json` carries the cut cascade and the material-floor table. Upstream of it: `Data_generation_exploration/Official_xdigi/training_v2/train_official_v2.npz` and its `.meta.json`, from the official test-file-database sample `expected_2024_minbias_xdigi`, production 00212966, 200 events, conditions `dddb-20231017` and `sim-20231017-vc-mu100`, data type 2024, magnet up. The truth dump is `Data_generation_exploration/Official_xdigi/dump_xdigi.py`, whose hit rows carry entry point, exit point, deposited energy, time, path length and sensor identifier and **not** `MCHit::p()`; the states are built by `Data_generation_exploration/Data/harvest_states.py`, which joins the momentum from the particle table, that is the production momentum. The label metadata records `"material_effects": "excluded by design (master-extrapolator responsibility)"`.

**The reference integrator.** `single_network_chain_discrete_approach/_shared/reference.py`: `deriv` for the equation of motion, `rk6_rows` for Butcher's seven-stage order-six method, `RK6_STEP = 0.1` mm, field map `/cvmfs/lhcb.cern.ch/lib/lhcb/DBASE/FieldMap/v8r1/cdf/field.v8r1.up.bin`, md5 `9e49ddc4313b589f273540e7e0bb513b`, loaded by the vendored `_shared/field_v8r1.py`.

**Network numbers (section 4.1).** `Block_F_reweighted_loss/F2_Analysis/against_true_state.py` and `results/against_true_state.csv`, which tabulate the `metrics.chain_scores` block already stored in each run's `record.json` under `Block_F_reweighted_loss/F1_Training/results/full/`, run folders `N064_q02`, `N128_q08` and `N256_q16`. Those three networks are two hidden layers of 128 units, 18,956 parameters, trained on 11,567 tracks with the reweighted loss, 1,000 restarts over 40 rounds for the first two and 1,807 restarts over 60 rounds for the third; farm cluster 5809660. Section 4.1's percentage and ratio ranges are recomputed from the same CSV by `decompose.py` into `results/three_references.csv`.

**Comparators and gates (sections 3.4 and 4.6).** `Data_generation_exploration/Data/results/gates.json` for G1, G2 and G3; `Block_E_single_network_chain/E3_Analysis/results/error_qdz_chain.csv` and `Block_F_reweighted_loss/F3_Analysis/results/error_qdz_chain.csv` for the exact-scheme column; `E3_Analysis/results/comparators.csv` and `F3_Analysis/results/comparators.csv` for the straight line and the material floor. Note that the exact-scheme values live in `error_qdz_chain.csv`, not in `comparators.csv`.

**Trust: Provisional.** Provenance located and checked; every number above was read from one of the files named here by `decompose.py` or quoted directly from a named results file. Only George sets Verified.
