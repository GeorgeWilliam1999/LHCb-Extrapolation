<callout icon="🧭">
	This page is a picture supplement. It contains no new measurement: every number on it was read from a results file in the repository, and the file is named beside it. It stands beside the write-up that derives those numbers, [What the field-only reference shares with the Geant4 truth, and what it cannot](https://app.notion.com/p/3e35d544b9d9818282fff3be2d394009), and points at it rather than repeating it. **The seven figures are served from the public GitHub repository and will stay blank until George commits and pushes the folder that holds them**, which is untracked today. Trust is **Provisional** until George confirms it.
</callout>
<table_of_contents/>

# 1. Introduction

## 1.1 Why a supplement made only of pictures

The quantities in this programme are all correct and all written down, and they span ten decades. The magnet crossing is 5,177.8 mm long. The straight line a particle would follow with the field switched off misses the real endpoint by about 451 mm. The gap between our reference trajectory and what the simulation says actually happened is about 1.7 mm. The error our networks make against that reference is about 80 µm. The exact scheme the networks approximate sits at 0.15 µm, and the reference integrator's own convergence is smaller still. No reader, including us, can hold five metres and eighty micrometres in mind at the same time, and a table of numbers does not help, because a table flattens exactly the thing that matters here: how far apart these quantities are.

So we drew them. This page is seven figures, each made from the measured data rather than sketched, each at its true scale where a true scale is possible, together with a short reading of what each one shows. The derivations behind the numbers, the definitions of the objects being compared and the physics of the gap are all in the companion write-up linked above; nothing here restates them.

## 1.2 The three objects, named once

Three different states exist at the far end of the crossing for one and the same particle, and every figure on this page is about the differences between them.

1. **The simulated truth.** The state built from the Monte Carlo hit that Geant4, the simulation toolkit that transports each particle through a model of the detector, recorded on the particle's first scintillating-fibre plane. It contains everything the simulation did to the particle, including energy loss and multiple scattering, and no detector effect at all: no digitisation and no pattern recognition enter anywhere in this pipeline.
2. **The field-only reference.** The trajectory obtained by taking the particle's real state on its last Upstream Tracker plane and integrating the LHCb equation of motion in the measured magnetic field across the magnet, with a sixth-order Runge-Kutta method at a fixed step of 0.1 mm. The equation holds the momentum constant and contains the field and nothing else: no energy loss, no scattering, no material.
3. **The network endpoint.** A small network, two hidden layers of 128 units, applied 64 times in succession from the same real Upstream Tracker state, one application per step of 80.9 mm. It was trained without labels, by requiring that its own predictions satisfy the same equation of motion inside each step, so whatever the equation omits the network omits too.

Their definitions, the equation in full and the reason the second and third differ from the first are given in [the companion write-up](https://app.notion.com/p/3e35d544b9d9818282fff3be2d394009), sections 1.3 and 3.

## 1.3 What a reader will be able to answer

After looking at the seven figures, a reader should be able to say where in the five metres each physical effect enters, which of the three descriptions above contains that effect and which does not, how large each of them is relative to the others, which part of the remaining disagreement with the simulation is a defect we will remove and which part is a floor that no deterministic method can cross, and what the network is actually asked to produce at each of its 64 steps.

This study lives in the repository folder `single_network_chain_discrete_approach/data_supplement/`, and it reads results produced by the neighbouring folders `Block_E_single_network_chain/`, `Block_F_reweighted_loss/` and `reference_vs_truth/`. Those folder names appear in this sentence and in the provenance only, never as names for the networks themselves, which we name by their step count, their stage count and their step length.

## 1.4 A roadmap

Section 2 states the three aims. Section 3 says how the figures were made, which file each one draws on, and the conventions a reader needs in order to read them without being misled. Section 4 is the seven figures, in order, each with a caption and a short reading. Section 5 says what the set of them establishes and what is still open. Section 6 is the provenance.

# 2. Aims

1. **Put every quantity in the problem on one honest scale**, so that the relative size of the straight line, the gap to the simulation, the network's error and the integrator's own floor can be seen rather than inferred from a table.
2. **Show which physical effect enters at which point in $`z`$**, and which description of the crossing contains it, in a form that answers the question directly rather than by implication.
3. **Separate what is a defect from what is a floor**: distinguish the part of the disagreement with the simulation that comes from something we did to our own data, and can therefore undo, from the part that is irreducible for any deterministic extrapolator.

# 3. Method

This section is short, because this is not a new measurement. It says how the pictures were produced, what they draw on and how to read them.

## 3.1 How the figures were made

Every figure is produced by a single script, `make_figures.py`, in about forty seconds, from three kinds of input: the stored crossing set, which holds the start state, the reference trajectory on 257 planes and the simulated-truth state for each of 14,482 magnet crossings; the LHCb field map, read at the points where it is needed; and the results files of the two studies that measured the quantities being drawn. Nothing on any figure is drawn by hand, and nothing is recomputed for this page.

The trajectories in figures 1, 3 and 7 are the stored reference tracks themselves. The field profile in figures 1 and 3 is read from the magnet-up field map, version v8r1, at 300 to 400 points along the axis. The detector envelopes in figure 3 are the $`z`$ ranges of the simulated hits themselves, not nominal drawings. Every number annotated on a figure is read from a results file and written into `results/figure_facts.json` as the figure is drawn, so that a figure and the table behind it cannot drift apart.

## 3.2 Which file each figure draws on

<table header-row="true">
	<tr>
		<td>figure</td>
		<td>what it shows</td>
		<td>drawn from</td>
	</tr>
	<tr>
		<td>1. the crossing to scale</td>
		<td>the 5,177.8 mm crossing at equal scale on both axes, three real tracks, and the field that bends them</td>
		<td>`tracks.npz` (test split); the v8r1 magnet-up field map</td>
	</tr>
	<tr>
		<td>2. the zoom cascade</td>
		<td>four panels at the far end, each about ten times closer than the last</td>
		<td>`tracks.npz`; `chain_states.npz` of the 64-step, two-stage network</td>
	</tr>
	<tr>
		<td>3. where each effect enters</td>
		<td>the detector along z, the five places something happens, and a ledger of what each description contains</td>
		<td>hit z ranges from `states.npz`; one real track from `tracks.npz`</td>
	</tr>
	<tr>
		<td>4. the error budget</td>
		<td>every error in the problem on one logarithmic axis, by momentum band</td>
		<td>`against_true_state.csv`, `decomposition_by_band.csv`, `comparators.csv`, `error_qdz_chain.csv`, `gates.json`</td>
	</tr>
	<tr>
		<td>5. the charge split</td>
		<td>the gap split by charge: the centre flips, the width does not</td>
		<td>`tracks.npz`; `decomposition_by_band.csv`</td>
	</tr>
	<tr>
		<td>6. the scattering floor</td>
		<td>the random part of the gap against momentum, with the air-only prediction</td>
		<td>`highland_air.csv`</td>
	</tr>
	<tr>
		<td>7. one step</td>
		<td>one step of 80.9 mm, the two stage states, and the network's own single-step error</td>
		<td>`tracks.npz`; `error_qdz_single_step.csv`</td>
	</tr>
</table>

The full paths of all of these files are in section 6.

## 3.3 Conventions a reader needs

**Scale.** Every figure is drawn at equal scale on both axes, or in the units written on the axis, with two exceptions that say so on their own face. Figure 7 states its vertical exaggeration, about 120 times, in its axis label, because a step of 80.9 mm departs from a straight line by less than two tenths of a millimetre and at equal scale there would be nothing to see. Figure 4 is logarithmic, because its content spans ten decades.

**Which distance.** Two different distance measures appear in the results files of this programme, and we name the one in use beside every number. The **radial** distance is $`\sqrt{\Delta x^2 + \Delta y^2}`$. The **max metric** is $`\max(|\Delta x|, |\Delta y|)`$, which is the more conservative of the two to quote and never larger than the radial value. Positions are in micrometres or millimetres, and we say which.

**Endpoint or single step.** An **endpoint error** is the distance at the far end of the whole crossing: the network applied 64 times in succession from the real last Upstream Tracker state, compared with the reference trajectory there. A **single-step error** is one application of the network from a reference state, over one step of 80.9 mm. These differ by more than two orders of magnitude and we say which one every number is.

**Slopes.** The two slopes in a track state, $`t_x = dx/dz`$ and $`t_y = dy/dz`$, are dimensionless ratios and not angles. In older figures of this programme a slope difference is labelled "mrad", which there means the dimensionless difference multiplied by 1,000; that is within one per cent of true milliradians for most tracks and within seven per cent at the edge of the acceptance. No figure on this page plots a slope, so the label does not appear here.

**The scales the figures are drawn at.** Every one of these is read from a results file, and all endpoint values are medians over the 1,452 test crossings.

<table header-row="true">
	<tr>
		<td>quantity</td>
		<td>value</td>
		<td>metric</td>
	</tr>
	<tr>
		<td>the crossing</td>
		<td>5,177.8 mm, from z = 2,648.2 mm to z = 7,826.0 mm</td>
		<td>length</td>
	</tr>
	<tr>
		<td>one step of the 64-step chain</td>
		<td>80.90 mm</td>
		<td>length</td>
	</tr>
	<tr>
		<td>the bend of a 2.9 GeV track</td>
		<td>1,204.8 mm</td>
		<td>displacement in x</td>
	</tr>
	<tr>
		<td>straight line to the reference, endpoint</td>
		<td>450,542 µm</td>
		<td>radial median</td>
	</tr>
	<tr>
		<td>reference to the simulated truth, endpoint</td>
		<td>1,694 µm</td>
		<td>max-metric median</td>
	</tr>
	<tr>
		<td>network to the reference, endpoint</td>
		<td>80.6 µm</td>
		<td>max-metric median</td>
	</tr>
	<tr>
		<td>network to the reference, one step</td>
		<td>0.43 µm</td>
		<td>radial median over 92,928 steps</td>
	</tr>
	<tr>
		<td>the exact collocation scheme, endpoint</td>
		<td>0.147 µm</td>
		<td>radial median</td>
	</tr>
	<tr>
		<td>the reference integrator's own step convergence</td>
		<td>2.8 × 10⁻⁵ µm, that is 28 picometres</td>
		<td>median over 200 tracks</td>
	</tr>
</table>

The last row is the difference between integrating the same track at a 5 mm step and at a 1 mm step, which the gate file records as 2.8459 × 10⁻⁸ mm. An earlier version of figure 4 drew that line a thousand times too high, at 0.0285 µm, because the gate is recorded in millimetres and was read as though it were micrometres. The same slip appears in the data-generation README, which calls it 28 nanometres. The figure now reads the value from the filed table rather than from a typed constant, and the ordering on it was never affected.

## 3.4 The three case-study tracks

Figures 2, 3 and 7 each show one real track rather than a population, so each was chosen to sit near the middle of the population rather than picked for effect.

1. **Figure 2** uses the track, among those between 4 and 25 GeV, whose network endpoint error and whose reference-to-truth gap are simultaneously closest to the medians of both over the test set. It carries a momentum of 8.81 GeV. The medians over all 1,452 test tracks are drawn beside it as dotted rings, so a reader can see at once whether this track is typical.
2. **Figures 3 and 7** use the same track: the one between 4 and 8 GeV whose network endpoint error is closest to the median. It carries a momentum of 5.86 GeV.

# 4. Results

Seven figures, in order. Each has a caption saying what is plotted and on which tracks, then a short reading.

## 4.1 The crossing to scale

![Figure 1. The magnet crossing drawn at equal scale on both axes, so the curvature on the page is the real curvature. Upper panel: three real reference trajectories from the test split, at 2.9, 11.9 and 45.0 GeV, from the last Upstream Tracker plane at z = 2,648.2 mm to the first fibre-tracker plane at z = 7,826.0 mm, with the dashed line showing where the lowest-momentum track would have gone with no field. Lower panel: the magnitude of the vertical field component on the axis, read from the v8r1 magnet-up map over the same z range.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/data_supplement/figures/fig1_crossing_to_scale.png)

The crossing is 5,177.8 mm long and the bend is large: 1,204.8 mm for the 2.9 GeV track, and it falls as the inverse of the momentum, so the 45.0 GeV track, fifteen times stiffer, bends by about a fifteenth as much over the same distance. Because both axes carry the same scale, the curvature drawn is the curvature a particle really has, and it is worth noticing how gentle it is at this scale. The lower panel says where the work is done: the field is 0.21 T at the entry plane, rises to a peak of 1.05 T, and has fallen back to 0.30 T by the exit plane, so almost all of the deflection happens in the middle third and almost none at the ends. This is why a chain of equal steps across the crossing is not a chain of equal difficulty: the steps in the middle have most of the physics in them.

## 4.2 The zoom cascade

![Figure 2. Four views of the same place, the far end of the crossing, for one real 8.81 GeV test track, each panel about ten times closer than the last. The field-only reference is the origin of each panel; the other markers are where the simulated truth was, where the network put the track, and where a straight line with no field would have ended. The dotted rings are the medians over all 1,452 test tracks, endpoint error: orange for the reference-to-truth gap, blue for the network-to-reference error. Half-widths of the four panels: 600 mm, 8 mm, 400 µm and 120 µm.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/data_supplement/figures/fig2_zoom_cascade.png)

The order in which things become visible is the point of this figure. In the first panel only the straight line is distinguishable, 342.3 mm from the reference for this track, and the reference, the truth and the network are all inside a single marker. In the second, at a half-width of 8 mm, the simulated truth separates and sits 1.48 mm to one side and 0.77 mm below the reference, on the ring that marks the typical gap. In the third, at 400 µm, the truth has left the panel and the network separates for the first time; in the fourth, at 120 µm, the network's own endpoint error for this track, 49.8 µm in $`x`$ and 68.4 µm in $`y`$, sits on the ring marking the median network error over the test set. Three panels of zoom separate the network from the reference, and one more separates it from the typical size of its own error.

## 4.3 Where each effect enters

![Figure 3. One real 5.86 GeV track drawn from its origin through to its first fibre-tracker hit, with the five places along z where something enters numbered on it, and beneath it a ledger of which description of the crossing contains which effect. The grey bands are the detector envelopes taken from the z ranges of the simulated hits themselves: the vertex detector from −288.2 to 750.7 mm, the Upstream Tracker from 2,306.7 to 2,663.3 mm, and the fibre tracker from 7,817.3 to 9,412.1 mm. The blue wash is the field.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/data_supplement/figures/fig3_where_effects_enter.png)

This is the figure that answers the supervisors' question directly. Reading along $`z`$: the particle is made at point 1, and the momentum recorded there is the one our reference is given; between there and point 3 it crosses the vertex detector and the Upstream Tracker and loses 16 to 18 MeV, which the reference is never told about; at point 3 both the reference and the network start from the particle's real state on the last Upstream Tracker plane, so neither of them inherits any earlier error except through that momentum; across the five metres at point 4 the particle is kicked at random by the material it passes through, which neither a reference nor a network can follow; and at point 5 sits the simulated hit we compare with. The ledger beneath says the rest: the simulated truth contains bending, energy loss and scattering, while the reference and the network's training target contain bending alone. The network is trained on the residual of the same equation the reference solves, so it inherits exactly the same scope, and no detector resolution enters any row of the ledger, because every state called true here is a Monte Carlo hit and not a reconstructed one.

## 4.4 The error budget

![Figure 4. Every error in the problem on one logarithmic axis, by momentum band, at the first fibre-tracker plane. The solid orange line is the gap from the field-only reference to the simulated truth and the dashed orange line is the charge-antisymmetric part of it attributable to the stale start momentum; the blue line is the endpoint error of the 64-step, two-stage network against the reference. The three horizontal lines are the straight line with no field, the exact collocation scheme at 64 steps and two stages, and the reference integrator's own step convergence.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/data_supplement/figures/fig4_error_budget.png)

Read from the top: a straight line with the field off misses by 450,542 µm (radial median); the reference misses the simulated truth by 5,219 µm at 2 to 5 GeV falling to 255 µm above 25 GeV (max-metric median); the stale start momentum accounts for 4,826 µm of that at 2 to 5 GeV and only 24 µm above 25 GeV; the network misses the reference by 335 µm falling to 17 µm over the same bands (max-metric median); the exact scheme the network approximates sits at 0.147 µm and the integrator's own step convergence below it. Two features are worth pointing out. Above 25 GeV the stale momentum, 24 µm, and the network's own error, 17 µm, are comparable, so at high momentum fixing the start momentum matters as much as improving the network. Below 10 GeV the gap is no longer dominated by that systematic alone: at 5 to 10 GeV the random half-width, 1,521 µm, exceeds the 1,100 µm systematic, and at 2 to 5 GeV the two are of the same size, 4,372 µm against 4,826 µm, so both parts have to be dealt with and neither can be ignored.

<table header-row="true">
	<tr>
		<td>momentum band</td>
		<td>reference to truth</td>
		<td>of which stale start momentum</td>
		<td>network to reference</td>
	</tr>
	<tr>
		<td>2 to 5 GeV</td>
		<td>5,219 µm</td>
		<td>4,826 µm</td>
		<td>335 µm</td>
	</tr>
	<tr>
		<td>5 to 10 GeV</td>
		<td>1,630 µm</td>
		<td>1,100 µm</td>
		<td>77 µm</td>
	</tr>
	<tr>
		<td>10 to 25 GeV</td>
		<td>613 µm</td>
		<td>118 µm</td>
		<td>24 µm</td>
	</tr>
	<tr>
		<td>25 to 200 GeV</td>
		<td>255 µm</td>
		<td>24 µm</td>
		<td>17 µm</td>
	</tr>
</table>

Two populations meet on this figure and we say so rather than letting it pass. The solid orange and blue lines are medians over the 1,452 test crossings, in the max metric, endpoint error. The dashed line is the absolute median offset of the positively charged tracks measured over all 14,482 crossings, and for the highest band it is the 25 to 50 GeV row of that table, since the decomposition is tabulated in narrower bands at high momentum. The two horizontal reference lines are radial medians rather than max-metric ones. None of these choices changes an ordering, and each is traceable to the file it came from in section 6.

## 4.5 The charge split

![Figure 5. The gap between the simulated truth and the field-only reference, split by the sign of the charge. Left: for the 430 test crossings between 5 and 10 GeV, the difference in x plotted against how far the field bent the track, with the median of each charge drawn as a dashed line. Right: the same difference expressed as a fraction of the bend, as parts per thousand, per momentum band, from the decomposition over all 14,482 crossings.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/data_supplement/figures/fig5_charge_separation.png)

In the left panel the two charges sit on opposite sides of zero while the spread about each median is the same width for both, which is the signature that separates a momentum error from a random one: a momentum error bends one charge one way and the other charge the other way, whereas scattering does not know the sign of the charge. Over all 14,482 crossings the medians in the 5 to 10 GeV band are −1,100 µm for positive tracks and +973 µm for negative tracks; the left panel is drawn on the test crossings in that band alone, so the two dashed lines on it are the test-set medians and sit slightly further from zero than those two numbers. The right panel is the test that settles it: expressed as a fraction of the bend the two charges agree, +2.51 and +2.24 parts per thousand in that band, which is what a momentum deficit looks like and is not what a wrong field map looks like. The deficit implied by that ratio is 16 to 18 MeV in every band below 10 GeV, a constant in momentum rather than a constant fraction of it, which is the behaviour of material crossed before the starting plane.

## 4.6 The scattering floor

![Figure 6. The random, charge-blind part of the gap between the simulated truth and the field-only reference, measured on all 14,482 crossings as the 68 per cent half-width in x after each charge has been moved onto its own median, plotted against momentum and compared with the prediction for scattering in the air alone. Each point is labelled with the thickness of material, in radiation lengths, that its width implies. The horizontal line is the network's own median endpoint error, for scale.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/data_supplement/figures/fig6_scattering_floor.png)

The measured width falls as the inverse of the momentum, from 4,372 µm at 3.4 GeV to 86 µm at 120 GeV, which is the signature of multiple Coulomb scattering: many small random deflections off the nuclei of whatever the particle passes through, each one inversely proportional to the momentum. The line beneath the points is what the 5,177.8 mm of air between the planes would give on its own, 1.703 per cent of a radiation length, where a **radiation length** is the thickness of a material in which an electron loses all but $`1/e`$ of its energy and is the standard way to quote how much a material scatters. The measured width sits a factor of 2.1 to 2.3 above the air-only curve from 5 GeV upwards, which implies an effective thickness of 6.7 to 8.1 per cent of a radiation length between the two planes rather than 1.7 per cent, in other words that the gap is not only air. The 3.4 GeV point implies 15.8 per cent, which we do not read as more material: below about 5 GeV the Gaussian-core approximation used for the air-only curve is known to degrade. The plain statement is that this width is a **floor** and not a defect: it is a random quantity, so no deterministic extrapolator, of any order, with any number of stages, and no network of any size, can predict the individual deflection, and the network's own error, the horizontal line, is already well below it everywhere.

## 4.7 One step

![Figure 7. One step of the chain, 80.9 mm long, taken from the middle of the crossing on the same real 5.86 GeV track as figure 3. The curve is the reference trajectory's departure from a straight line drawn from the state at the start of the step; the two marked points are the stage states inside the step that the network is asked to predict. The vertical axis is in micrometres against a horizontal axis in millimetres, an exaggeration of about 120 times, stated on the axis. The inset magnifies the end of the step 150 times and draws the network's own median single-step error, 0.43 µm, as a bar.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/data_supplement/figures/fig7_one_step.png)

This is what the label-free loss is written about, and it is much smaller than the endpoint numbers suggest. Over one step of 80.90 mm the trajectory departs from a straight line by 178.5 µm for this track, and the network is not asked for that endpoint directly: it is asked for two **stage states**, intermediate states at the two Gauss-Legendre collocation points of the step, which lie at 21.1 and 78.9 per cent of the way across it. A **collocation point** is a place inside the step where we insist the equation of motion holds, and **Gauss-Legendre** says where to put those places so that the resulting scheme is as accurate as possible for the number of points used; the training loss is built only from the residual of the equation at those two points, which is why no measured position enters the picture and no labels are needed. Over all 92,928 single steps of the test set the network's median single-step error is 0.43 µm, against 95.2 µm for a straight line over the same step, both radial. Set that against the 80.6 µm median endpoint error of the same network after 64 steps: the endpoint error is the accumulation of 64 small steps and of the way an early error in the slope is carried the rest of the way, not the consequence of a bad step.

# 5. Conclusion

## 5.1 What the seven figures establish

The reference is exact for the equation it solves, and these figures show the scope of that equation rather than an error in it: figure 3's ledger says that the equation contains bending alone, and figure 4 puts the integrator's own floor more than seven decades below the millimetre-scale quantities being discussed. One part of the remaining gap to the simulation is a defect and will be removed, namely the start momentum, which is the particle's production momentum rather than its momentum at the Upstream Tracker, worth 16 to 18 MeV and worth 4,826 µm of the 5,219 µm gap in the 2 to 5 GeV band; one part is a floor and will not be removed, namely multiple scattering, 647 µm of half-width at 10 to 25 GeV and irreducible for any deterministic method. The network sits well inside both of these, 17 to 335 µm by band against the reference, so the thing that now separates this programme from the simulated truth is the reference, not the network.

## 5.2 What is still open

The decisions that follow from this are set out in the companion write-up, [What the field-only reference shares with the Geant4 truth, and what it cannot](https://app.notion.com/p/3e35d544b9d9818282fff3be2d394009), sections 5.3 and 5.4, and we do not restate them here. In short, they are whether the reference and therefore the training target should acquire an energy-loss term as LHCb's production extrapolator has, what the networks should be scored against in future write-ups, and the identity of the field map used by the simulation, which remains the candidate explanation for the residual above 25 GeV. The first step, re-dumping the truth with the momentum recorded at each hit, is a fix that needs no decision and has not yet been run; everything on this page is measured on the current crossing set, which stays untouched so that every existing network number remains reproducible.

# 6. Provenance

Repository [github.com/GeorgeWilliam1999/LHCb-Extrapolation](https://github.com/GeorgeWilliam1999/LHCb-Extrapolation). **Working tree: the folder `single_network_chain_discrete_approach/data_supplement/` is untracked.** Local HEAD is b84693f and the local reference for `origin/main` is at the same commit, so the seven figures are not on GitHub yet and **the images on this page will render only after George commits and pushes that folder**. No commit hash has been invented for them.

**The figures.** `single_network_chain_discrete_approach/data_supplement/make_figures.py`, run on 22 September 2026 with `PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python make_figures.py`, producing `figures/fig1_crossing_to_scale.png` through `figures/fig7_one_step.png` and `results/figure_facts.json`, which records every number annotated on every figure. The folder `README.md` maps each figure to its inputs.

**Input data.** `Block_E_single_network_chain/E0_Track_dataset/results/tracks.npz`, built by `E0_Track_dataset/build_tracks.py`, holding 14,482 magnet crossings split 11,567 training, 1,463 validation and 1,452 test; it carries the start state at z = 2,648.2 mm, the reference trajectory on 257 planes to z = 7,826.0 mm, and the simulated-truth state on the particle's own first fibre-tracker plane. Upstream of it: `Data_generation_exploration/Official_xdigi/training_v2/train_official_v2.npz`, from the official test-file-database sample `expected_2024_minbias_xdigi`, production 00212966, 200 events, conditions `dddb-20231017` and `sim-20231017-vc-mu100`, data type 2024, magnet up. The detector envelopes of figure 3 come from `Data_generation_exploration/Data/results/states.npz`.

**The network.** `Block_F_reweighted_loss/F1_Training/results/full/N064_q02/chain_states.npz`, the chained states of the network with 64 steps, 2 stages and a step length of 80.9 mm: two hidden layers of 128 units, 18,956 parameters, trained on the 11,567 training crossings with the reweighted loss, 1,000 restarts over 40 rounds, farm cluster 5809660.

**Results files read for the numbers on this page.** `Block_F_reweighted_loss/F2_Analysis/results/against_true_state.csv` (network-to-reference and reference-to-truth endpoint medians, max metric, per band, 1,452 test crossings); `Block_F_reweighted_loss/F3_Analysis/results/comparators.csv` (the straight line, radial); `Block_F_reweighted_loss/F3_Analysis/results/error_qdz_chain.csv` (the exact collocation scheme, radial); `Block_F_reweighted_loss/F3_Analysis/results/error_qdz_single_step.csv` (the single-step medians over 92,928 step pairs, radial); `reference_vs_truth/results/decomposition_by_band.csv` (the per-charge medians, the ratios to the bend, the implied momentum deficit and the charge-corrected 68 per cent half-widths, over all 14,482 crossings); `reference_vs_truth/results/highland_air.csv` (the air-only prediction and the implied radiation lengths); `Data_generation_exploration/Data/results/gates.json` (the reference integrator's own closure and step convergence).

**The reference integrator and the field.** `single_network_chain_discrete_approach/_shared/reference.py`: Butcher's seven-stage Runge-Kutta method of order six at a fixed step of 0.1 mm, in double precision, with the field map `/cvmfs/lhcb.cern.ch/lib/lhcb/DBASE/FieldMap/v8r1/cdf/field.v8r1.up.bin`, md5 `9e49ddc4313b589f273540e7e0bb513b`.

**One correction, recorded.** An earlier version of figure 4 drew the integrator's own floor at 0.0285 µm. The gate file records the 5 mm against 1 mm step convergence as 2.8459 × 10⁻⁸ mm, that is 2.8 × 10⁻⁵ µm or 28 picometres, with a worst case of 2.8 µm over 200 tracks, so the drawn line was a thousand times too high. The cause was a unit slip, the gate being recorded in millimetres and read as micrometres, and the same slip appears in `Data_generation_exploration/Data/README.md`, which calls the gate 28 nanometres. Figure 4 was regenerated on 2026-09-22 and now reads the value from `reference_vs_truth/results/rk6_self_consistency.csv` rather than from a constant typed into the script. No ordering on the figure changed. Separately, the two dashed medians drawn on the left panel of figure 5 are computed on the 1,452 test crossings (−1,267 and +1,061 µm), while the medians quoted in section 4.5 are from the decomposition over all 14,482 crossings (−1,100 and +973 µm); `figure_facts.json` now records both, and each is named where it is used.

**Trust: Provisional.** The provenance was located and checked, and every number above was read from one of the files named here. Only George sets Verified.
