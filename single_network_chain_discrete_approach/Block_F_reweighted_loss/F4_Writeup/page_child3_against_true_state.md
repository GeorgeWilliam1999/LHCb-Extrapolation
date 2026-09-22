<callout icon="🎯">
	Child page of the reweighted-loss write-up. It answers one question put by the project's supervisors: does the endpoint number account for what actually happens to the particle between the two planes? Everything here is the max metric, $`\max(|\Delta x|, |\Delta y|)`$, median over the 1,452 test tracks, at the particle's own first SciFi plane. **The figure renders only after George pushes.**
</callout>
<table_of_contents/>
# 1. Three references, and what each one knows

A crossing has three states at its far end, and they are not the same thing.

1. **The RK6 endpoint.** A sixth-order Runge–Kutta integration of the equation of motion through the field map, at 0.1 mm steps, from the particle's real state on the last Upstream Tracker plane. It knows the magnetic field and **nothing else**. It is what every network is trained towards, without labels, and what every other number in this write-up is scored against.
1. **The Geant4-true SciFi state.** The particle's real state on its own first SciFi plane, taken from the simulated hit: the midpoint of the hit's entry and exit, with slopes from the hit's own displacement divided by its own $`dz`$. It contains **everything the simulation did to the particle** between the two planes, which is multiple scattering and energy loss in the material, and **no detector resolution at all**: there is no digitisation and no pattern recognition in this pipeline. Every "true" state on this page is a simulated hit, never a reconstructed track.
1. **The network endpoint.** The network applied $`N`$ times from the same real last Upstream Tracker state.

The comparison is made at the particle's own SciFi plane. Each run's record has scored its chain against both references since the previous study: the network's state at $`z_1`$ and the RK6 truth are each carried from $`z_1`$ to the particle's own plane with RK6 (a distance of less than 60 mm) and compared with the true state there. The script for this page only tabulates those stored numbers per momentum band and draws them; it recomputes nothing.

# 2. The table

Median $`\max(|\Delta x|, |\Delta y|)`$ at the SciFi plane, in micrometres. The three networks are given in the order N = 64 q = 2, N = 128 q = 8, N = 256 q = 16. From `against_true_state.csv`.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>band</td>
		<td>test tracks</td>
		<td>network against RK6</td>
		<td>network against the true state</td>
		<td>RK6 against the true state</td>
	</tr>
	<tr>
		<td>all</td>
		<td>1,452</td>
		<td>81 / 96 / 84</td>
		<td>1,698 / 1,649 / 1,699</td>
		<td>1,694</td>
	</tr>
	<tr>
		<td>1–2 GeV</td>
		<td>5</td>
		<td>2,214 / 2,419 / 1,577</td>
		<td>13,750 / 15,091 / 17,098</td>
		<td>17,034</td>
	</tr>
	<tr>
		<td>2–5 GeV</td>
		<td>500</td>
		<td>335 / 383 / 316</td>
		<td>5,217 / 5,251 / 5,130</td>
		<td>5,219</td>
	</tr>
	<tr>
		<td>5–10 GeV</td>
		<td>430</td>
		<td>77 / 97 / 79</td>
		<td>1,686 / 1,675 / 1,688</td>
		<td>1,630</td>
	</tr>
	<tr>
		<td>10–25 GeV</td>
		<td>383</td>
		<td>24 / 28 / 36</td>
		<td>623 / 628 / 610</td>
		<td>613</td>
	</tr>
	<tr>
		<td>25–200 GeV</td>
		<td>134</td>
		<td>17 / 20 / 23</td>
		<td>276 / 270 / 274</td>
		<td>255</td>
	</tr>
</table>

![The three references per momentum band, on the 1,452 test tracks. Solid black: the field-only RK6 reference against the Geant4-true SciFi state. Solid coloured: each network against the true state. Dashed: each network against RK6. Median of max(|dx|, |dy|) at the first SciFi plane, logarithmic vertical axis.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/Block_F_reweighted_loss/F2_Analysis/figures/against_true_state.png)

# 3. What it says

**Against the true state, the networks and the field-only reference are indistinguishable.** The network-against-truth median and the RK6-against-truth median agree to within **3.5 %** in every band from 2 to 25 GeV, and to within **8.4 %** in the 25 to 200 GeV band, for all three networks. (The 1 to 2 GeV row holds five tracks and should not be read as a measurement; it scatters from −19 % to +0.4 %.) Exact percentages per band and per network are in `writeup_numbers.json` under `true_state_ratios`.

**The gap to the truth is 11 to 25 times larger than anything the network contributes.** In each band the RK6-against-truth distance is 11.3 to 25.4 times the network-against-RK6 error, and 7.0 to 10.8 times it in the five-track 1 to 2 GeV row. The network's own error is a small perturbation on a much larger physical effect.

**That effect is the material.** The gap scales as $`1/p`$, which is the signature of multiple Coulomb scattering: 5.2 mm at 2 to 5 GeV, 1.6 mm at 5 to 10 GeV, 0.61 mm at 10 to 25 GeV and 0.26 mm at 25 to 200 GeV. No field-only method can reproduce it, RK6 included, and the loss never asked the networks to learn it. The loss is a statement about the equation of motion in a magnetic field, and the equation of motion in a magnetic field is not the whole of what happens to a particle crossing five metres of a detector.

**So the right reading of every other number in this write-up** is that it measures how faithfully the network reproduces the field-only reference, which is the routine it would replace inside the track fit. It does not measure how close the network is to the particle's real trajectory, and it was never intended to. The fit handles material separately, through the noise terms in its own model.

# 4. Caveats, stated rather than buried

- **"True" means the simulated hit**, with scattering and energy loss included and no detector resolution. Reconstructed states, which would add digitisation and pattern-recognition effects, are not in this pipeline at all.
- **The comparison plane is the particle's own**, not $`z_1`$; both the network state and the RK6 truth are carried there by RK6 over less than 60 mm, so a small amount of RK6 enters both sides of the comparison equally.
- **The RK6 reference is not itself perfect against the truth.** The 1.7 mm overall gap is not all multiple scattering: part of it is a charge-antisymmetric systematic, traced to the start state carrying the particle's production momentum rather than its momentum on the last Upstream Tracker plane, a deficit of 16 to 18 MeV. That is a separate investigation with its own to-do and its own write-up ([link to be filled by George]), and the interpretation of the gap belongs there. **It changes none of the numbers in this write-up**, all of which are measured against RK6 on both sides.
- **The max metric is used here**, because that is what the stored chain scores carry, whereas most of this write-up uses the radial metric. The two differ by a factor of order one and are never mixed within a table.

# Sources

`F2_Analysis/against_true_state.py` → `results/against_true_state.csv`, `figures/against_true_state.png`, which tabulates the `metrics.chain_scores` block of each run's `record.json`; percentages and ratios from `F4_Writeup/numbers.py` → `F4_Writeup/results/writeup_numbers.json`. Full provenance is on the parent page.
