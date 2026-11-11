<callout icon="🔬">
	Child page of the reweighted-loss write-up. Everything here comes from the previous study's own analysis scripts, **imported** by a runner and pointed at these three networks, so that every table and plot is the earlier code running on the new weights rather than a retyped copy. All numbers are on the 1,452 test tracks. **Endpoint** means after the full chain of $`N`$ applications, at the first SciFi plane; **single step** means one application from an RK6 state on some intermediate plane; each is named beside its number. **The figures render only after George pushes.**
</callout>
<table_of_contents/>
# 1. One step, in isolation

Each network was applied **once**, from an RK6 state on every one of its start planes, for every test track: 92,928 pairs at $`N = 64`$ and 371,712 at $`N = 256`$. The straight line is the null comparison, the displacement the track would have if the magnet were switched off over that one step, and it says what the step had to get right. From `error_qdz_single_step.csv`.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>network</td>
		<td>single-step median [µm]</td>
		<td>single-step p95 [µm]</td>
		<td>the straight line over one step [µm]</td>
		<td>step as a fraction of it</td>
		<td>single step above 10 GeV [µm]</td>
		<td>single step below 5 GeV [µm]</td>
	</tr>
	<tr>
		<td>N = 64, q = 2, dz = 81 mm</td>
		<td>0.433</td>
		<td>4.92</td>
		<td>95.2</td>
		<td>0.46 %</td>
		<td>0.149</td>
		<td>1.690</td>
	</tr>
	<tr>
		<td>N = 128, q = 8, dz = 40 mm</td>
		<td>0.221</td>
		<td>2.32</td>
		<td>23.8</td>
		<td>0.93 %</td>
		<td>0.066</td>
		<td>0.846</td>
	</tr>
	<tr>
		<td>N = 256, q = 16, dz = 20 mm</td>
		<td>**0.058**</td>
		<td>0.79</td>
		<td>5.96</td>
		<td>0.98 %</td>
		<td>**0.021**</td>
		<td>**0.234**</td>
	</tr>
</table>

A single step is **sub-micrometre** at every step length, and eight to eleven times better in the band than on the soft tracks, which is the reweighting visible at the level of one application. The ratio of the endpoint error to the single-step error is the accumulation factor:

<table fit-page-width="true" header-row="true">
	<tr>
		<td>network</td>
		<td>single step [µm]</td>
		<td>endpoint after N steps [µm]</td>
		<td>endpoint over single step</td>
		<td>steps in the chain</td>
	</tr>
	<tr>
		<td>N = 64, q = 2, dz = 81 mm</td>
		<td>0.433</td>
		<td>88.8</td>
		<td>205</td>
		<td>64</td>
	</tr>
	<tr>
		<td>N = 128, q = 8, dz = 40 mm</td>
		<td>0.221</td>
		<td>104.6</td>
		<td>474</td>
		<td>128</td>
	</tr>
	<tr>
		<td>N = 256, q = 16, dz = 20 mm</td>
		<td>0.058</td>
		<td>92.2</td>
		<td>1,587</td>
		<td>256</td>
	</tr>
</table>

The accumulation factor is **three to six times the number of steps**. Independent per-step errors would give roughly $`\sqrt{N}`$, which is 8, 11 and 16. The errors are not independent: they are correlated along the track and they add. That is what the coherence numbers on the parent page measure directly, and it is the reason a better step does not buy a proportionally better endpoint.

![One step of each network from the RK6 state on every start plane, test tracks, one panel per network. Single-step errors, not endpoint errors.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/Block_F_reweighted_loss/F3_Analysis/figures_writeup/single_step.png)

# 2. How the error grows along the crossing

The chain was stopped at every intermediate plane and scored there. From `error_vs_z.csv` through `numbers.py`.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>network</td>
		<td>at a quarter of the crossing [µm]</td>
		<td>at the half [µm]</td>
		<td>at the end [µm]</td>
		<td>end over half</td>
		<td>half over quarter</td>
	</tr>
	<tr>
		<td>N = 64, q = 2, dz = 81 mm</td>
		<td>9.4</td>
		<td>25.8</td>
		<td>88.8</td>
		<td>3.45</td>
		<td>2.75</td>
	</tr>
	<tr>
		<td>N = 128, q = 8, dz = 40 mm</td>
		<td>11.7</td>
		<td>30.8</td>
		<td>104.6</td>
		<td>3.39</td>
		<td>2.64</td>
	</tr>
	<tr>
		<td>N = 256, q = 16, dz = 20 mm</td>
		<td>10.2</td>
		<td>32.3</td>
		<td>92.2</td>
		<td>2.85</td>
		<td>3.19</td>
	</tr>
</table>

The growth is faster than linear in the distance travelled: the second half of the crossing adds about 2.5 times what the first half did. A random walk would give a factor $`\sqrt{2}`$ and linear accumulation a factor 2. The excess is the lever arm, because a slope error made early keeps converting itself into displacement for the rest of the crossing.

![How the endpoint error grows along the crossing, one panel per network: the median and the 95th percentile of the radial error against the RK6 track, at every intermediate plane, test tracks.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/Block_F_reweighted_loss/F3_Analysis/figures_writeup/error_vs_z.png)

# 3. Over-training, and the comparators

The splits are by particle, so no track appears in two of them. From `overtraining.csv` and `split_comparison.csv`.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>network</td>
		<td>train [µm]</td>
		<td>validation [µm]</td>
		<td>test [µm]</td>
		<td>test over train</td>
		<td>validation over train</td>
	</tr>
	<tr>
		<td>N = 64, q = 2, dz = 81 mm</td>
		<td>83.2</td>
		<td>85.1</td>
		<td>80.6</td>
		<td>0.969</td>
		<td>1.023</td>
	</tr>
	<tr>
		<td>N = 128, q = 8, dz = 40 mm</td>
		<td>97.5</td>
		<td>101.9</td>
		<td>95.5</td>
		<td>0.980</td>
		<td>1.045</td>
	</tr>
	<tr>
		<td>N = 256, q = 16, dz = 20 mm</td>
		<td>84.0</td>
		<td>87.6</td>
		<td>84.4</td>
		<td>1.004</td>
		<td>1.042</td>
	</tr>
</table>

**No over-training.** The held-out error is within 5 % of the training error in every case, and the test split is if anything slightly easier than the training split. That is expected: the loss is label-free, so nothing in the objective can be memorised from a particular track's trajectory. The metric here is the max metric, which is why these numbers differ slightly from the radial medians elsewhere.

The comparators put the endpoint error on a scale. From `error_qdz_chain.csv`, `comparators.csv` and `case_study_headline.csv`.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>what</td>
		<td>median endpoint radial error [µm]</td>
	</tr>
	<tr>
		<td>the exact collocation scheme at N = 64, q = 2, solved without a network</td>
		<td>0.147</td>
	</tr>
	<tr>
		<td>the exact collocation scheme at N = 128, q = 8</td>
		<td>0.011</td>
	</tr>
	<tr>
		<td>the exact collocation scheme at N = 256, q = 16</td>
		<td>0.0011</td>
	</tr>
	<tr>
		<td>**N = 64, q = 2 with the reweighted loss**</td>
		<td>**88.8**</td>
	</tr>
	<tr>
		<td>one network per step, the earlier fixed-step study, same N and q</td>
		<td>2,304</td>
	</tr>
	<tr>
		<td>the material floor: the real SciFi state against the RK6 track</td>
		<td>1,896</td>
	</tr>
	<tr>
		<td>the straight line, no magnet at all</td>
		<td>450,542</td>
	</tr>
</table>

The exact scheme is the ceiling: it is what a network that solved these collocation equations perfectly would reach, and it is a factor 604, 9,100 and 87,800 below where the three networks are. The gap is entirely the optimiser's, not the scheme's. In the other direction, the endpoint error is a factor 5,000 below the straight line and a factor 21 below the material floor.

# 4. The tails

From `tails.csv` and `case_study_tails.csv`.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>network</td>
		<td>median [µm]</td>
		<td>p95 [µm]</td>
		<td>p99 [µm]</td>
		<td>worst [µm]</td>
		<td>fraction beyond 1 mm</td>
		<td>share of the total error carried by the worst 5 %</td>
	</tr>
	<tr>
		<td>N = 64, q = 2, dz = 81 mm</td>
		<td>88.8</td>
		<td>1,399</td>
		<td>4,033</td>
		<td>8,533</td>
		<td>7.2 %</td>
		<td>45.9 %</td>
	</tr>
	<tr>
		<td>N = 128, q = 8, dz = 40 mm</td>
		<td>104.6</td>
		<td>1,611</td>
		<td>4,135</td>
		<td>11,800</td>
		<td>9.6 %</td>
		<td>43.0 %</td>
	</tr>
	<tr>
		<td>N = 256, q = 16, dz = 20 mm</td>
		<td>92.2</td>
		<td>1,230</td>
		<td>3,404</td>
		<td>10,000</td>
		<td>6.5 %</td>
		<td>44.4 %</td>
	</tr>
</table>

About 45 % of the total endpoint error sits in 5 % of the tracks. Those tracks have a name. For N = 64, q = 2 the worst 73 tracks have a median endpoint error of 2,296 µm against 79 µm for the other 1,379; their median momentum is 3.05 GeV against 7.22 GeV; their median pseudorapidity is 2.28 against 3.60; and 82 % of them are below 5 GeV. They are soft tracks at the edge of the acceptance, where $`\eta`$ is small and the track is far from the beam line by the time it reaches the magnet.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>group (N = 64, q = 2)</td>
		<td>tracks</td>
		<td>median endpoint radial error [µm]</td>
		<td>median momentum [GeV]</td>
	</tr>
	<tr>
		<td>η 2.0–2.5</td>
		<td>158</td>
		<td>885</td>
		<td>3.7</td>
	</tr>
	<tr>
		<td>η 2.5–3.0</td>
		<td>255</td>
		<td>286</td>
		<td>4.7</td>
	</tr>
	<tr>
		<td>η 3.0–3.5</td>
		<td>292</td>
		<td>133</td>
		<td>5.4</td>
	</tr>
	<tr>
		<td>η 3.5–4.0</td>
		<td>302</td>
		<td>51</td>
		<td>7.8</td>
	</tr>
	<tr>
		<td>η 4.0–4.5</td>
		<td>249</td>
		<td>28</td>
		<td>12.2</td>
	</tr>
	<tr>
		<td>η 4.5–5.0</td>
		<td>196</td>
		<td>16</td>
		<td>15.4</td>
	</tr>
	<tr>
		<td>positive charge</td>
		<td>739</td>
		<td>84</td>
		<td>6.7</td>
	</tr>
	<tr>
		<td>negative charge</td>
		<td>713</td>
		<td>92</td>
		<td>7.0</td>
	</tr>
</table>

The two charges agree to within 9 %, which is the check that nothing in the training or the field handling is charge-dependent.

# 5. The dependence on where the track starts

Splitting the endpoint error by the track's $`x`$ on the last Upstream Tracker plane, at fixed momentum, separates the two effects. From `case_study_error_vs_p_x0.csv` through `numbers.py`, for the validation-best network (N = 64, q = 2), in the 5 to 7 GeV band where every bin is populated.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>x on the last Upstream Tracker plane [mm]</td>
		<td>tracks</td>
		<td>median |Δx| at z1 [µm]</td>
		<td>median |Δy| at z1 [µm]</td>
	</tr>
	<tr>
		<td>−700 to −400</td>
		<td>6</td>
		<td>183</td>
		<td>522</td>
	</tr>
	<tr>
		<td>−400 to −250</td>
		<td>11</td>
		<td>88</td>
		<td>138</td>
	</tr>
	<tr>
		<td>−250 to −150</td>
		<td>22</td>
		<td>51</td>
		<td>126</td>
	</tr>
	<tr>
		<td>−150 to −75</td>
		<td>24</td>
		<td>30</td>
		<td>85</td>
	</tr>
	<tr>
		<td>**−75 to 0**</td>
		<td>46</td>
		<td>**40**</td>
		<td>86</td>
	</tr>
	<tr>
		<td>**0 to +75**</td>
		<td>48</td>
		<td>59</td>
		<td>**48**</td>
	</tr>
	<tr>
		<td>+75 to +150</td>
		<td>35</td>
		<td>32</td>
		<td>61</td>
	</tr>
	<tr>
		<td>+150 to +250</td>
		<td>18</td>
		<td>54</td>
		<td>91</td>
	</tr>
	<tr>
		<td>+250 to +400</td>
		<td>16</td>
		<td>107</td>
		<td>265</td>
	</tr>
	<tr>
		<td>+400 to +700</td>
		<td>7</td>
		<td>350</td>
		<td>245</td>
	</tr>
</table>

The error is smallest near the beam line and rises by a factor 6 to 9 at the edges, symmetrically in the sign of $`x`$, at **fixed momentum**. So the acceptance-edge effect is not simply the momentum effect in disguise. The same shape appears at 10 to 15 GeV, where the innermost bin is 9.4 µm in $`x`$ and the outermost populated one (six tracks at +400 to +700 mm) is 162 µm. Two things plausibly drive it, and this study does not separate them: the field map is less well sampled far off axis, and a track that far out has a larger $`|t_x|`$, so the same slope error costs more displacement.

![Endpoint error against momentum and against the track's starting x, for the validation-best network. Each cell is a median over the test tracks that fall in it. Endpoint errors, against RK6 at the first SciFi plane.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/Block_F_reweighted_loss/F3_Analysis/figures_writeup/case_study_components_x0_maps.png)

# 6. Forward cost

One full crossing, eager PyTorch in double precision on one CPU thread. These are indicative only: they are not the deployment path, which would be a fused single-precision kernel. From `cost_accuracy.csv`.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>network</td>
		<td>microseconds per track</td>
		<td>parameters</td>
		<td>endpoint median [µm]</td>
	</tr>
	<tr>
		<td>N = 64, q = 2, dz = 81 mm</td>
		<td>**657**</td>
		<td>18,956</td>
		<td>88.8</td>
	</tr>
	<tr>
		<td>N = 128, q = 8, dz = 40 mm</td>
		<td>1,322</td>
		<td>22,052</td>
		<td>104.6</td>
	</tr>
	<tr>
		<td>N = 256, q = 16, dz = 20 mm</td>
		<td>3,655</td>
		<td>26,180</td>
		<td>92.2</td>
	</tr>
</table>

Cost rises by a factor 2.0 and then 2.8 as the step count doubles, while accuracy does not improve, so the cheapest network is also the best in band. If a chained network is ever to be deployed, the long step is the one to pursue.

# 7. The two figures mirrored from the mini paper

These reproduce, for these three networks, the two summary figures of the project's mini paper. Neither maps one to one, because the paper's originals compare seeds and loss variants (its Figure 2) and the two magnet polarities (its Figure 6), and neither of those is varied here. Each panel is the nearest analogue, and the mapping is stated so that nothing is implied by the layout. **Every error in both figures is an endpoint error** except the squares of the per-run panel, which are single-step errors.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>the paper's panel</td>
		<td>what is drawn here</td>
	</tr>
	<tr>
		<td>Figure 2, error histogram (best seed per loss)</td>
		<td>endpoint radial error per network, with the exact-scheme ceiling and the straight line marked</td>
	</tr>
	<tr>
		<td>Figure 2, loss histories to stall</td>
		<td>training loss per restart, per network</td>
	</tr>
	<tr>
		<td>Figure 2, error against momentum</td>
		<td>binned medians of the endpoint error, per network, with the 10–50 GeV band shaded</td>
	</tr>
	<tr>
		<td>Figure 2, per run (open / filled / squares)</td>
		<td>circles are the endpoint median; **squares are the single-step error** from the RK6 state</td>
	</tr>
	<tr>
		<td>Figure 2, the fiducial effect</td>
		<td>endpoint median by band (all, 10–50 GeV, below 5 GeV), one bar per network</td>
	</tr>
	<tr>
		<td>Figure 2, cut trajectories leaving the map</td>
		<td>x(z) of test tracks under the N = 64, q = 2 chain: the worst 5 % in pink, 150 others in dark grey on top</td>
	</tr>
	<tr>
		<td>Figure 6, histogram, both polarities</td>
		<td>endpoint error per network (the validation-best one filled), straight line in grey, ceilings marked, each network's median as a thin line</td>
	</tr>
	<tr>
		<td>Figure 6, scatter against momentum</td>
		<td>every test track for the validation-best network, with the binned median on top</td>
	</tr>
	<tr>
		<td>Figure 6, three tracks, both polarities</td>
		<td>three test tracks of **both charges**, the analogue of the sign check: RK6 reference solid, the network's chain states as markers</td>
	</tr>
</table>

![The three networks on their own, mirroring the mini paper's Figure 2. Six panels: the endpoint error histogram with the exact-scheme ceiling and the straight line marked; the training loss per restart; the endpoint error against momentum with the 10-50 GeV band shaded; per network, the endpoint median (circles) beside the single-step error (squares); the endpoint median by momentum band; and the x(z) trajectories of the worst 5 % of test tracks in pink over 150 others in grey.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/Block_F_reweighted_loss/F3_Analysis/figures/mirror_fig2_baseline.png)

![Mirroring the mini paper's Figure 6. The endpoint error histogram per network with the validation-best one filled; the endpoint error of every test track against momentum for that network, with the binned median on top; and three individual test tracks of both charges, the RK6 reference as a solid line and the network's chain states as markers.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/Block_F_reweighted_loss/F3_Analysis/figures/mirror_fig6_magnet_up.png)

**The charge-sign check, in three tracks.** The third panel of the second figure follows a negatively charged 2.31 GeV track that bends to $`x`$ = +1,288 mm, a positively charged 5.93 GeV track that bends to $`x`$ = −151 mm, and a negatively charged 21.9 GeV one that ends at $`x`$ = −5.4 mm. The network's endpoint is within 0.31 mm, 0.075 mm and 0.0094 mm of the reference for the three. Both signs of charge are handled and the error falls with momentum as it should. From `mirror_mini_paper.json`.

# 8. Two caveats on the reproduced set

1. **Two of the reproduced panels evaluate the previous study's loss on these networks, not the loss they were trained on.** They are the loss column of the over-training check and the case study's loss-against-momentum panel. Neither is used for any claim on these pages: the over-training verdict is read from the chain-error ratios in section 3 (0.969, 0.980, 1.004), and the loss-against-momentum panel is not reproduced here at all, because it would show the **pooled** weighting's view of a reweighted network, with 2 to 5 GeV dominating, which is not what these networks optimised.
1. **The one-sided error anatomy is superseded.** The previous study's anatomy script carries the lever-arm sum for the $`x`$ slope only. It accounted for 73 % of the pooled-loss network's endpoint error but only 38 % of the reweighted one's, because the remaining error is now in the other plane. The two-slope version, which accounts for 97.7 to 99.4 %, is the one used on the parent page.

One presentational note: the right panel of the error-against-step-length figure promises dotted exact-scheme lines, and with one point per step length there is nothing for them to join, so they do not appear. The exact-scheme values are in the table of section 3.

# Sources

`F3_Analysis/results/error_qdz_single_step.csv`, `error_qdz_chain.csv`, `error_vs_z.csv`, `overtraining.csv`, `split_comparison.csv`, `tails.csv`, `comparators.csv`, `case_study_headline.csv`, `case_study_tails.csv`, `case_study_error_vs_p_x0.csv`, `cost_accuracy.csv`, `mirror_mini_paper.json`, `run_log.json`; figures `F3_Analysis/figures_writeup/single_step.png`, `error_vs_z.png`, `case_study_components_x0_maps.png` (redrawn by `F4_Writeup/make_figures.py`) and `F3_Analysis/figures/mirror_fig2_baseline.png`, `mirror_fig6_magnet_up.png`; derived quantities from `F4_Writeup/numbers.py` → `F4_Writeup/results/writeup_numbers.json`. Full provenance is on the parent page.
