<callout icon="🧭">
	This page is the complete record of one experiment: the same chained network as the previous study, at three step lengths, trained with **one change** to the loss. It assumes no prior knowledge: the physics, the loss, the training protocol and every statistic are defined before they are used. Every number comes from a results file in the repository, named in the Provenance section at the bottom. **The figures will be blank until George pushes.** Nothing in this folder is committed yet (working tree uncommitted, local HEAD `1040ee9`, GitHub `main` at `a062bd8`), and the images are served from `main`, so they render only after the commit and push. Trust is **Provisional** until George confirms it.
</callout>
<table_of_contents/>
# 1. Introduction

This section says what the machine is, what the previous experiment measured, and what question this one was built to answer. Section 2 gives the rules that were fixed before any training. Section 3 derives the new loss and states the protocol. Section 4 reports the result. Section 5 says what it does and does not settle, and what the next lever is. Three child pages at the bottom carry the long tables and the figure sets.

## 1.1 The machine, in one page

LHCb is a forward spectrometer at the LHC. A charged particle made in a proton collision leaves hits in the vertex detector, in the four planes of the **Upstream Tracker** just before the magnet, and in the twelve layers of the **scintillating-fibre tracker (SciFi)** just after it. Between the two it crosses a dipole field of about one tesla over five metres, and its path bends by an amount inversely proportional to its momentum. That bend is the momentum measurement.

Reconstructing a track means fitting a trajectory through those hits. The fit works with a **track state** on a plane of constant $`z`$: five numbers $`(x, y, t_x, t_y, q/p)`$, where $`x`$ and $`y`$ are the transverse positions in millimetres, $`t_x = dx/dz`$ and $`t_y = dy/dz`$ are the two slopes (dimensionless), and $`q/p`$ is the signed inverse momentum. Every iteration of the fit has to carry a state from one plane to the next through the field, many times per track and millions of times per event. That routine is the **track extrapolator**. Across the magnet it is the most expensive one in the fit, because it integrates the equation of motion numerically and reads the field map at every step.

This programme asks whether a small neural network can replace that routine across the magnet, trained **without labels**: not by being shown where particles went, but by being required to satisfy the equations of an implicit Runge–Kutta scheme, the discrete-time construction of Raissi, Perdikaris and Karniadakis (2019, section 3).

The crossing is fixed. It runs from $`z_0 = 2{,}648.2`$ mm, the last Upstream Tracker plane, to $`z_1 = 7{,}826.0`$ mm, the first SciFi plane, a distance $`L = 5{,}177.8`$ mm. The crossing is cut into $`N`$ equal steps of length $`\Delta z = L/N`$. **One** network maps a state on any start plane to the state $`\Delta z`$ further along, and a crossing applies that same network $`N`$ times. Inside each step the equation of motion is enforced at $`q`$ **Gauss–Legendre collocation points**, which are $`q`$ fixed positions inside the step chosen so that a polynomial of degree $`2q-1`$ is integrated exactly; the network predicts the state at each of them, and the loss asks that those predicted stage states, put back through the Runge–Kutta reconstruction formula, return the state the network was given. That residual is the whole training signal. It needs no truth trajectory, which is what "label-free" means here.

Two words recur and are worth fixing now.

- **Endpoint error**: the network is applied $`N`$ times starting from a track's real state on the last Upstream Tracker plane, and its state on the first SciFi plane is compared with the reference track carried to the same plane. This is the quantity that matters for the fit.
- **Single-step error**: the network is applied **once**, starting from a reference state on some intermediate plane. It measures the network in isolation, with no accumulation.

The reference throughout is **RK6**: a sixth-order Runge–Kutta integration of the same equation of motion through the same field map, at 0.1 mm steps, from the same start state. It is a field-only reference: it knows the magnetic field and nothing about the material the particle crosses. Section 4.5 and the third child page report what that costs; the separate write-up on the reference itself takes the interpretation further (see [link to be filled by George]).

## 1.2 What the previous study measured

The previous experiment trained this one-network-chained-to-itself design over a grid of step counts and stage counts, with the standard loss: the mean squared reconstruction residual, each of the four output components divided by its **pooled** spread over the training states. Its findings set this experiment up.

- The chains ended at 137 to 246 µm of endpoint radial error, while a **single step** of the same network was under a micrometre. The endpoint error is therefore accumulation, not a weak step.
- The accumulation is made of **slope** errors made early and carried to the end. Decomposing the endpoint error as a sum of each step's own error plus that step's slope error times the distance still to travel accounts for 73 % of it with the $`x`$ slope alone (`anatomy_xy_blockE_N064_q02.json`, `explained.x_slope_only` = 0.733).
- The loss was not looking where the error was. On 8,000 states spread evenly over the planes, **97.5 %** of the pooled loss came from tracks of 2 to 5 GeV, **1.5 %** from the 10 to 50 GeV band that matters for physics, and **0.56 %** from the first quarter of the crossing, which is where an error costs the most (`preflight_shares.csv`).

Neither of those two distributions is what the loss was asked for; they are what a pooled normalisation produces. Hence the question of this experiment.

> **If the loss is asked for the right thing instead, how much of the error goes away?**

## 1.3 Scope and where the code lives

One change is made, to the weights inside the loss, and nothing else. The same tracks and splits, the same network code (imported, not copied), the same seed, the same optimiser protocol, the same number of restarts per round, the same field map. Three step lengths were trained rather than the full grid, because the point is the loss and not the grid.

The code is in `single_network_chain_discrete_approach/Block_F_reweighted_loss/` of the repository `github.com/GeorgeWilliam1999/LHCb-Extrapolation`; the study it follows is `Block_E_single_network_chain/`, and the earlier fixed-step study it descends from is `multi_network_chain_discrete_approach/Block_D_fixed_step_crossing/`. Those folder names are the only place in this page where a block letter appears, because a letter is a filing label and not a description of a network. Networks are named here by what they are: $`N`$ steps, $`q`$ stages, $`\Delta z`$ millimetres. The companion write-up on the same networks under the pooled loss is [link to be filled by George].

# 2. Aims

The rules below were written into the folder README **before** the runs were submitted, and were applied to the pooled-loss runs as well, so that neither side could be favoured by a rule chosen afterwards. They are reproduced verbatim.

> **What would count as a result.** Fixed before the runs, and applied the same way to Block E's existing logs so neither side is favoured. A run has **plateaued** once the median validation error of its last ten rounds has stopped falling, no more than 5 % below the median of the ten before it, for three rounds running; its result is that median with the round-to-round spread beside it.<br><br>**Success**: the N = 64, q = 2 test radial error clears the ±13 % band downward **and** the median per-step slope error falls below its present 0.0016 mrad. Reported overall and in the 10–50 GeV band.<br>**Partial**: in-band improves, overall does not: the window was too hard, and the clamp is the knob.<br>**Null**: nothing clears the band. Then the residual is not the binding constraint, and the next lever is supervising the step Jacobian.<br><br>The ablations (`--weighting no_lever` / `no_track` / `no_window`) are **not** run up front. They are the diagnosis if the combined result is ambiguous; the trainer takes them from the start so they cost farm time and no new code.

Three terms in that paragraph need definitions.

1. **A restart** is one run of the L-BFGS optimiser to its own convergence from the current weights. L-BFGS is a quasi-Newton method: it builds an approximation of the loss surface's curvature from the last few gradient differences and takes a step using it. It converges quickly and then stops, so training proceeds as a sequence of restarts rather than as one long descent.
2. **A round** is 25 consecutive restarts, after which a fresh batch of 32,000 collocation states is drawn and the run is scored on the validation tracks. The round is the unit in which progress is measured, because a single restart's validation number wanders by 10 to 20 %.
3. **The plateau rule** is one-sided on purpose. An earlier two-sided version ("the last ten rounds are within 5 % of the ten before") was tried and discarded: the round-to-round error wanders by 8 to 23 %, so ten-round medians rarely land within 5 % of each other even long after a run has finished. The question that matters for stopping is not whether a run is steady but whether it is **still getting better**, so the rule fires only when the improvement has gone, and it must hold at each of the last three rounds.

Applying that rule to the pooled-loss runs' own round logs was itself a correction. None of the sixteen runs of the previous study had plateaued when training was stopped by hand on 18 September 2026: all eleven with twenty or more rounds were still improving, by 5 to 24 % per ten rounds. Eleven falls out of eleven has probability 0.0005 if the wandering were pure noise, so that is a trend and not scatter. Their published numbers were therefore where training stopped, not where it converges, and their counterparts had to be extended to their own plateau before any comparison could mean anything. The comparison in section 4.3 uses the extended runs.

# 3. Method

This section is the protocol. Subsection 3.1 lists what was held fixed; 3.2 derives the new weight term by term and works a numerical example; 3.3 rules out the obvious alternative; 3.4 and 3.5 are the checks that were run before any training; 3.6 and 3.7 are the runs and the stopping decision; 3.8 defines every statistic used in section 4.

**A note on units, once, for the whole page.** $`t_x`$ and $`t_y`$ are slopes $`dx/dz`$ and $`dy/dz`$, dimensionless, built from a simulated hit's exit minus entry displacement divided by its own $`dz`$. No arctangent is applied anywhere. Every label reading "mrad" in the result files and figures is therefore a **slope difference multiplied by** $`10^3`$, not a true milliradian. For the accepted tracks ($`2 < \eta < 5`$, so $`|t_x| \lesssim 0.27`$) the angle difference is $`\Delta\theta \approx \Delta t_x/(1 + t_x^2)`$, so the two agree to within 7 % at the edge of the acceptance and within 1 % for most tracks. In the text below the correct form, $`\times 10^{-3}`$ (slope), is used. Positions are in micrometres unless millimetres are stated. **Radial** error means $`\sqrt{\Delta x^2 + \Delta y^2}`$; **max metric** means $`\max(|\Delta x|, |\Delta y|)`$; the metric is named beside every number.

## 3.1 What is kept fixed

<table fit-page-width="true" header-row="true">
	<tr>
		<td>what is held fixed</td>
		<td>value</td>
	</tr>
	<tr>
		<td>tracks</td>
		<td>14,482 crossings from the official simulated sample, split by particle into 11,567 train / 1,463 validation / 1,452 test</td>
	</tr>
	<tr>
		<td>network</td>
		<td>the previous study's model module, **imported** rather than copied, so it cannot drift; two hidden layers of 128 units</td>
	</tr>
	<tr>
		<td>seed</td>
		<td>0, the same one</td>
	</tr>
	<tr>
		<td>optimiser protocol</td>
		<td>L-BFGS, 25 restarts per round, the same trainer with the loss function swapped</td>
	</tr>
	<tr>
		<td>states per round</td>
		<td>32,000 collocation states, redrawn each round</td>
	</tr>
	<tr>
		<td>field map</td>
		<td>v8r1, magnet up, the same file</td>
	</tr>
	<tr>
		<td>reference</td>
		<td>RK6 at 0.1 mm from the same start state</td>
	</tr>
</table>

The claim that nothing else changed is checkable rather than asserted, in two ways. First, `F1_Training/results/train_weighted.diff` is the literal diff of the new trainer against the old one. Second, and stronger, the trainer carries a mode `--weighting blockE` that restores the previous weights, and **gate F-1** requires that in this mode it reproduce the previous training bit for bit:

```bash
PY=/data/bfys/gscriven/conda/envs/TE/bin/python; export PYTHONNOUSERSITE=1
SCR=$(mktemp -d)
(cd ../../Block_E_single_network_chain/E1_Network_grid \
   && $PY train_network.py  --N 64 --q 2 --n-train 300 --n-eval 200 --stop-after 3 --out $SCR/blockE)
$PY train_weighted.py --N 64 --q 2 --weighting blockE --n-train 300 --n-eval 200 --stop-after 3 --out $SCR/blockF
diff <(cut -d, -f5,6 $SCR/blockE/N064_q02/history.csv) \
     <(cut -d, -f5,6 $SCR/blockF/blockE/N064_q02/history.csv) && echo "gate F-1 passes"
```

The diff is empty: identical losses to the last digit over three restarts (1.0011e-4, then 8.6855e-8, 3.2594e-8, 1.9375e-8). In that mode the trainer calls the shared loss itself, so the reproduction is exact by construction; what proves that the **weighted** code path agrees with the shared one numerically is gate 2 of section 3.4.

## 3.2 The weight, derived

Write $`r_{n,j,d}`$ for the reconstruction residual: the mismatch, for training state $`n`$, output plane $`j`$ and component $`d \in \{x, y, t_x, t_y\}`$, between the state the Runge–Kutta formula rebuilds from the network's stage predictions and the state the network was actually given. The previous loss was

$$
\mathcal{L}_{\text{pooled}} = \operatorname*{mean}_{n,j,d} \left( \frac{r_{n,j,d}}{s_d} \right)^2 ,
$$

where $`s_d`$ is the pooled spread of component $`d`$ over the training states. That asks for equal **absolute** accuracy in every component, on every track, at every plane, and it gets it: the measured per-step slope error is flat along the crossing. The two consequences quoted in section 1.2 follow directly. A soft track bends more, so it has a larger residual in absolute terms and takes over the loss; and a slope residual counts the same as a position residual, although a slope error made at the first plane is multiplied by more than five metres before it reaches the SciFi plane and a position error is not.

The new loss measures every residual as **the displacement it would cause at the SciFi plane, as a fraction of that track's own total bend**, with the 10 to 50 GeV band weighted up:

$$
\mathcal{L} = \operatorname*{mean}_{n,j,d} \left( \frac{a_n \, e_{n,j,d}}{D_{\text{ref}}} \right)^2 ,
$$

$$
e_x = r_x, \quad e_y = r_y, \quad e_{t_x} = r_{t_x}\,\ell_{n,j}, \quad e_{t_y} = r_{t_y}\,\ell_{n,j} \quad [\text{mm}] ,
$$

where each factor is defined and justified below.

**The lever arm** $`\ell_{n,j}`$.

$$
\ell_{n,j} = z_1 - (z_{\text{start},n} + c_j\,\Delta z) + \Delta z ,
$$

where $`z_{\text{start},n}`$ is the start plane of state $`n`$, $`c_j \in [0,1]`$ is the position of collocation point $`j`$ inside the step, and $`\Delta z`$ is the step length. It is the distance the output still has to travel, **plus one step**. Multiplying a slope residual by it turns a slope into the millimetres of displacement that slope will have produced by the time the track reaches $`z_1`$, so a slope residual and a position residual are finally in the same units and can be added.

The extra $`\Delta z`$ is **additive, not a floor**, and that choice matters. Without it the lever arm would be exactly zero at the last plane of the crossing, and the loss would place no weight at all on the final step, which is wrong: an error made there still lands in the fit. Adding one step makes the last plane's weight small but finite, and leaves every other plane essentially unchanged, because $`\Delta z`$ is 1.6 % of $`L`$ at $`N = 64`$ and less at the shorter steps. At $`N = 64`$ the lever arm runs from 80.9 mm at the last plane to 5,258.7 mm at the first, a range of 65.

**The track factor** $`a_n`$.

$$
a_n = \sqrt{W(p_n)}\;\frac{D_{\text{ref}}}{D_n}, \qquad D_n = \kappa\,\left|q/p\right|_n\,\bar{I}\,L ,
$$

where $`D_n`$ is the track's **total transverse bend** across the whole crossing, $`\kappa = 10^{-3}`$ is the unit conversion of the equation of motion, $`\bar{I} = 3{,}752.05`$ T·mm is the field integral along the $`z`$ axis over the crossing, and $`D_{\text{ref}}`$ is the median of $`D_n`$ over the run's first-round states (861.56 mm at $`N = 64`$, $`q = 2`$). Dividing by $`D_n`$ converts an absolute residual into a **relative** one: an error of a given size counts for more on a track that barely bends than on one that bends a lot. That is what removes the momentum skew, because $`D_n \propto 1/p`$.

$`D_n`$ is deliberately the bend over the **whole crossing**, a constant of the track, and not the field integral of the particular step. Dividing by the per-step integral would ask for constant relative accuracy at every $`z`$, which in absolute terms would make the high-field middle of the magnet worse, and the middle is exactly where the bending happens. $`D_{\text{ref}}`$ is likewise a constant of the run, fixed from the first round, so that the objective does not drift between rounds; it only sets the overall scale of the loss, which the trainer rescales anyway.

**The momentum window** $`W(p)`$. A smooth top hat, equal to 1 on 10 to 50 GeV and falling off as a Gaussian in $`\log p`$ outside it, with a floor:

$$
W(p) = \max\!\left( 0.05,\; \exp\!\left[-\left(\frac{\log(p/p_{\text{edge}})}{\log 2}\right)^{2}\right] \right) ,
$$

with $`p_{\text{edge}} = 10`$ GeV below the band and 50 GeV above, and $`W = 1`$ inside. The roll-off constant $`\log 2`$ means the window falls by a factor $`1/e`$ a factor of two outside the band. It is a window and not a cut because one network still has to work everywhere: $`W`$ = 0.05 at 1 and 2 GeV, 0.368 at 5 GeV, 1 at 10, 20 and 50 GeV, 0.368 at 100 GeV and 0.05 at 200 GeV. The square root in $`a_n`$ is there because $`a_n`$ is squared in the loss, so $`\sqrt{W}`$ makes $`W`$ itself the weight on the squared residual.

**The clamp.** $`a_n`$ scales as $`|q/p|`$, which spans about a factor 200 across the sample and 40,000 once squared. Left alone, the 29 tracks above 100 GeV would take a fifth of the loss. So $`a_n`$ is clamped to $`[1/5,\,5]`$ of its median over the round's own batch. The threshold moves slightly between rounds because the batch does; $`D_{\text{ref}}`$ does not. Section 3.5 is the scan that chose the factor 5.

Both factors are functions of the input state and the field map alone. Neither reads a truth trajectory, so **the loss stays label-free**, which is the entire point of the construction.

**A worked example.** Take the $`N = 64`$, $`q = 2`$ run, where $`\Delta z`$ = 80.9 mm and $`D_{\text{ref}}`$ = 861.56 mm, and compare a 3 GeV track with a 20 GeV one. The 3 GeV track bends $`D_n`$ = 1,941 mm across the crossing and sits outside the window, so $`W`$ = 0.05 (the floor) and $`a_n = \sqrt{0.05} \times 861.56/1941 = 0.0992`$. The 20 GeV track bends $`D_n`$ = 291 mm and sits inside the window, so $`W`$ = 1 and $`a_n = 861.56/291 = 2.959`$, a factor 29.8 larger. Now place each of them at a plane. A slope residual on the 20 GeV track at the **first** output plane carries a weight $`a_n \ell / D_{\text{ref}}`$ = 17.8, against 0.596 for the same residual on the 3 GeV track at the same plane (the 29.8 again), against 0.278 for the 20 GeV track at the **last** plane (a factor 64 from the lever arm alone), and against 0.0093 for the 3 GeV track at the last plane. The extremes differ by a factor **1,908**. Under the pooled loss all four would have counted the same. (These are the unclamped values, from `numbers.py`; the clamp then trims the tails of the $`a_n`$ distribution.)

## 3.3 Why not weight by the field

The obvious alternative is to weight up the strong-field region, where the bending happens. It is wrong, and measurably so. Over the 64 steps of the crossing, using the previous study's own step-by-step error anatomy as the input (`field_correlations.py` → `field_correlations.json`, magnitude of the field on axis running from 0.221 T to 1.048 T):

<table fit-page-width="true" header-row="true">
	<tr>
		<td>correlation over the 64 steps</td>
		<td>Spearman</td>
		<td>Pearson</td>
	</tr>
	<tr>
		<td>per-step slope error against field magnitude</td>
		<td>−0.575</td>
		<td>−0.623</td>
	</tr>
	<tr>
		<td>a step's cost at the SciFi plane against the distance it still has to travel</td>
		<td>**+0.996**</td>
		<td>+0.982</td>
	</tr>
	<tr>
		<td>a step's cost at the SciFi plane against field magnitude</td>
		<td>+0.124</td>
		<td>−0.018</td>
	</tr>
</table>

The per-step slope error is **worst where the field is weakest**, at the two low-field ends, not in the magnet's core. And what a step's error costs at the SciFi plane follows the distance left to travel almost perfectly and the field hardly at all. The lever arm is therefore the right variable and the field is not. This is also the reason $`D_n`$ uses the bend over the whole crossing rather than the field integral of its own step.

## 3.4 The gates, before any training

Four gates were run before a single network was trained (`check_weights.py` → `check_weights.json`, `all_pass: true`).

<table fit-page-width="true" header-row="true">
	<tr>
		<td>gate</td>
		<td>what it proves</td>
		<td>result</td>
	</tr>
	<tr>
		<td>1</td>
		<td>the torch weights equal an independent numpy implementation of the same formulas, and the lever arm runs from one step at the last plane to $`L + \Delta z(1 - c_1)`$ at the first</td>
		<td>worst relative difference 1.0e-15; the lever arm range is exact</td>
	</tr>
	<tr>
		<td>2</td>
		<td>in mode `blockE` the weighted code path reproduces the shared pooled loss</td>
		<td>relative difference 0.0, identical to the last bit</td>
	</tr>
	<tr>
		<td>3</td>
		<td>the reweighting does not move the minimum: collocation states solved exactly, without a network, give machine zero under **every** mode</td>
		<td>6e-30 to 2e-29 of the straight line's loss</td>
	</tr>
	<tr>
		<td>4</td>
		<td>the pre-flight: where the loss comes from, and whether it looks where the error is</td>
		<td>section 3.5</td>
	</tr>
</table>

Gate 3 is the one worth dwelling on. A reweighting changes how much each residual counts, but it must not change **what a zero residual is**; otherwise the network would be trained towards a different solution rather than towards the same one with a different emphasis. Solving the collocation equations exactly and confirming that the loss is machine zero under every weighting mode is the statement that the minimum has not moved.

Gate 1 caught a real difference on its first run. `torch.median` returns the lower of the two middle values while `numpy.median` averages them, which moved the clamp threshold between the two implementations. Both now use the 0.5 quantile, which the two libraries agree on. This is the kind of defect that would never have shown up as a crash.

## 3.5 The pre-flight, and the clamp it chose

Gate 4 asks the practical question: with this weight, where does the loss actually come from? It is measured on the previous study's trained $`N = 64`$, $`q = 2`$ network with 8,000 (track, plane) states drawn evenly over the planes. "Cost" is what a state's error actually incurs at the SciFi plane, namely its local error against RK6 taken from the same state, position plus slope times the distance left. That reference is a diagnostic only and never enters the loss.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>weighting</td>
		<td>2–5 GeV</td>
		<td>10–50 GeV</td>
		<td>above 50 GeV</td>
		<td>first quarter of z</td>
		<td>last quarter</td>
		<td>ρ(share, cost) in 10–50 GeV</td>
	</tr>
	<tr>
		<td>the pooled loss</td>
		<td>97.5 %</td>
		<td>1.5 %</td>
		<td>0.01 %</td>
		<td>0.56 %</td>
		<td>6.2 %</td>
		<td>+0.559</td>
	</tr>
	<tr>
		<td>**the new loss (full)**</td>
		<td>29.2 %</td>
		<td>**57.8 %**</td>
		<td>6.7 %</td>
		<td>**21.2 %**</td>
		<td>1.7 %</td>
		<td>**+0.889**</td>
	</tr>
	<tr>
		<td>lever off</td>
		<td>24.2 %</td>
		<td>61.7 %</td>
		<td>6.4 %</td>
		<td>9.7 %</td>
		<td>21.2 %</td>
		<td>+0.552</td>
	</tr>
	<tr>
		<td>track factor off</td>
		<td>82.4 %</td>
		<td>11.3 %</td>
		<td>0.27 %</td>
		<td>6.9 %</td>
		<td>0.97 %</td>
		<td>+0.951</td>
	</tr>
	<tr>
		<td>window off</td>
		<td>71.8 %</td>
		<td>22.2 %</td>
		<td>3.1 %</td>
		<td>9.4 %</td>
		<td>0.87 %</td>
		<td>+0.881</td>
	</tr>
</table>

Read across the first two rows. The pooled loss is 97.5 % about 2 to 5 GeV tracks and puts 0.56 % of its attention on the first quarter of the crossing, where an error costs the most. The new weighting moves that to 57.8 % in the band that matters and 21.2 % in the first quarter, and inside that band it ranks states by their true endpoint cost at +0.889 rather than +0.559.

The **overall** rank correlation falls, from +0.690 to +0.561, and that is expected rather than a defect: the network's largest errors today are on the 2 to 5 GeV tracks, which this weighting deliberately stops chasing. The last three rows are the ablations, shown here only as pre-flight diagnostics; they were not trained (section 5).

**The clamp scan.** The clamp factor was the one free number, and this gate set it.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>clamp factor</td>
		<td>2–5 GeV</td>
		<td>10–50 GeV</td>
		<td>above 50 GeV</td>
	</tr>
	<tr>
		<td>off</td>
		<td>14.8 %</td>
		<td>60.9 %</td>
		<td>19.8 %</td>
	</tr>
	<tr>
		<td>2</td>
		<td>80.9 %</td>
		<td>15.3 %</td>
		<td>0.54 %</td>
	</tr>
	<tr>
		<td>3</td>
		<td>59.2 %</td>
		<td>33.7 %</td>
		<td>2.0 %</td>
	</tr>
	<tr>
		<td>**5 (chosen)**</td>
		<td>**29.2 %**</td>
		<td>**57.8 %**</td>
		<td>**6.7 %**</td>
	</tr>
	<tr>
		<td>10</td>
		<td>15.0 %</td>
		<td>61.8 %</td>
		<td>18.6 %</td>
	</tr>
</table>

Five buys nearly all of the in-band weight while keeping the tail above 50 GeV below 7 %. A tighter clamp squeezes the weight back into the soft tracks; a looser one hands a fifth of the loss to 29 tracks.

**One thing the weighting does not fix.** The loss stays very concentrated: the top 1 % of states carry **83.8 %** of it, against 97.9 % under the pooled loss. That is the heavy tail of the residual distribution itself, not the weighting, and no choice of weight removes it. It is worth remembering when reading anything that depends on a batch mean.

## 3.6 Training and the farm

Three runs were trained with the full weighting, chosen to span the step length at roughly constant work per crossing: $`N = 64`$ with $`q = 2`$ ($`\Delta z`$ = 80.9 mm), $`N = 128`$ with $`q = 8`$ (40.5 mm) and $`N = 256`$ with $`q = 16`$ (20.2 mm). Runs land in `results/<weighting>/N<NNN>_q<qq>/`, so an ablation can never overwrite a real run.

All three were submitted to the batch farm on 18 September 2026 as **cluster 5809660**, with caps of 40 rounds and 1,000 restarts, which are the same number at 25 restarts a round. The caps were set generously so that the plateau rule and not the cap would decide. Jobs are resumable after every restart; a resubmission script returns anything that leaves the queue unfinished, and a keeper process releases the farm's wall-time and memory holds every half hour and stops resubmitting a run once the plateau rule holds.

In the event the two shorter runs reached the 1,000-restart cap (19 and 20 September) and the rule held at that point, so the cap ended them and the rule agreed. The third was reopened with an extension to 1,500 restarts and 60 rounds and finished on 22 September at 10:07 under **cluster 5823724**, with a duplicate job, **cluster 5833416**, training the same run at the same time from restart 1,198 onward. The duplicate arose from a fault in the resubmission script, which read an empty queue when the queue query failed and sent a second copy. Both jobs ran to the cap, so nothing was lost, but that run's `history.csv` holds 1,807 rows for 1,500 restarts (302 restart numbers written twice) and its `rounds.csv` holds rounds 48 to 60 twice. Any reading of that run must first **de-duplicate by number, keeping the last row written**, which is what the convergence script of section 3.7 does; it records what it dropped. The two shorter runs have no duplicates.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>network</td>
		<td>restarts</td>
		<td>rounds</td>
		<td>median wall per restart</td>
		<td>recorded training wall</td>
	</tr>
	<tr>
		<td>N = 64, q = 2, dz = 80.9 mm</td>
		<td>1,000</td>
		<td>40</td>
		<td>77.7 s</td>
		<td>21.6 h</td>
	</tr>
	<tr>
		<td>N = 128, q = 8, dz = 40.5 mm</td>
		<td>1,000</td>
		<td>40</td>
		<td>118.3 s</td>
		<td>32.8 h</td>
	</tr>
	<tr>
		<td>N = 256, q = 16, dz = 20.2 mm</td>
		<td>1,500</td>
		<td>60</td>
		<td>184.1 s</td>
		<td>100.4 h</td>
	</tr>
</table>

The per-restart medians are over the de-duplicated history. For the longest run the two columns do not multiply out: 1,500 restarts at 184.1 s is 76.7 h, and the sum of the de-duplicated per-restart times is 81.1 h, against the 100.4 h its record reports. The difference is the duplicate job's own work, which the record counts and the de-duplicated history does not. Training cost therefore grows steeply with $`N`$, roughly as the number of steps times the number of stages.

## 3.7 Stopping

All three runs are flat by the pre-registered rule. The verdicts below are from `convergence.py` rerun on the finished weights on 22 September 2026, into `F4_Writeup/results/convergence_check.csv`, with the longest run's interleaved history de-duplicated as described above.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>network</td>
		<td>rounds</td>
		<td>flat by the rule</td>
		<td>first round the rule held</td>
		<td>last ten rounds against the previous ten</td>
	</tr>
	<tr>
		<td>N = 64, q = 2, dz = 80.9 mm</td>
		<td>40</td>
		<td>yes</td>
		<td>35</td>
		<td>+2.8 %</td>
	</tr>
	<tr>
		<td>N = 128, q = 8, dz = 40.5 mm</td>
		<td>40</td>
		<td>yes</td>
		<td>35</td>
		<td>+0.5 %</td>
	</tr>
	<tr>
		<td>N = 256, q = 16, dz = 20.2 mm</td>
		<td>60</td>
		<td>yes</td>
		<td>60</td>
		<td>+0.5 %</td>
	</tr>
</table>

A positive percentage means the last ten rounds were slightly **worse** than the ten before, which is what a plateau looks like through round-to-round noise.

![Validation error at the first SciFi plane, round by round (red), with the ten-round running median the headline is read from (blue). One panel per network; the horizontal axis is the training round, and the longest run's history has been de-duplicated, so it ends at round 60. Endpoint metric, validation tracks.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/Block_F_reweighted_loss/F4_Writeup/figures/convergence_check.png)

The lesson behind the rule is worth stating plainly, because it cost this programme a week. **Convergence is never called from the loss.** In the previous study the training loss went flat while the validation error was still falling 5 to 24 % per ten rounds, in eleven runs out of eleven. The loss and the held-out error are not the same signal, and here the loss keeps improving by more than 1 % per restart long after the validation error has stopped moving.

![Training loss (blue, left axis) and validation error at the first SciFi plane (red, right axis) against L-BFGS restart, one panel per network. The saw teeth in the loss are the round boundaries, where a fresh batch of collocation states is drawn. The right-hand panel's axis runs past 1,500 because it plots the raw history rows, including the duplicate job's.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/Block_F_reweighted_loss/F3_Analysis/figures_writeup/convergence_grid.png)

## 3.8 Scoring, and what each statistic means

Every network was scored twice over: once by the comparison scripts written for this study, and once by the full analysis set of the previous study, reproduced here by a runner that **imports** each of those scripts and points its input and output paths at these runs, so that every plot is the earlier code running on these networks rather than a retyped copy of it. Four of those scripts lay their panels out over a full grid of step counts and stage counts; these three runs sit on a diagonal, so a companion module holds the same computation with one panel per run.

The statistics used in section 4 and in the child pages are these.

- **Endpoint radial median**: the median over the 1,452 test tracks of $`\sqrt{\Delta x^2 + \Delta y^2}`$ at the first SciFi plane, after the network has been applied $`N`$ times from the track's real last Upstream Tracker state, against the RK6 track carried to the same plane.
- **In-band**: the same, restricted to test tracks with 10 GeV ≤ p < 50 GeV (487 of the 1,452, 33.5 %).
- **p95**: the 95th percentile of the same distribution, which is where the heavy tail shows.
- **Validation headline**: the median validation endpoint error of a run's last ten rounds, with the round-to-round spread beside it. Not the final checkpoint, which wanders by 10 to 20 %.
- **Per-step slope error**: the median over steps and tracks of the network's own slope error in one step, measured against RK6 taken from the same state the network was given at that step. It is the per-step quantity the accumulation is built from.
- **Signed median**: the median of the signed deviation rather than of its magnitude. A signed median much smaller than the median magnitude means the typical track is missed in either direction, not bent systematically one way.
- **68 % half-width**: half the distance between the 16th and 84th percentiles of the signed deviation. For a Gaussian it equals the standard deviation, but unlike the standard deviation and the RMS it is not dragged by a few outliers. Where the two disagree, both are reported.
- **Coherence**: the ratio of the magnitude of the **sum** of the per-step slope errors along a track to the sum of their magnitudes, normalised so that $`1/\sqrt{N}`$ would be the value for $`N`$ independent random errors. A value well above that means the steps err in the same direction and their errors add rather than cancel.
- **Explained fraction**: how much of the measured endpoint error a lever-arm model reproduces, where the model is the sum over steps of each step's own position error plus its slope error times the distance left. If the fraction is near 1, the decomposition is complete and the endpoint error really is accumulated slope error.

# 4. Results

Every number in this section is from the finished weights of 22 September 2026, on the 1,452 test tracks, endpoint metric against RK6 unless stated otherwise. Section 4.1 gives the three networks; 4.2 answers the pre-registered question; 4.3 says where the gain came from and what it cost; 4.4 takes the error apart. The long tables and the figure sets are in the three child pages listed at the end.

## 4.1 The three networks

<table fit-page-width="true" header-row="true">
	<tr>
		<td>network</td>
		<td>validation headline (median of the last 10 rounds ± spread)</td>
		<td>test radial median</td>
		<td>10–50 GeV</td>
		<td>below 5 GeV</td>
		<td>p95</td>
		<td>per-step slope error, tx / ty</td>
		<td>training wall</td>
	</tr>
	<tr>
		<td>N = 64, q = 2, dz = 80.9 mm</td>
		<td>89.3 µm ± 12 %</td>
		<td>**88.8 µm**</td>
		<td>**24.8 µm**</td>
		<td>373 µm</td>
		<td>1,399 µm</td>
		<td>0.00073 / 0.00097</td>
		<td>21.6 h</td>
	</tr>
	<tr>
		<td>N = 128, q = 8, dz = 40.5 mm</td>
		<td>107.0 µm ± 11 %</td>
		<td>104.6 µm</td>
		<td>27.3 µm</td>
		<td>446 µm</td>
		<td>1,611 µm</td>
		<td>0.00038 / 0.00050</td>
		<td>32.8 h</td>
	</tr>
	<tr>
		<td>N = 256, q = 16, dz = 20.2 mm</td>
		<td>86.4 µm ± 26 %</td>
		<td>92.2 µm</td>
		<td>34.6 µm</td>
		<td>357 µm</td>
		<td>1,230 µm</td>
		<td>0.00019 / 0.00024</td>
		<td>100.4 h</td>
	</tr>
</table>

Positions are micrometres, radial, at the first SciFi plane; the slope errors are per step, in units of $`10^{-3}`$ (slope), and are medians over steps and test tracks.

Two things are visible at once. The per-step slope error falls almost exactly in proportion to the step length, by roughly a factor 2 for each halving of $`\Delta z`$, which is what a well-trained step should do. The **endpoint** error does not follow it: the shortest step is not the best network overall, and is the worst in band. Shorter steps mean more of them, and what accumulates is the number of steps times the error each makes. Section 4.4 makes that explicit.

## 4.2 The two pre-registered criteria

For the pre-registered network, $`N = 64`$, $`q = 2`$, **both criteria hold**.

1. **Test radial error clears the band downward.** 88.8 µm against 145.7 µm for the same network trained under the pooled loss and extended to its own plateau, a fall of **39 %**, well outside that run's ±17 % round-to-round band.
2. **Median per-step slope error below 0.0016.** It is **0.00073** (in units of $`10^{-3}`$ slope), a factor 2.2 below the threshold, and the counterpart's is 0.00130.

In the 10 to 50 GeV band the endpoint error falls from 108.1 µm to **24.8 µm**, a factor 4.4.

The price was also pre-registered, as the "partial" outcome, and it was paid in two places: below 5 GeV the error is 1.5 times worse (373 µm against 242), and the 95th percentile is 1.6 times worse (1,399 µm against 898). Section 4.3 and the first child page give the full breakdown.

## 4.3 Where the gain came from, and what it cost

This is the one comparison the study exists to make, so it is shown as a table. The counterpart in each pair is **the same network under the pooled loss**, trained to its own plateau by the same rule. Median magnitude of the endpoint error, with the signed median beside it, on the 1,452 test tracks.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>N = 64, q = 2, dz = 80.9 mm</td>
		<td>x [µm]</td>
		<td>y [µm]</td>
		<td>radial [µm]</td>
		<td>radial, 10–50 GeV [µm]</td>
	</tr>
	<tr>
		<td>reweighted loss, median magnitude</td>
		<td>**37.5**</td>
		<td>55.2</td>
		<td>**88.8**</td>
		<td>**24.8**</td>
	</tr>
	<tr>
		<td>the same network under the pooled loss</td>
		<td>110.7</td>
		<td>52.3</td>
		<td>145.7</td>
		<td>108.1</td>
	</tr>
	<tr>
		<td>reweighted loss, signed median (in band)</td>
		<td>−3.6 (−6.3)</td>
		<td>+2.9 (+2.0)</td>
		<td></td>
		<td></td>
	</tr>
	<tr>
		<td>the same network under the pooled loss, signed median (in band)</td>
		<td>−39.4 (−54.4)</td>
		<td>−15.5 (−22.8)</td>
		<td></td>
		<td></td>
	</tr>
</table>

The whole gain is in $`x`$, and $`x`$ is the bending plane. The error in $`x`$ falls by a factor 3.0, from 110.7 to 37.5 µm; the error in $`y`$ does not move (52.3 to 55.2 µm, 6 % worse). The signed medians are the sharper statement: the pooled-loss network carries a real offset in the band, −54 µm in $`x`$ and −23 µm in $`y`$, meaning it systematically under-bends; the reweighted one carries −6 µm and +2 µm, which is no offset at all. The reweighting removed a bias, not just a width.

The same pattern holds at the other two step lengths. At $`N = 128`$, $`q = 8`$: radial 104.6 against 122.2 µm (−14 %, only just outside that counterpart's ±12 % band, and that run was still falling 7 % per ten rounds when it was stopped), in band 27.3 against 70.5 µm (−61 %), $`x`$ 51.3 against 81.7, $`y`$ 61.2 against 59.2. At $`N = 256`$, $`q = 16`$: radial 92.2 against 147.7 µm, in band 34.6 against 89.6, $`x`$ 37.6 against 103.4, $`y`$ 61.3 against 69.7.

**The price, by momentum.** Radial median per band, the two losses side by side, for $`N = 64`$, $`q = 2`$:

<table fit-page-width="true" header-row="true">
	<tr>
		<td>band</td>
		<td>test tracks</td>
		<td>reweighted loss [µm]</td>
		<td>the same network under the pooled loss [µm]</td>
	</tr>
	<tr>
		<td>1–2 GeV</td>
		<td>5</td>
		<td>2,510</td>
		<td>681</td>
	</tr>
	<tr>
		<td>2–5 GeV</td>
		<td>500</td>
		<td>370</td>
		<td>241</td>
	</tr>
	<tr>
		<td>5–10 GeV</td>
		<td>430</td>
		<td>84</td>
		<td>122</td>
	</tr>
	<tr>
		<td>10–20 GeV</td>
		<td>317</td>
		<td>**29**</td>
		<td>85</td>
	</tr>
	<tr>
		<td>20–50 GeV</td>
		<td>170</td>
		<td>**16**</td>
		<td>145</td>
	</tr>
	<tr>
		<td>50–100 GeV</td>
		<td>24</td>
		<td>20</td>
		<td>136</td>
	</tr>
	<tr>
		<td>100–200 GeV</td>
		<td>6</td>
		<td>51</td>
		<td>315</td>
	</tr>
</table>

The crossover is at about 5 GeV. Above it the reweighted network is three to nine times better; below it, one and a half to four times worse. That is precisely what the pre-flight predicted when it moved 68 percentage points of the loss out of the 2 to 5 GeV band, and it is the trade that was accepted in advance. The 1 to 2 GeV row holds five tracks and should be read as an indication only.

## 4.4 What the remaining error is made of

Decomposing the endpoint error as a sum over steps of each step's own position error plus its slope error times the distance left, with **both** slopes included, reproduces the measured endpoint error to 97.7 to 99.4 % for all three networks. The decomposition is therefore complete: the endpoint error is accumulated per-step error and nothing else.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>N = 64, q = 2, dz = 80.9 mm</td>
		<td>per-step tx / ty, all tracks</td>
		<td>per-step tx / ty, 10–50 GeV</td>
		<td>x part / y part at z1 [µm]</td>
		<td>coherence tx / ty</td>
	</tr>
	<tr>
		<td>reweighted loss</td>
		<td>0.00073 / 0.00097</td>
		<td>0.00023 / 0.00027</td>
		<td>36 / 55</td>
		<td>0.49 / 0.52</td>
	</tr>
	<tr>
		<td>the same network under the pooled loss</td>
		<td>0.00130 / 0.00072</td>
		<td>0.00095 / 0.00040</td>
		<td>108 / 52</td>
		<td>0.36 / 0.40</td>
	</tr>
</table>

Slope errors are in units of $`10^{-3}`$ (slope); the reference value for independent steps is 0.125 at $`N = 64`$.

Read it this way. The reweighted network's per-step $`x`$ slope is 1.8 times better overall and 4.1 times better in the band. Its per-step $`y`$ slope is 1.5 times better in the band but 1.3 times **worse** overall, because the soft tracks that dominate the overall number were deliberately de-weighted. What is left of the error is now mostly the $`y`$ term (55 µm of $`y`$ against 36 µm of $`x`$ at the SciFi plane), and it is **more coherent** along the track than before: 0.49 and 0.52 against 0.36 and 0.40, where 0.125 would mean independent steps. Smaller per-step errors that point the same way. The same shift shows at the other step lengths, where the $`y`$ part is 60 and 62 µm against $`x`$ parts of 52 and 37 µm.

This is also why a one-sided decomposition is no longer enough. The $`x`$-slope-only model accounted for 73 % of the pooled-loss network's endpoint error but only 38 % of the reweighted one's, because the reweighted one's remaining error is in the other plane.

![Endpoint error against momentum, per component, one panel per network. Each hexagon is coloured by how many of the 1,452 test tracks fall in it, on a logarithmic scale. The x component is the bending plane. Endpoint errors, against RK6 at the first SciFi plane.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/Block_F_reweighted_loss/F3_Analysis/figures_writeup/error_vs_p_x.png)

![The same, for the y component. Above about 5 GeV the y error is now the larger of the two, which is the shift the reweighting produced.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/Block_F_reweighted_loss/F3_Analysis/figures_writeup/error_vs_p_y.png)

## 4.5 Two results that belong here in one line each

**Around 5 GeV**, where this project's supervisors asked for the expected deviation, the $`N = 64`$, $`q = 2`$ network at 4 to 6 GeV (286 test tracks, endpoint) has a median $`|\Delta x|`$ of 93 µm, a 68 % half-width of 146 µm and an RMS of 618 µm; the RMS is six times the median because 17 of the 286 tracks land beyond 1 mm, and those start at the edge of the acceptance. The honest single number for the core is the **68 % half-width**. The first child page has the full table and the tail analysis.

**Against the Geant4-true SciFi state**, rather than against RK6, the networks and the RK6 reference are **indistinguishable**: their median deviations from the true state agree to within 3.5 % in every band from 2 to 25 GeV and to within 8.4 % at 25 to 200 GeV, while the network-against-RK6 error is 11 to 25 times smaller than either. The whole gap to the truth is the material the particle crosses, which no field-only method reproduces and which the loss never asked for. The third child page has the table and the caveats.

---

The three child pages below carry the rest: the full per-component and per-momentum tables and the behaviour around 5 GeV; the reproduced analysis set, with the single-step error, the growth along the crossing, the over-training checks, the forward cost and the case study; and the comparison against the Geant4-true state.

# 5. Conclusion and next steps

## 5.1 What the reweighting did

Measuring every residual as the displacement it would cause at the SciFi plane, as a fraction of the track's own bend, and weighting up the 10 to 50 GeV band, does what it was designed to do, and the improvement is larger than the run-to-run scatter.

- The endpoint radial error of the pre-registered network falls 39 %, from 145.7 to 88.8 µm, and in the 10 to 50 GeV band by a factor 4.4, from 108.1 to 24.8 µm.
- The per-step $`x`$ slope error falls by a factor 1.8 overall and 4.1 in the band, clearing the pre-registered threshold by a factor 2.2.
- A systematic under-bend of −54 µm in the band is removed, leaving −6 µm.
- The loss stays label-free. Both weight factors are functions of the input state and the field map, so nothing about the construction's central claim is given up to get this.

## 5.2 What it did not do

- **Below 5 GeV the error is worse**, by a factor 1.5 at 2 to 5 GeV and more below that, and the 95th percentile is 1.6 times worse. This is the accepted trade, not a surprise, but it is a real cost and it matters if soft tracks turn out to be the priority.
- **The error moved from $`x`$ into $`y`$.** The $`y`$ endpoint error did not improve at all, and the per-step $`y`$ slope error is slightly worse overall. What now limits the endpoint is the $`y`$ plane, which the weighting treats symmetrically with $`x`$ but which the network parameterises separately.
- **The remaining error is more coherent**, 0.49 and 0.52 against 0.36 and 0.40 for independent-step values of 0.125. Smaller per-step errors that all point the same way accumulate as efficiently as larger random ones would, which is why the endpoint error did not fall as far as the per-step error did.
- **The heavy tail is untouched.** Seven per cent of test tracks end more than 1 mm away, the worst 5 % carry 46 % of the total error, and those tracks are soft and near the edge of the acceptance. No momentum window addresses that.
- **Shorter steps did not help.** The shortest-step network has the smallest per-step error and the largest in-band endpoint error of the three.

## 5.3 Next steps, in the order they are worth doing

1. **Move the window down.** The supervisors' region of interest is around 5 GeV, and the loss window is 10 to 50 GeV, which is why the core around 5 GeV is about seven times wider than at 10 to 20 GeV. The window parameters are two constants in the loss module and a retrain is about 22 hours for the cheapest network. A follow-up study is built and gated end to end and is waiting only on the choice of window; its own pre-flight over six candidate windows recommends **3 to 8 GeV**, which puts 61.0 % of the loss (measured with the most extreme 1 % of states trimmed) inside that band against 48.7 % under the current window, and ranks states by their true endpoint cost at +0.94 inside it against +0.76 (`preflight_windows.csv`). Expect the trade to run the other way: better around 5 GeV, worse in 10 to 50 GeV.
2. **Run the ablations if the mechanism is disputed.** `no_lever`, `no_track` and `no_window` each switch one factor off by replacing the quantity that varies with its own average, so the size of the loss is unchanged and only its distribution moves. They were deliberately held back, because they are the diagnosis for an ambiguous result and this result is not ambiguous. They become worth the farm time if someone asks which of the three factors did the work, or if the window move in step 1 behaves unexpectedly. The trainer takes them already, so they cost farm time and no new code.
3. **Supervise the step Jacobian.** The residual is now weighted by what an error costs, and the error that remains is coherent along the track: the steps are making the same mistake repeatedly rather than independent mistakes. That is a property of the map the network has learned, not of any single residual, and it is the next lever, as the pre-registration anticipated for the null outcome.
4. **Treat the heavy tail separately.** The tracks beyond 1 mm start far from the beam line: at 4 to 6 GeV their median starting $`|x|`$ is 189 mm against 114 mm for the rest of the band. A weight on the starting $`|x|`$ is a different lever from a momentum window and does not interact with it.
5. **Settle the reference.** Everything here is measured against a field-only RK6 track. That reference differs from the Geant4 truth by 1.7 mm in the median, part of it a charge-antisymmetric systematic rather than scattering. It does not change any number on this page, all of which are against RK6, but it sets the ceiling the whole programme is working towards. That question has its own to-do and its own write-up ([link to be filled by George]).

# Provenance

- Repository `github.com/GeorgeWilliam1999/LHCb-Extrapolation`. **The working tree is uncommitted**: local HEAD `1040ee9`, GitHub `main` at `a062bd8`, and the whole folder `single_network_chain_discrete_approach/` is untracked. Every figure on this page and its children is served from `main`, so **the images render only after George commits and pushes**. No commit hash is claimed for this work.
- Folder `single_network_chain_discrete_approach/Block_F_reweighted_loss/`. It imports the network, the metrics, the tracks and the trainer from `../Block_E_single_network_chain/` and the numerical machinery from `../_shared/`, both of which are read-only to it.
- **Data**: `Block_E_single_network_chain/E0_Track_dataset/results/tracks.npz`, 14,482 crossings split by particle into 11,567 train / 1,463 validation / 1,452 test. Field map v8r1, magnet up. Reference: `_shared/reference.py`, sixth-order Runge–Kutta at 0.1 mm steps.
- **The loss and its gates**: `F0_Weighting/weighted_loss.py`; `F0_Weighting/check_weights.py` → `results/check_weights.json` (gates 1 to 4, `all_pass: true`) and `results/preflight_shares.csv`; `F0_Weighting/field_correlations.py` → `results/field_correlations.json`.
- **Training**: `F1_Training/train_weighted.py`, with `results/train_weighted.diff` against the trainer it was derived from; `F1_Training/make_jobs.py`, `condor/resubmit.py`, `condor/keeper.sh`; run folders `F1_Training/results/full/N064_q02`, `N128_q08`, `N256_q16` (`record.json`, `rounds.csv`, `history.csv`, `chain_states.npz`, `network.pt`, `scale.json`). Farm clusters **5809660** (all three runs), **5823724** and **5833416** (the extension of N = 256, q = 16; two jobs on the same run from restart 1,198, 302 of 1,807 history rows and 13 of 74 round rows duplicated). Restarts and rounds: 1,000 / 40, 1,000 / 40, 1,500 / 60, at 25 restarts a round.
- **Comparison and stopping**: `F2_Analysis/compare_to_blockE.py` → `results/headline.csv`; `F2_Analysis/components_and_momentum.py` → `results/components.csv`, `results/momentum_bands.csv`; `F2_Analysis/error_tables_by_component.py` → `results/error_by_component_5-30GeV.csv`, `figures/error_by_component_5-30GeV.png`; `F2_Analysis/errors_near_5gev.py` → `results/errors_near_5gev.csv`, `figures/signed_errors_4-6GeV.png`; `F2_Analysis/anatomy_xy.py` → `results/anatomy_xy_*.json`; `F2_Analysis/against_true_state.py` → `results/against_true_state.csv`, `figures/against_true_state.png`. All produced on 2026-09-22 between 16:27 and 16:41 on the finished weights.
- **The reproduced analysis set**: `F3_Analysis/run_e3_for_block_f.py`, which imports each analysis script from `../Block_E_single_network_chain/E3_Analysis/` and runs it on these runs, with `F3_Analysis/grid_free_panels.py` for the four grid-bound ones; `F3_Analysis/mirror_mini_paper.py`. Outputs `F3_Analysis/results/*.csv`, `*.json` and `F3_Analysis/figures/*.png`, produced 2026-09-22 between 16:40 and 16:43; `results/run_log.json` records which scripts ran.
- **Convergence and the tail, rerun for this page**: `Block_G_low_momentum_window/G2_Analysis/convergence.py --runs ../../Block_F_reweighted_loss/F1_Training/results/full --out ../../Block_F_reweighted_loss/F4_Writeup` → `F4_Writeup/results/convergence_check.csv` and `F4_Writeup/figures/convergence_check.png`; the same folder's `errors_near_5gev.py` with the same arguments → `F4_Writeup/results/errors_near_5gev.csv` and `F4_Writeup/results/tail_near_5gev.csv`. Both 2026-09-22.
- **Numbers computed for this page**: `F4_Writeup/numbers.py` → `F4_Writeup/results/writeup_numbers.json` (the growth along the crossing, the chain-over-single-step ratios, the wall clock per restart, the run constants, the worked example of section 3.2, the momentum content of the splits and the starting-x profile). **Figures redrawn for this page**: `F4_Writeup/make_figures.py` → `F3_Analysis/figures_writeup/`, which wraps matplotlib's title setters so that the redrawn figures carry no folder label, and otherwise runs the same drawing code on the same runs.
- **The window recommendation of section 5.3** is from `Block_G_low_momentum_window/G0_Weighting/results/preflight_windows.csv`.
- **Trust: Provisional.** Provenance located and checked; only George sets Verified.
