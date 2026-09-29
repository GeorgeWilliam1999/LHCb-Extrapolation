# The Notion project page: master index and workflow guide

**Last updated:** 2026-09-28 · **State:** published on 2026-09-28 · **Page:** LHCb Extrapolator, `39f5d544-b9d9-8093-8d27-e70185e12909`, <https://app.notion.com/p/39f5d544b9d980938d27e70185e12909>

This is the content for the project page itself. It is not a write-up. It is the guide that is always open: how we work, what every component is, and which experiments exist.

**How it was published.** The sections below were inserted at the top of the existing page in one call (`update-page`, `insert_content`, position start), after a rehearsal on a scratch page showed that tables, toggles and equations arrive as real blocks. The page was fetched back and checked: no backslash-escaped table, image or equation, and the earlier content unchanged underneath. No child pages were needed.

**How it is placed on the page.** The sections below are at the top, in this order, under the heading "Master index and workflow guide". Everything that was already on the page stays, unchanged, underneath, under the heading "Earlier content of this page (unchanged)": the earlier status, the simulation guide, meetings, to-dos, literature and write-ups. Each entry of the master index is a toggle, closed by default, so the page reads as an index and opens into theory. The three targets of section 3.5 are three toggles on the page, one per row of the table here.

**What it covers.** Work from 28 September 2026 onward. Earlier write-ups are left as they are and are reached through the write-ups database.

---

## 1. Status

*Kept by the agent, dated, rewritten at the end of each session.*

- **Goal.** Train networks that carry a track across the LHCb magnet by applying one learned Runge–Kutta step to its own output, and score them in one standard way.
- **Where we are.** Phases 1 to 5 of the package `rkpinn` are built and gated; 446 gates pass. Phase 1: the equation of motion, the field map, the Gauss–Legendre tableau, the exact collocation scheme, the sixth-order reference and the registry, each identical to the last bit to the frozen code run on the same machine. Phase 2: the store at `/data/bfys/gscriven/rkpinn_store`; the track set `12a8d35c3165` (11,567 training, 1,463 validation and 1,452 test tracks on 257 planes), identical to the last bit to the frozen file; the exact scheme's states of the test split. Phase 3: the predicted track, the two networks, the four losses and four targets; the end state is summed from the stages or predicted, and the terms of a loss are the stages or the stages and the end state, both settings of each experiment. Phase 4: the configuration and its run key, the round trainer, the optimiser, the two protocols, the stopping rule, snapshots and the lock; the optimiser is identical to the frozen trainer's to the last bit, and a run stopped and resumed is identical to one not stopped. Phase 5: the standard evaluation and the report of a run; run on the stored states of the six old runs of the second mini-paper, it gives the paper's numbers. No network is trained yet beyond the small runs of the gates.
- **Waiting on George.** The push of the local commits, and the word to start phase 6, the pilot run: the first network trained by the package.
- **Ruled.** The questions and George's answers are in the to-do "Rule on the three assumptions and the precision risk before the networks and losses are built", Status = Done, <https://app.notion.com/p/3e95d544b9d98176a53cd9f467ada79b>. The study of the root finder is the to-do <https://app.notion.com/p/3e95d544b9d981528d6edcc6961c9b36>.
- **The build.** Its to-do is "Build the package that trains and scores the self-chained networks", <https://app.notion.com/p/3e95d544b9d98134a723ce392d68f374>.

---

## 2. How we work

*Condensed from `RK_Pinn_module/WORKFLOW.md`, which is the full rule book.*

| Rule | |
|---|---|
| The package holds what any experiment would use | an experiment holds its own configurations and analysis, outside the package |
| An experiment is a set of configuration files | a controlled comparison changes one block |
| Every run is keyed, snapshotted and traceable to a commit | snapshots are never overwritten |
| Gates run before training | a change is finished when every gate passes |
| Convergence is judged on the validation error | never on the loss |
| Every directory explains everything in it | enforced by a test |
| Theory is written once, in this index | a write-up links here and does not restate it |
| Findings are discussed before a write-up is made | a write-up is Provisional until George verifies it |
| Everything is named in plain English | networks are named by their steps, stages and step length |

**To add a component:** confirm it is wanted, write it, register it, gate it, describe it in its directory, write its entry below, run all gates.

**To run an experiment:** write the question, agree the design, add the row to the experiments table, write the configurations, run a pilot, submit, produce the standard report, discuss, write up.

---

## 3. Master index

One entry per component. Each holds what it is, its theory, its source and its gates.

| Collection | Entries |
|---|---|
| The problem | the crossing; the state; the equation of motion; the field map |
| Numerical methods | the Gauss–Legendre tableau; the exact collocation scheme; the sixth-order reference |
| Networks | the stage network; collecting and summing the stages; the self chain; the whole-crossing supervised network |
| Loss functions | the stage residual; unweighted; pooled; cost-weighted; supervised endpoint |
| Targets | the reference state; the exact stage states; the Geant4-true state |
| Data | the tracks on 257 planes |
| Evaluation | the conventions; the ten standard outputs; stage errors held and carried |

### 3.1 The problem

#### The crossing

I study the leg from the last plane of the Upstream Tracker, at $z_0 = 2648.2$ mm, to the first plane of the scintillating-fibre tracker, at $z_1 = 7826.0$ mm. Its length is $L = z_1 - z_0 = 5177.8$ mm. The dipole magnet lies between the two planes, so this is the leg on which a track bends most.

#### The state

A track on a plane of constant $z$ is described by five numbers,

$$S = \left(x,\ y,\ t_x,\ t_y,\ q/p\right),$$

where $x$ and $y$ are the position on the plane in millimetres, $t_x = dx/dz$ and $t_y = dy/dz$ are the slopes and have no unit, and $q/p$ is the charge divided by the momentum. With the field alone acting, $q/p$ does not change along the track. It is an input to every step and is never predicted.

#### The equation of motion

The Lorentz force, written with $z$ as the independent variable, gives

$$\frac{dx}{dz} = t_x, \qquad \frac{dy}{dz} = t_y,$$

$$\frac{dt_x}{dz} = \kappa\,\frac{q}{p}\,\sqrt{1 + t_x^2 + t_y^2}\;\Big(t_x t_y B_x - (1 + t_x^2) B_y + t_y B_z\Big),$$

$$\frac{dt_y}{dz} = \kappa\,\frac{q}{p}\,\sqrt{1 + t_x^2 + t_y^2}\;\Big((1 + t_y^2) B_x - t_x t_y B_y - t_x B_z\Big),$$

where $(B_x, B_y, B_z)$ is the magnetic field at the point $(x, y, z)$ and $\kappa$ is the constant that fixes the units. I write the four equations together as $dS/dz = f(S, z)$. The function $f$ depends on $z$ through the field, so the equation is not autonomous.

- **Source:** `equation_of_motion/lhcb.py`.
- **Gates:** identical rates to the frozen code on 500 real states; the torch form equals the numpy form.

#### The field map

The field is the v8r1 map of the experiment, stored on a grid of 100 mm and interpolated linearly in each direction. The interpolated field is continuous, but its derivative jumps at every cell face. This matters: a high-order method that steps across a jump behaves like a low-order one.

- **Source:** `equation_of_motion/field_map.py`.
- **Gates:** the hash of the map file; identical field values to the frozen loader.

### 3.2 Numerical methods

#### The Gauss–Legendre tableau

A Runge–Kutta method with $q$ stages advances a state by a step $\Delta z$ through

$$S_j = S_0 + \Delta z \sum_{k=1}^{q} A_{jk}\, f\!\left(S_k,\ z_0 + c_k\,\Delta z\right), \qquad j = 1, \dots, q,$$

$$S_{\mathrm{end}} = S_0 + \Delta z \sum_{k=1}^{q} b_k\, f\!\left(S_k,\ z_0 + c_k\,\Delta z\right),$$

where $S_0$ is the state at the start of the step, the $S_j$ are the **stage states**, the $c_k$ are the **nodes** that place the stage planes inside the step, $A$ is the **stage matrix** and the $b_k$ are the **weights**. The three together are the tableau.

In the Gauss–Legendre family the nodes are the roots of the Legendre polynomial of degree $q$, mapped onto the interval from 0 to 1. The matrix $A$ is full, so every stage depends on every other: the method is implicit. Its formal order is $2q$, the highest a method with $q$ stages can have.

The tableau is built, not looked up, and checked every time against five identities: each row of $A$ sums to its node; the weights integrate polynomials exactly to degree $2q - 1$; the collocation conditions hold; the symplecticity identity $b_j A_{jk} + b_k A_{kj} = b_j b_k$ holds; and for $q = 1, 2, 3$ the result equals the closed forms in the literature.

- **Source:** `integrators/gauss_legendre_tableau.py`.
- **Gates:** the five identities to machine precision; identical to the frozen tableau at 16 stages.

#### The exact collocation scheme

The stage equations can be solved directly with a root finder, with no network. The answer is what a network with a loss of exactly zero would produce. Its distance from the reference is therefore the **ceiling** for that number of steps and stages: no network of that geometry can do better.

- **Source:** `integrators/exact_collocation.py`.
- **Gates:** every solve converges to a residual of $10^{-9}$; the stored exact states are reproduced.

#### The sixth-order reference

Everything is scored against a numerical solution of the equation of motion. The reference is Butcher's explicit method with seven stages and order six, in double precision, at a fixed step of 0.1 mm. On the real field map, halving its step moves the endpoint by 28 picometres at the median.

The reference contains the field and nothing else: no scattering, no energy loss.

- **Source:** `integrators/runge_kutta_sixth_order.py`.
- **Gates:** the tableau identities; fitted order on a smooth field; a forward-then-back integration closes.

### 3.3 Networks

#### The stage network

The network takes a state and the $z$ of the plane the step starts on, and emits the $q$ stage states of one Gauss–Legendre step:

$$\mathcal{M}_\theta : \left(S_0,\ z_{\mathrm{start}}\right) \longmapsto \left(\hat S_1,\ \hat S_2,\ \dots,\ \hat S_q\right),$$

where $\theta$ are the weights of the network and $\hat S_k$ is its estimate of the state on the plane $z_{\mathrm{start}} + c_k\,\Delta z$. The network emits the states themselves. It has $4q$ outputs.

The end of the step is formed in one of two ways, set by the experiment. It is summed from the stages, and the network has $4q$ outputs. Or it is predicted, and the network has $4(q+1)$ outputs, the last being the end state, as in the paper.

Knowing $z_{\mathrm{start}}$ is what lets one network serve every step, because the field a step crosses depends on where along the magnet it starts.

- **Source:** `networks/stage_network.py`.
- **Ruled:** George, 28 September 2026. The network learns the stages, not a correction to a straight line.

#### Collecting and summing the stages

The end of the step is not predicted. It is formed from the stages:

$$\hat S_{\mathrm{end}} = S_0 + \Delta z \sum_{k=1}^{q} b_k\, f\!\left(\hat S_k,\ z_{\mathrm{start}} + c_k\,\Delta z\right).$$

The end state is then consistent with the stages by construction.

- **Source:** `networks/collect_and_sum.py`.
- **Gate:** with the exact stages as input, the sum equals the exact scheme's end state.
- **Set by the experiment:** the end state is summed, as here, or predicted by the network.

#### The self chain

The step length is $\Delta z = L/N$. The same network is applied $N$ times. Each end state, with $q/p$ carried unchanged, is the input of the next step, and $z_{\mathrm{start}}$ advances by $\Delta z$.

- **Source:** `networks/self_chain.py`.

#### The whole-crossing supervised network

A separate network, applied once. It takes the state at the last plane of the Upstream Tracker and emits the state at the first plane of the fibre tracker. It has no stages and is trained on the reference end state. It is the comparison that shows what labels buy.

- **Source:** `networks/whole_crossing_network.py`.

### 3.4 Loss functions

#### The stage residual

Each stage must reconstruct the input through the stage equations:

$$r_{n,j,d} = \hat S_{n,j,d} - \Delta z \sum_{k=1}^{q} A_{jk}\, f_d\!\left(\hat S_{n,k},\ z_n + c_k\,\Delta z\right) - S_{n,d},$$

where $n$ labels the training state, $j$ the stage, and $d$ the component, one of $x$, $y$, $t_x$, $t_y$. The residual is zero for every $n$, $j$ and $d$ exactly when the network's stages solve the scheme. No label appears in it.

A predicted end state has a residual of the same form, with the weights $b_k$ in place of the row $A_{jk}$. Which terms a loss sums over is set by the experiment: the $q$ stages, or the $q$ stages and the end state. When the end state is summed from the stages its residual is zero, because it is built by the formula the residual tests.

#### The three label-free losses

All three are one formula with a different divisor:

$$\mathcal{L}(\theta) = \operatorname*{mean}_{n,j,d}\left(\frac{r_{n,j,d}}{s_{n,j,d}}\right)^{2}.$$

| Loss | Divisor $s$ | What it asks for |
|---|---|---|
| Unweighted | 1 | equal accuracy in raw units |
| Pooled | $s_d$: the spread of component $d$ over the first round's states | equal absolute accuracy in every component, on every track, at every plane |
| Cost-weighted | $D_{\mathrm{ref}} / \left(a_n\, g_{n,j,d}\right)$ | accuracy where an error costs most at the end of the track |

In the cost-weighted loss, $g_{n,j,d} = \left(1,\ 1,\ \ell_{n,j},\ \ell_{n,j}\right)$ for $d = x, y, t_x, t_y$, where $\ell_{n,j}$ is the distance from the stage plane to the end of the track plus one step. A slope error is multiplied by the distance it still has to travel; a position error is not. The factor $a_n$ is

$$a_n = \operatorname{clamp}\!\left(\sqrt{W(p_n)}\ \frac{D_{\mathrm{ref}}}{D_n},\ \left[\tfrac{1}{5}m,\ 5m\right]\right),$$

where $D_n$ is the total bend of track $n$ across the crossing, $D_{\mathrm{ref}}$ is a fixed reference bend, $W$ is a smooth window in momentum, and $m$ is the median of the unclamped factor over the batch.

All three have the same minimiser. They differ in how the optimiser spends its effort on the way there.

- **Source:** `losses/`.
- **Gates:** the exact solution gives machine zero under every loss; with every weight off, the weighted loss equals the unweighted one to the last bit.

#### The supervised endpoint loss

The mean squared difference between the predicted end state and the reference end state, per component, divided by the pooled spread.

### 3.5 Targets

| Target | Contains | Role |
|---|---|---|
| The reference state | the field only | the score, and the label of the supervised network |
| The exact stage states | the scheme's own answer | the ceiling; and the comparison for stage errors |
| The Geant4-true state | field, scattering, energy loss, interactions | the floor no field-only method can beat |

### 3.6 Data

#### The tracks on 257 planes

14,482 forward particles of the official sample simulated with the magnet polarity up, with pseudorapidity between 2 and 5, momentum between 1 and 200 GeV, and electrons excluded. Each particle's true state at the last plane of the Upstream Tracker is carried to $z_0$ and then across the crossing with the reference integrator. The state is stored on the 257 planes $z_0 + k L/256$. The split is by particle: 11,567 for training, 1,463 for validation, 1,452 for testing.

- **Source:** `track_data/build_tracks.py`.
- **Gate:** the same particles, in the same order, with the same states as the frozen file.
- **In the store:** the track set with the key `12a8d35c3165`.

### 3.7 Evaluation

#### The conventions

Positions in micrometres. Slopes as dimensionless numbers. Momentum bands below 3, 3 to 8, 8 to 20, 20 to 50 and above 50 GeV, with the loss's own window as an extra row. An error is named as an **endpoint** error, after the whole chain, or a **single-step** error.

#### The ten standard outputs

| # | Output |
|---|---|
| 1 | endpoint error per component |
| 2 | endpoint error per component and momentum band |
| 3 | signed distributions per component |
| 4 | single-step error per component |
| 5 | error along the track |
| 6 | stage errors, held and carried |
| 7 | the three references |
| 8 | convergence |
| 9 | where the loss puts its weight |
| 10 | cost |

#### Stage errors, held and carried

At every step $m$ and stage $k$ I store the stage error against the exact scheme started from the same input, $\delta S^{(m)}_k = \hat S^{(m)}_k - S^{(m),\mathrm{exact}}_k$. To first order, the error this causes at the end of the step is

$$\delta S^{(m)}_{\mathrm{end}} = \Delta z \sum_{k=1}^{q} b_k\, J^{(m)}_k\, \delta S^{(m)}_k,$$

where $J^{(m)}_k$ is the derivative of $f$ with respect to the state, evaluated at the stage. That error is then carried to the end of the track with the reference integrator. The contributions of all steps are checked to sum to the measured endpoint error.

This is the working form of the robustness question and will be revisited.

---

## 4. Experiments

The table is defined in `experiments_table.md`. Each row points to the write-up of record.

On the page it is the inline database "Experiments", data source `bb30377e-dd8d-4460-9c7d-7546cc5f857c`, <https://app.notion.com/p/1db7011d78f04d4fb3179320d56365e4>. It has no rows yet.
