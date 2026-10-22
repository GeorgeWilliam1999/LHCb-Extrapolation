# Output form: the straight line plus a scaled correction

**Date:** 2026-10-22. **Source:** `src/rkpinn/networks/output_forms.py`, class `StraightLinePlusScaledCorrection`. **Gates:** `tests/test_networks.py`, `tests/test_training.py`.

## What it is

An output form turns the raw outputs of a network, numbers of order one from its last linear layer, into states: positions in millimetres and slopes. This form writes every output of a stage network as the straight-line extrapolation of the input state plus a correction, and the network's raw output is that correction divided by its expected size.

It is the form of every network of the second mini-paper (`Archive/single_network_chain_discrete_approach/Block_E_single_network_chain/E1_Network_grid/chain_network.py`, class `ChainNetwork`), chosen by George on 14 September 2026 and ported on 22 October 2026 with the arithmetic unchanged.

## The straight line

A step starts on the plane z_start with the input state S = (x, y, tx, ty, q/p), where tx = dx/dz and ty = dy/dz are the slopes. Output j of the network is the state on the plane z_j = z_start + c_j dz, where dz is the step length and c_j the node of the tableau (c_j = 1 for a predicted end state). With no field the track is a straight line, so

    straight_j(S) = ( x + tx c_j dz,  y + ty c_j dz,  tx,  ty ).

With the last layer of the network zero, the form returns this line. At 64 steps of 80.9 mm it is within about 0.1 mm of the truth for a 10 GeV track, where the states themselves span hundreds of millimetres.

## The correction and its scale

    output_j = straight_j(S) + scale (x) raw_j,

where raw_j is the network's raw output for output j, shape (4,), and scale is a vector of four numbers, one per component, multiplied element by element. The scale is one of three things, chosen by the setting `scale_of_outputs`.

**Per track** (`per_track`, the scale of the frozen code). The bending of a track over one step is set by the field integral along its straight line. With kappa = 1e-3 the units of the equation of motion, and sixteen midpoint samples s_m = (m + 1/2) dz / 16 along the step,

    I_B  = (1/16) SUM_m |B(x + tx s_m, y + ty s_m, z_start + s_m)| |dz|
    I_y  = (1/16) SUM_m sqrt(1 + tx^2 + ty^2) ( (1 + ty^2) |Bx| + |tx ty By| + |tx Bz| ) |dz|

    base_x = kappa |q/p| I_B
    base_y = max( kappa |q/p| I_y,  1e-3 base_x )

    scale = ( max(base_x |dz| / 2, 1e-9 mm),  max(base_y |dz| / 2, 1e-9 mm),
              max(base_x, 1e-12),             max(base_y, 1e-12) ).

base_x is the bend in the slope tx over the step for a field of strength |B|; |dz| / 2 turns a change of slope into a change of position over the step. I_y bounds the y-rate of the equation of motion, dty/dz = kappa (q/p) N ((1 + ty^2) Bx - tx ty By - tx Bz), term by term, so it does not vanish by cancellation where the terms do; the floor of 1e-3 base_x keeps the y scale from collapsing where Bx is small. The floors of 1e-9 mm and 1e-12 keep the scale positive for a track of very high momentum. The scale depends on the input and the field only; it carries no gradient.

**Global** (`median_of_per_track_scale_over_first_round_states`). The four numbers are the median, component by component, of the per-track scale over the 32,000 states of the first round. They are measured once, when the run is made, and recorded in `constants.json`; a resumed run reads them. At 64 steps of 80.9 mm they are about 0.095 mm in x, 0.0008 mm in y, 2.4e-3 in tx and 2e-5 in ty.

**Given** (four numbers in the configuration).

## Why it matters

The pooled loss divides every residual by the spread of the states, about 445 mm in x. A network that emits the states themselves has to reproduce x + tx dz through two tanh layers to two parts in a million before the bending is visible in the loss; the loss starts near 1 and L-BFGS in double precision stops near 3e-8. With the straight line a raw output of zero is already within 0.1 mm of the truth; the loss starts near 1e-4 and reaches 1e-9. Measured on the login node with the same trainer, loss, states, seed and body (correction plan of 2026-10-20, section 4): the straight line lowers the loss 190 to 390 times and the single-step error 12 to 15 times in the same number of restarts; the per-track scale and the global scale were within 20 per cent of each other.

## Gates

| Gate | Proves |
|---|---|
| `test_stage_network_with_the_straight_line_form_is_identical_to_the_frozen_chain_network` | with `per_track` and a predicted end state, outputs and gradient identical to the frozen `ChainNetwork` to the last bit, at 2/2, 64/2 and 256/16, two seeds; the per-track scale identical to the frozen `residual_scale` and within 1e-14 of the frozen numpy twin |
| `test_self_chain_with_the_straight_line_form_is_identical_to_the_frozen_carry` | the chain over 64 steps identical to the frozen `carry`, on every plane |
| `test_the_global_scale_is_the_median_of_the_per_track_scale` | the global scale is the median of the per-track scale; with fixed numbers the output is the straight line plus the numbers times the raw outputs; the straight line is the frozen `straight_numpy` |
| `test_zeroed_last_layer_gives_the_declared_answer` | with the last layer zero, the straight line |
| `test_the_straight_line_form_refuses_what_it_cannot_be_built_with` | a scale that is not `per_track` or four positive numbers is refused; a network with no field map is refused |
| `test_a_network_is_built_again_from_its_settings_and_weights` | a network with this form, either scale, is built again from its settings and its snapshot |
| `test_the_straight_line_form_trains_from_a_configuration` | a run trains with either scale, records the scale it resolved to, and starts from a loss at least 100 times below that of direct states on the same states |
