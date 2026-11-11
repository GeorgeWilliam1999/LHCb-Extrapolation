One network on its own, chosen on the **validation** split by its mean validation error over the last eight rounds, so that the test tracks played no part in picking it. It is the 64-step network with two Gauss-Legendre stages, step length 80.9 mm, 18,956 parameters, at its checkpoint of 18 September 2026. Its momentum dependence, its starting-position dependence, its components in the 10 to 20 GeV band and its anatomy are on the page **Results: momentum, starting position, the components, and what the endpoint error is made of**; what follows is the headline and the shape of its error distribution.
# 1. The headline, against every comparator
Radial error at the first fibre-tracker plane, 1,452 test tracks:
<table fit-page-width="true" header-row="true">
<tr>
<td>what</td>
<td>median \[µm\]</td>
<td>95th pct \[µm\]</td>
<td>99th pct \[µm\]</td>
<td>worst \[µm\]</td>
<td>fraction beyond 1 mm</td>
</tr>
<tr>
<td>this network</td>
<td>166.3</td>
<td>1,154</td>
<td>3,304</td>
<td>24,985</td>
<td>6.3 %</td>
</tr>
<tr>
<td>the exact scheme at the same step and stage count</td>
<td>0.147</td>
<td>1.17</td>
<td>18.1</td>
<td>170</td>
<td>0 %</td>
</tr>
<tr>
<td>the previous study: one network per step, same settings</td>
<td>2,304</td>
<td>10,201</td>
<td>17,244</td>
<td>64,638</td>
<td>78.7 %</td>
</tr>
<tr>
<td>the straight line</td>
<td>450,542</td>
<td>1,252,354</td>
<td>1,448,745</td>
<td>1,832,614</td>
<td>100 %</td>
</tr>
<tr>
<td>the material floor: the real fibre-tracker state against the reference</td>
<td>1,896</td>
<td>13,432</td>
<td>30,182</td>
<td>117,634</td>
<td>66.6 %</td>
</tr>
</table>
Three readings. The scheme is a thousand times more accurate than the network at these settings, so nothing here is the discretisation. The same construction with a separate network per step is fourteen times worse, with the caveat that it was stopped early. And the network sits an order of magnitude below the material floor, so against the particle's real state it is indistinguishable from the reference it was trained towards; that comparison belongs to the companion page on the reference and the simulated truth, not here.
# 2. The shape of the error
![Left: the median radial error of the chain plane by plane along the crossing, with the single-step error from the reference track beneath it. Middle: the training loss against restart in blue and the validation error after each round in red. Right: the distribution of the endpoint error, split by momentum. The 64-step, two-stage network at its 18 September 2026 checkpoint, test tracks.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/Block_E_single_network_chain/E3_Analysis/figures_writeup/case_study_overview.png)
The left panel is the whole story of the study in one picture: the red line, what one step gets wrong, is flat at about 0.4 µm along the entire crossing, while the blue line, the chain, climbs by nearly three orders of magnitude. The middle panel is the mistake of 18 September 2026: the loss in blue is visibly flat over the last few hundred restarts while the red validation error is still stepping down, and training was stopped there anyway. The right panel shows the distribution: 1 to 5 GeV has a median of 299 µm, 5 to 20 GeV 124 µm and 20 to 200 GeV 149 µm, the U-shape again, with a long right tail in every band.
The distributions are wide, not offset. In the 10 to 20 GeV band the median absolute error is 83 µm in $`x`$ and 56 µm in $`y`$ while the signed medians are $`+19`$ and $`-16`$ µm, so the typical track is missed in either direction. What sets the width is momentum and the track's starting position, in that order.
