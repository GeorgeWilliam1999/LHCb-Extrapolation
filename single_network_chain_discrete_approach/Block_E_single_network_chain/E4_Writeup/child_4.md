<callout icon="🗓️">
	**Two dates, and it matters which is which.** *As stopped on 18 September 2026* means the checkpoints where training was halted by hand; every figure on these pages and almost every table comes from those networks, regenerated from the kept snapshots by `E4_Writeup/make_figures.py`, whose reproduction check reports zero mismatches over 192 compared cells against the published tables. *After the extension, 22 September 2026* means the same sixteen networks trained on under the plateau rule, scored into `headline.csv`. Those are the numbers to quote for what the construction achieves.
</callout>
# 1. The sixteen endpoints
Median radial endpoint error at $`z_1`$ in µm, over the 1,452 test tracks. First, where training was stopped by hand:
<table fit-page-width="true" header-row="true" header-column="true">
<tr>
<td>steps, step length</td>
<td>q = 2</td>
<td>q = 4</td>
<td>q = 8</td>
<td>q = 16</td>
</tr>
<tr>
<td>2 steps, 2,588.9 mm</td>
<td>9,250</td>
<td>284</td>
<td>188</td>
<td>224</td>
</tr>
<tr>
<td>64 steps, 80.9 mm</td>
<td>166</td>
<td>172</td>
<td>204</td>
<td>194</td>
</tr>
<tr>
<td>128 steps, 40.5 mm</td>
<td>207</td>
<td>196</td>
<td>159</td>
<td>246</td>
</tr>
<tr>
<td>256 steps, 20.2 mm</td>
<td>185</td>
<td>196</td>
<td>194</td>
<td>137</td>
</tr>
</table>
And after the extension, which is what the construction actually reaches:
<table fit-page-width="true" header-row="true" header-column="true">
<tr>
<td>steps, step length</td>
<td>q = 2</td>
<td>q = 4</td>
<td>q = 8</td>
<td>q = 16</td>
</tr>
<tr>
<td>2 steps, 2,588.9 mm</td>
<td>9,250</td>
<td>285</td>
<td>184</td>
<td>203</td>
</tr>
<tr>
<td>64 steps, 80.9 mm</td>
<td>146</td>
<td>119</td>
<td>138</td>
<td>164</td>
</tr>
<tr>
<td>128 steps, 40.5 mm</td>
<td>142</td>
<td>154</td>
<td>122</td>
<td>143</td>
</tr>
<tr>
<td>256 steps, 20.2 mm</td>
<td>126</td>
<td>150</td>
<td>128</td>
<td>148</td>
</tr>
</table>
Three things to read from these.
- **At 64 steps and above the endpoint error is 119 to 164 µm and depends usefully on neither the step count nor the stage count.** The best cell, 119 µm, is 64 steps with 4 stages; the worst, 164 µm, is 64 steps with 16 stages. The ten-round validation spread of a single run is 9 to 21 per cent at these settings, so the ordering inside that band is not a measurement.
- **Two steps is a different regime.** At two stages the network is at 9,250 µm and the exact scheme at the same setting is at 9,271 µm: the network has reached the scheme's own limit, and the error is entirely discretisation. From 8 stages a 2,589 mm step reaches 184 to 203 µm, and there the error is the network's again.
- **The extension was worth 12 to 42 per cent** at 64 steps and above for eleven of the twelve runs there, and cost 8 per cent on one, 256 steps with 16 stages, which went 137 to 148 µm, inside its own round-to-round spread.
![Left: the median radial endpoint error at the first fibre-tracker plane for each of the sixteen networks, at the checkpoints of 18 September 2026, on the 1,452 test tracks. Right: the same against stage count, one line per step count, with the exact collocation scheme at the same settings dotted in the same colour.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/Block_E_single_network_chain/E3_Analysis/figures_writeup/error_qdz.png)
# 2. A single step is excellent
The same networks, applied **once** from a reference state on each start plane and compared with the reference one step later. Median radial single-step error in µm, at the checkpoints of 18 September 2026, over every (track, plane) pair of the test split:
<table fit-page-width="true" header-row="true" header-column="true">
<tr>
<td>steps, step length</td>
<td>q = 2</td>
<td>q = 4</td>
<td>q = 8</td>
<td>q = 16</td>
</tr>
<tr>
<td>2 steps, 2,588.9 mm</td>
<td>1,430.8</td>
<td>92.46</td>
<td>62.53</td>
<td>72.81</td>
</tr>
<tr>
<td>64 steps, 80.9 mm</td>
<td>0.415</td>
<td>0.557</td>
<td>0.648</td>
<td>0.810</td>
</tr>
<tr>
<td>128 steps, 40.5 mm</td>
<td>0.151</td>
<td>0.257</td>
<td>0.309</td>
<td>0.374</td>
</tr>
<tr>
<td>256 steps, 20.2 mm</td>
<td>0.091</td>
<td>0.085</td>
<td>0.128</td>
<td>0.094</td>
</tr>
</table>
At 64 steps and above a step is accurate to between **0.085 and 0.81 µm**, which is 0.44 to 2.1 per cent of the straight line's error over the same step.
**So the endpoint error is almost entirely accumulation.** Dividing the radial endpoint error by the radial single-step error of the same network gives 240 to 2,291 at 64 steps and above: at 128 steps with 8 stages the chain is 514 times its own single step, not 128 times. Errors that merely added would give the step count.
A single step is also four to five times worse below 5 GeV than above 10 GeV at every setting from 64 steps up, which is the momentum dependence of section 1 of the next results page in its simplest form.
![Left: the median radial error of one step, one line per step count. Middle: the endpoint error divided by the single-step error, which would equal the step count if the errors merely added. Right: where along the magnet the single-step error sits. All at the checkpoints of 18 September 2026 on the test tracks. The middle panel divides by the endpoint error as each run's record stores it, in the max metric, which runs 10 to 20 per cent below the radial ratios quoted in the text.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/Block_E_single_network_chain/E3_Analysis/figures_writeup/single_step.png)
# 3. The error grows faster than linearly along the crossing
Median radial error against the reference at the quarter point, the halfway point and the end, in µm, at the checkpoints of 18 September 2026:
<table fit-page-width="true" header-row="true">
<tr>
<td>network</td>
<td>a quarter of the way</td>
<td>half way</td>
<td>at the end</td>
<td>end over half</td>
</tr>
<tr>
<td>64 steps, 80.9 mm, 2 stages</td>
<td>18.8</td>
<td>57.2</td>
<td>166.3</td>
<td>2.91</td>
</tr>
<tr>
<td>128 steps, 40.5 mm, 8 stages</td>
<td>21.2</td>
<td>56.1</td>
<td>158.6</td>
<td>2.83</td>
</tr>
<tr>
<td>256 steps, 20.2 mm, 16 stages</td>
<td>15.0</td>
<td>46.6</td>
<td>136.8</td>
<td>2.94</td>
</tr>
</table>
If the steps were independent the error would grow as the square root of the distance, a factor 1.41 from half way to the end; if they were perfectly aligned it would grow linearly, a factor 2. It grows by a factor of about 2.9. The reason is on the next results page: a slope error made early is not a fixed displacement, it is a displacement that keeps accumulating over the distance that is left.
![The median radial error of the chain plane by plane along the crossing, one panel per step count, at the checkpoints of 18 September 2026, on the test tracks.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/Block_E_single_network_chain/E3_Analysis/figures_writeup/error_vs_z.png)
# 4. Against the exact scheme, the straight line and the material floor
<table fit-page-width="true" header-row="true">
<tr>
<td>what</td>
<td>median radial error at z1 \[µm\]</td>
<td>95th percentile \[µm\]</td>
</tr>
<tr>
<td>the straight line, the magnet switched off</td>
<td>450,542</td>
<td>1,252,354</td>
</tr>
<tr>
<td>the material floor: the real fibre-tracker state against the reference</td>
<td>1,896</td>
<td>13,432</td>
</tr>
<tr>
<td>the networks at 64 steps and above, after the extension</td>
<td>119 to 164</td>
<td>807 to 1,284</td>
</tr>
<tr>
<td>the exact collocation scheme at 64 steps and above</td>
<td>0.00018 to 0.51</td>
<td>not tabulated</td>
</tr>
</table>
**The scheme is not the limit at 64 steps and above.** Its own discretisation error there is under a micrometre, between 300 and 1,000 times below the networks: what is measured is the network and its optimiser, not the numerical method. At 2 steps with 2 stages the opposite holds, the scheme is the whole story, and the network sits within 0.3 per cent of it, 9,250 µm against 9,271 µm.
# 5. Against one network per step
At the eight settings the two studies share, with both re-scored radially on the same test tracks and both at their own stopping point, the single chained network is **11.5 to 20.3 times more accurate** than a chain of separately trained networks: 166 µm against 2,304 µm at 64 steps with 2 stages, and 159 µm against 3,212 µm at 128 steps with 8 stages. The comparison carries a caveat in both directions. The per-step chains were stopped early by the optimiser tolerances that the rescaling of protocol 5 was introduced to defeat, so they are not that design at its best; and they trained on 2,000 particles against 11,567 here, because a per-step network need only see its own plane.
