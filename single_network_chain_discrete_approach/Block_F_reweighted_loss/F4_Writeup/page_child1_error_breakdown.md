<callout icon="📐">
	Child page of the reweighted-loss write-up. Every number here is an **endpoint** error: the network applied $`N`$ times from the track's real state on the last Upstream Tracker plane, its state on the first SciFi plane compared with the RK6 track carried to the same plane, on the 1,452 test tracks unless another count is given. Slope errors are in units of $`10^{-3}`$ (slope); the result files label that column "mrad", which is the same number to within 7 % at the edge of the acceptance. **The figures render only after George pushes.**
</callout>
<table_of_contents/>
# 1. Per component, all test tracks

Median magnitude with the signed median beside it, overall and in the 10 to 50 GeV band the loss weights up. The counterpart row in each block is **the same network under the pooled loss**, trained to its own plateau by the same rule. From `components.csv`.

## N = 64, q = 2, dz = 80.9 mm

<table fit-page-width="true" header-row="true">
	<tr>
		<td>component</td>
		<td>reweighted: median magnitude</td>
		<td>reweighted: signed median</td>
		<td>reweighted: in band, magnitude</td>
		<td>reweighted: in band, signed</td>
		<td>pooled loss: median magnitude</td>
		<td>pooled loss: signed median</td>
		<td>pooled loss: in band, magnitude</td>
		<td>pooled loss: in band, signed</td>
	</tr>
	<tr>
		<td>x [µm]</td>
		<td>**37.5**</td>
		<td>−3.6</td>
		<td>**11.6**</td>
		<td>−6.3</td>
		<td>110.7</td>
		<td>−39.4</td>
		<td>86.9</td>
		<td>−54.4</td>
	</tr>
	<tr>
		<td>y [µm]</td>
		<td>55.2</td>
		<td>+2.9</td>
		<td>13.9</td>
		<td>+2.0</td>
		<td>52.3</td>
		<td>−15.5</td>
		<td>39.6</td>
		<td>−22.8</td>
	</tr>
	<tr>
		<td>tx</td>
		<td>0.0323</td>
		<td>−0.0108</td>
		<td>0.0106</td>
		<td>−0.0075</td>
		<td>0.0368</td>
		<td>−0.0162</td>
		<td>0.0342</td>
		<td>−0.0208</td>
	</tr>
	<tr>
		<td>ty</td>
		<td>0.0389</td>
		<td>+0.0079</td>
		<td>0.0126</td>
		<td>+0.0043</td>
		<td>0.0214</td>
		<td>−0.0030</td>
		<td>0.0155</td>
		<td>−0.0074</td>
	</tr>
</table>

The in-band columns are the sharpest statement in this write-up. In 10 to 50 GeV the reweighted network's typical miss is 11.6 µm in $`x`$ and 13.9 µm in $`y`$, against 86.9 and 39.6 µm; and its signed medians are −6.3 and +2.0 µm against −54.4 and −22.8, so the systematic under-bend is gone as well as most of the width.

## N = 128, q = 8, dz = 40.5 mm

<table fit-page-width="true" header-row="true">
	<tr>
		<td>component</td>
		<td>reweighted: median magnitude</td>
		<td>reweighted: signed median</td>
		<td>reweighted: in band, magnitude</td>
		<td>pooled loss: median magnitude</td>
		<td>pooled loss: signed median</td>
		<td>pooled loss: in band, magnitude</td>
	</tr>
	<tr>
		<td>x [µm]</td>
		<td>51.3</td>
		<td>−0.8</td>
		<td>13.7</td>
		<td>81.7</td>
		<td>+8.2</td>
		<td>53.9</td>
	</tr>
	<tr>
		<td>y [µm]</td>
		<td>61.2</td>
		<td>−2.3</td>
		<td>16.9</td>
		<td>59.2</td>
		<td>+6.2</td>
		<td>34.7</td>
	</tr>
	<tr>
		<td>tx</td>
		<td>0.0382</td>
		<td>+0.0048</td>
		<td>0.0117</td>
		<td>0.0268</td>
		<td>−0.0019</td>
		<td>0.0175</td>
	</tr>
	<tr>
		<td>ty</td>
		<td>0.0350</td>
		<td>+0.0009</td>
		<td>0.0105</td>
		<td>0.0209</td>
		<td>+0.0003</td>
		<td>0.0125</td>
	</tr>
</table>

## N = 256, q = 16, dz = 20.2 mm

<table fit-page-width="true" header-row="true">
	<tr>
		<td>component</td>
		<td>reweighted: median magnitude</td>
		<td>reweighted: signed median</td>
		<td>reweighted: in band, magnitude</td>
		<td>pooled loss: median magnitude</td>
		<td>pooled loss: signed median</td>
		<td>pooled loss: in band, magnitude</td>
	</tr>
	<tr>
		<td>x [µm]</td>
		<td>37.6</td>
		<td>−1.4</td>
		<td>12.3</td>
		<td>103.4</td>
		<td>+31.5</td>
		<td>68.9</td>
	</tr>
	<tr>
		<td>y [µm]</td>
		<td>61.3</td>
		<td>−4.8</td>
		<td>27.0</td>
		<td>69.7</td>
		<td>+20.7</td>
		<td>40.4</td>
	</tr>
	<tr>
		<td>tx</td>
		<td>0.0349</td>
		<td>+0.0042</td>
		<td>0.0087</td>
		<td>0.0365</td>
		<td>+0.0116</td>
		<td>0.0213</td>
	</tr>
	<tr>
		<td>ty</td>
		<td>0.0326</td>
		<td>−0.0052</td>
		<td>0.0119</td>
		<td>0.0259</td>
		<td>+0.0064</td>
		<td>0.0134</td>
	</tr>
</table>

Across all three, the same shape: $`x`$ improves by a factor 1.6 to 3.0, $`y`$ does not improve, and the sign offsets shrink to a few micrometres. The $`y`$ error is the larger of the two in every reweighted network and the smaller of the two in every pooled-loss one.

# 2. Per component in 5 to 30 GeV, the band the supervisors asked to see

On the 862 test tracks with 5 GeV ≤ p \< 30 GeV, the three networks on their own, median magnitude with the 95th percentile in brackets. From `error_by_component_5-30GeV.csv`.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>component</td>
		<td>N = 64, q = 2, dz = 81 mm</td>
		<td>N = 128, q = 8, dz = 40 mm</td>
		<td>N = 256, q = 16, dz = 20 mm</td>
	</tr>
	<tr>
		<td>x [µm]</td>
		<td>20.1 (219)</td>
		<td>31.1 (266)</td>
		<td>**19.0** (209)</td>
	</tr>
	<tr>
		<td>y [µm]</td>
		<td>**31.5** (264)</td>
		<td>33.3 (407)</td>
		<td>40.9 (249)</td>
	</tr>
	<tr>
		<td>tx</td>
		<td>0.019 (0.178)</td>
		<td>0.022 (0.165)</td>
		<td>**0.014** (0.170)</td>
	</tr>
	<tr>
		<td>ty</td>
		<td>0.023 (0.204)</td>
		<td>**0.020** (0.188)</td>
		<td>0.020 (0.178)</td>
	</tr>
</table>

The signed medians are all small against the magnitudes; the largest is $`x`$ at −4.8 µm for N = 64, q = 2. In this band $`y`$ is about 1.5 times $`x`$, and $`t_y`$ is larger than $`t_x`$ for two of the three networks. What remains of the error is in the non-bending plane.

![Endpoint error per state component on the 862 test tracks with 5 GeV ≤ p \< 30 GeV, one bar group per network, named by step count, stage count and step length. Medians with the 95th percentile marked. Endpoint errors after the full chain, against RK6 at the first SciFi plane.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/Block_F_reweighted_loss/F2_Analysis/figures/error_by_component_5-30GeV.png)

# 3. Per momentum band

Radial endpoint median with the 95th percentile in brackets, for the three networks, from `momentum_bands.csv`. The 1 to 2 GeV row holds five test tracks and is an indication only.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>band</td>
		<td>test tracks</td>
		<td>N = 64, q = 2, dz = 81 mm [µm]</td>
		<td>N = 128, q = 8, dz = 40 mm [µm]</td>
		<td>N = 256, q = 16, dz = 20 mm [µm]</td>
	</tr>
	<tr>
		<td>1–2 GeV</td>
		<td>5</td>
		<td>2,510 (6,693)</td>
		<td>2,466 (3,248)</td>
		<td>1,638 (3,622)</td>
	</tr>
	<tr>
		<td>2–5 GeV</td>
		<td>500</td>
		<td>370 (2,349)</td>
		<td>444 (3,110)</td>
		<td>357 (2,287)</td>
	</tr>
	<tr>
		<td>5–10 GeV</td>
		<td>430</td>
		<td>84 (606)</td>
		<td>110 (810)</td>
		<td>89 (455)</td>
	</tr>
	<tr>
		<td>10–20 GeV</td>
		<td>317</td>
		<td>**29** (176)</td>
		<td>35 (218)</td>
		<td>44 (176)</td>
	</tr>
	<tr>
		<td>20–50 GeV</td>
		<td>170</td>
		<td>**16** (64)</td>
		<td>21 (67)</td>
		<td>20 (76)</td>
	</tr>
	<tr>
		<td>50–100 GeV</td>
		<td>24</td>
		<td>20 (75)</td>
		<td>31 (66)</td>
		<td>47 (90)</td>
	</tr>
	<tr>
		<td>100–200 GeV</td>
		<td>6</td>
		<td>51 (318)</td>
		<td>56 (77)</td>
		<td>55 (111)</td>
	</tr>
</table>

The three curves have the same shape: a steep fall to a minimum in 20 to 50 GeV and a slow rise above it, where the window's roll-off takes weight away again and the tracks are few. The shortest-step network is best below 5 GeV and worst in 10 to 20 GeV, which is the opposite of what the per-step error alone would predict.

# 4. Around 5 GeV, signed

This section answers a direct question from the project's supervisors: for tracks of around 5 GeV, what deviation should be expected, and does it have an offset? Each row is the endpoint deviation of one component in one momentum band, on the test tracks. From `errors_near_5gev.csv`.

Four widths are given because they disagree, and the disagreement is itself the result. **Median magnitude** and the **68 % half-width** (half the 16th-to-84th-percentile range) describe the core. **RMS** and the standard deviation are taken about zero and about the mean respectively, and both are dragged upward by a few outliers. The **signed median** says whether there is an offset.

## N = 64, q = 2, dz = 80.9 mm, across the bands

<table fit-page-width="true" header-row="true">
	<tr>
		<td>band</td>
		<td>n</td>
		<td>component</td>
		<td>median magnitude</td>
		<td>68 % half-width</td>
		<td>RMS</td>
		<td>signed median</td>
		<td>p95 of the magnitude</td>
	</tr>
	<tr>
		<td>3–5 GeV</td>
		<td>341</td>
		<td>x [µm]</td>
		<td>138.4</td>
		<td>212.7</td>
		<td>764.9</td>
		<td>+2.5</td>
		<td>1,253</td>
	</tr>
	<tr>
		<td>3–5 GeV</td>
		<td>341</td>
		<td>y [µm]</td>
		<td>207.2</td>
		<td>337.2</td>
		<td>850.2</td>
		<td>+38.1</td>
		<td>1,258</td>
	</tr>
	<tr>
		<td>**4–6 GeV**</td>
		<td>286</td>
		<td>**x [µm]**</td>
		<td>**92.8**</td>
		<td>**146.2**</td>
		<td>**618.1**</td>
		<td>**−26.5**</td>
		<td>**920**</td>
	</tr>
	<tr>
		<td>**4–6 GeV**</td>
		<td>286</td>
		<td>**y [µm]**</td>
		<td>**130.0**</td>
		<td>**223.3**</td>
		<td>**425.4**</td>
		<td>**+28.1**</td>
		<td>**655**</td>
	</tr>
	<tr>
		<td>4–6 GeV</td>
		<td>286</td>
		<td>tx</td>
		<td>0.0776</td>
		<td>0.1027</td>
		<td>0.2837</td>
		<td>−0.0536</td>
		<td>0.520</td>
	</tr>
	<tr>
		<td>4–6 GeV</td>
		<td>286</td>
		<td>ty</td>
		<td>0.0955</td>
		<td>0.1611</td>
		<td>0.3286</td>
		<td>+0.0349</td>
		<td>0.516</td>
	</tr>
	<tr>
		<td>5–7 GeV</td>
		<td>233</td>
		<td>x [µm]</td>
		<td>57.4</td>
		<td>91.4</td>
		<td>519.9</td>
		<td>−15.2</td>
		<td>530</td>
	</tr>
	<tr>
		<td>5–7 GeV</td>
		<td>233</td>
		<td>y [µm]</td>
		<td>88.0</td>
		<td>147.5</td>
		<td>617.2</td>
		<td>+8.0</td>
		<td>543</td>
	</tr>
	<tr>
		<td>10–20 GeV</td>
		<td>317</td>
		<td>x [µm]</td>
		<td>12.5</td>
		<td>19.1</td>
		<td>94.4</td>
		<td>−5.0</td>
		<td>87</td>
	</tr>
	<tr>
		<td>10–20 GeV</td>
		<td>317</td>
		<td>y [µm]</td>
		<td>19.4</td>
		<td>34.2</td>
		<td>119.1</td>
		<td>+3.8</td>
		<td>119</td>
	</tr>
	<tr>
		<td>10–20 GeV</td>
		<td>317</td>
		<td>tx</td>
		<td>0.0133</td>
		<td>0.0153</td>
		<td>0.0590</td>
		<td>−0.0084</td>
		<td>0.067</td>
	</tr>
	<tr>
		<td>10–20 GeV</td>
		<td>317</td>
		<td>ty</td>
		<td>0.0161</td>
		<td>0.0217</td>
		<td>0.0682</td>
		<td>+0.0074</td>
		<td>0.084</td>
	</tr>
</table>

Two readings.

1. **There is no offset.** At 4 to 6 GeV the signed medians are −26.5 µm in $`x`$ and +28.1 µm in $`y`$, small against medians of 93 and 130 µm. The typical track is missed in either direction, not bent systematically one way. The width is the whole story.
1. **The core at 5 GeV is about seven times wider than in the band.** The 68 % half-width in $`x`$ is 146 µm at 4 to 6 GeV against 19.1 µm at 10 to 20 GeV, a factor 7.6. That is the loss window at work, and it is the reason the follow-up study exists.

## The same band for the other two networks

At 4 to 6 GeV, 286 test tracks, the three networks are within 10 to 20 % of each other. Median magnitude / 68 % half-width / RMS / signed median.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>network</td>
		<td>x [µm]</td>
		<td>y [µm]</td>
		<td>tx</td>
		<td>ty</td>
	</tr>
	<tr>
		<td>N = 64, q = 2, dz = 81 mm</td>
		<td>92.8 / 146.2 / 618 / −26.5</td>
		<td>130.0 / 223.3 / 425 / +28.1</td>
		<td>0.078 / 0.103 / 0.284 / −0.054</td>
		<td>0.095 / 0.161 / 0.329 / +0.035</td>
	</tr>
	<tr>
		<td>N = 128, q = 8, dz = 40 mm</td>
		<td>87.1 / 146.6 / 550 / −13.3</td>
		<td>133.3 / 237.9 / 620 / −25.1</td>
		<td>0.084 / 0.135 / 0.307 / +0.002</td>
		<td>0.087 / 0.135 / 0.413 / −0.015</td>
	</tr>
	<tr>
		<td>N = 256, q = 16, dz = 20 mm</td>
		<td>**80.7** / **128.6** / 476 / −25.4</td>
		<td>**100.0** / **171.0** / 428 / −10.2</td>
		<td>0.085 / 0.156 / 0.301 / +0.020</td>
		<td>**0.067** / **0.113** / 0.257 / −0.013</td>
	</tr>
</table>

![Signed endpoint deviation per component at 4 to 6 GeV, one histogram per network, on the 286 test tracks in that band, clipped at the 98th percentile for display. The legends give the signed median, the RMS and the 68 % half-width. Endpoint errors after the full chain from the last Upstream Tracker plane, against RK6.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/Block_F_reweighted_loss/F2_Analysis/figures/signed_errors_4-6GeV.png)

## Why the RMS and the median disagree sixfold

A handful of tracks in the band land more than a millimetre from the reference and carry most of the RMS. Removing them is not a correction and the numbers with them in are the honest ones; the point of the table below is to say **what** the RMS is made of, and to record a baseline for the follow-up study, whose pre-registration expects the tail fraction not to move when the loss window moves. From `tail_near_5gev.csv`, generated for this page; "the tail" is a radial endpoint error beyond 1 mm, and $`|x_0|`$ is the track's distance from the beam line on the last Upstream Tracker plane.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>network, 4–6 GeV (286 tracks)</td>
		<td>tracks in the tail</td>
		<td>median |x0|, tail</td>
		<td>median |x0|, the rest</td>
		<td>RMS in x, all</td>
		<td>RMS in x, tail removed</td>
		<td>RMS in y, all</td>
		<td>RMS in y, tail removed</td>
	</tr>
	<tr>
		<td>N = 64, q = 2, dz = 81 mm</td>
		<td>17 (5.9 %)</td>
		<td>189 mm</td>
		<td>114 mm</td>
		<td>618 µm</td>
		<td>**184 µm**</td>
		<td>425 µm</td>
		<td>238 µm</td>
	</tr>
	<tr>
		<td>N = 128, q = 8, dz = 40 mm</td>
		<td>20 (7.0 %)</td>
		<td>201 mm</td>
		<td>114 mm</td>
		<td>550 µm</td>
		<td>192 µm</td>
		<td>620 µm</td>
		<td>267 µm</td>
	</tr>
	<tr>
		<td>N = 256, q = 16, dz = 20 mm</td>
		<td>16 (5.6 %)</td>
		<td>250 mm</td>
		<td>114 mm</td>
		<td>476 µm</td>
		<td>163 µm</td>
		<td>428 µm</td>
		<td>201 µm</td>
	</tr>
</table>

The tail tracks start at the edge of the acceptance: their median $`|x_0|`$ is 189 to 250 mm against 114 mm for the rest of the band. The tail grows towards lower momentum (12.9 % of the 3 to 5 GeV band for N = 64, q = 2, against 3.5 % of 5 to 8 GeV), which is the same population on both counts: soft and far from the beam line. A momentum window does not address it; a weight on the starting $`|x|`$ would be the separate lever.

**The honest single number** for the deviation of a 5 GeV track is therefore the 68 % half-width, 146 µm in $`x`$ and 223 µm in $`y`$ for N = 64, q = 2. The median is 93 and 130 µm; the RMS is 618 and 425 µm with the tail and 184 and 238 µm without it; and 21 to 24 % of the band is beyond 200 µm in $`x`$.

# Sources

`F2_Analysis/results/components.csv`, `error_by_component_5-30GeV.csv`, `momentum_bands.csv`, `errors_near_5gev.csv`; `F2_Analysis/figures/error_by_component_5-30GeV.png`, `signed_errors_4-6GeV.png`; `F4_Writeup/results/tail_near_5gev.csv` (from `Block_G_low_momentum_window/G2_Analysis/errors_near_5gev.py` run on these runs). Full provenance is on the parent page.
