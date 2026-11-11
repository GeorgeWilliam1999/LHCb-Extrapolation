Where the error lives and what it is built from. Unless the text says otherwise, every number is the 64-step, two-stage network, step length 80.9 mm, chosen on the validation split, at its checkpoint of 18 September 2026, on the 1,452 test tracks.
# 1. Momentum
The endpoint error is **U-shaped in momentum**. Median $`|\Delta x|`$ at the endpoint by band:
<table fit-page-width="true" header-row="true">
<tr>
<td>momentum</td>
<td>1-2</td>
<td>2-3</td>
<td>3-5</td>
<td>5-7</td>
<td>7-10</td>
<td>10-15</td>
<td>15-20</td>
<td>20-30</td>
<td>30-50</td>
<td>50-100</td>
<td>100-200 GeV</td>
</tr>
<tr>
<td>test tracks</td>
<td>5</td>
<td>159</td>
<td>341</td>
<td>233</td>
<td>197</td>
<td>208</td>
<td>109</td>
<td>115</td>
<td>55</td>
<td>24</td>
<td>6</td>
</tr>
<tr>
<td>median \|Δx\| \[µm\]</td>
<td>1,467</td>
<td>334</td>
<td>173</td>
<td>115</td>
<td>88</td>
<td>72</td>
<td>96</td>
<td>124</td>
<td>113</td>
<td>202</td>
<td>827</td>
</tr>
<tr>
<td>median \|Δy\| \[µm\]</td>
<td>61</td>
<td>267</td>
<td>109</td>
<td>67</td>
<td>63</td>
<td>50</td>
<td>67</td>
<td>54</td>
<td>34</td>
<td>48</td>
<td>33</td>
</tr>
</table>
Soft tracks bend most and are hardest, which is expected. The rise above 20 GeV is not: those tracks barely bend, their residuals are the smallest, and a loss divided by the pooled spread weights them least. That is the same mechanism as the loss-share result on the next results page, seen from the other end.
# 2. Starting position matters about as much as momentum
The median radial endpoint error against the track's $`|x|`$ on the last Upstream Tracker plane, with the number of tracks in brackets:
<table fit-page-width="true" header-row="true">
<tr>
<td>\|x\| at z0</td>
<td>0-75 mm</td>
<td>75-150 mm</td>
<td>150-250 mm</td>
<td>250-400 mm</td>
<td>400-700 mm</td>
</tr>
<tr>
<td>all test tracks</td>
<td>127 µm (693)</td>
<td>165 µm (346)</td>
<td>237 µm (199)</td>
<td>328 µm (131)</td>
<td>651 µm (83)</td>
</tr>
<tr>
<td>10-20 GeV only</td>
<td>101 µm (202)</td>
<td>129 µm (80)</td>
<td>169 µm (22)</td>
<td>277 µm (5)</td>
<td>720 µm (8)</td>
</tr>
</table>
Starting position and momentum are correlated, Spearman $`-0.405`$ over the test split: tracks far off axis at $`z_0`$ tend to be soft. But the trend survives at fixed momentum, so it is a second variable in its own right. A track entering at the edge of the acceptance is missed about **five times worse** than one of the same momentum near the beam line. The median $`|x|`$ at $`z_0`$ is 81 mm and the 95th percentile 413 mm, so the worst bin is a real but small population.
![The endpoint error per component over momentum and the track's starting x: surfaces of the median absolute error, and the 10 to 20 GeV band as a signed distribution against starting x. The 64-step, two-stage network at its 18 September 2026 checkpoint, test tracks.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/Block_E_single_network_chain/E3_Analysis/figures_writeup/case_study_components_3d.png)
![The same numbers drawn flat, because a surface hides cells behind ridges. Cells with fewer than the minimum number of tracks are left blank.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/Block_E_single_network_chain/E3_Analysis/figures_writeup/case_study_components_x0_maps.png)
# 3. The tails
<table fit-page-width="true" header-row="true">
<tr>
<td>group</td>
<td>tracks</td>
<td>median radial error \[µm\]</td>
<td>median momentum \[GeV\]</td>
<td>fraction below 5 GeV</td>
</tr>
<tr>
<td>the worst 5 per cent</td>
<td>73</td>
<td>1,940</td>
<td>3.6</td>
<td>67 %</td>
</tr>
<tr>
<td>the rest</td>
<td>1,379</td>
<td>157</td>
<td>7.1</td>
<td>33 %</td>
</tr>
<tr>
<td>pseudorapidity 2.0 to 2.5</td>
<td>158</td>
<td>881</td>
<td>3.7</td>
<td>70 %</td>
</tr>
<tr>
<td>pseudorapidity 3.0 to 3.5</td>
<td>292</td>
<td>180</td>
<td>5.4</td>
<td>43 %</td>
</tr>
<tr>
<td>pseudorapidity 4.5 to 5.0</td>
<td>196</td>
<td>98</td>
<td>15.4</td>
<td>7 %</td>
</tr>
<tr>
<td>positive charge</td>
<td>739</td>
<td>184</td>
<td>6.7</td>
<td>36 %</td>
</tr>
<tr>
<td>negative charge</td>
<td>713</td>
<td>148</td>
<td>7.0</td>
<td>34 %</td>
</tr>
</table>
The tail is the soft, low-pseudorapidity population, which is also the population that starts furthest off axis. The difference between the charges is within the scatter of the two samples and is not read as an asymmetry.
# 4. Per component in the 10 to 20 GeV band
On the 317 test tracks with momentum between 10 and 20 GeV:
<table fit-page-width="true" header-row="true" header-column="true">
<tr>
<td>10-20 GeV, 317 tracks</td>
<td>x \[µm\]</td>
<td>y \[µm\]</td>
<td>tx \[×10⁻³ slope\]</td>
<td>ty \[×10⁻³ slope\]</td>
</tr>
<tr>
<td>median absolute error</td>
<td>83.0</td>
<td>55.7</td>
<td>0.0263</td>
<td>0.0179</td>
</tr>
<tr>
<td>signed median</td>
<td>+18.7</td>
<td>-15.6</td>
<td>-0.0168</td>
<td>-0.0035</td>
</tr>
<tr>
<td>signed mean</td>
<td>-59.2</td>
<td>+14.8</td>
<td>-0.0362</td>
<td>+0.0051</td>
</tr>
</table>
The signed median is small beside the median absolute error, so the typical track is missed in either direction rather than bent systematically. The signed **mean** is not a bias here and should not be read as one: it is pulled by the heavy tails and changes sign against the median in both $`x`$ and $`y`$. That mistake was made once in this study, on 18 September 2026, and corrected the same day.
![Top: the absolute endpoint error per component against momentum, each hexagon coloured by how many of the 1,452 test tracks fall in it on a log scale, with the median per momentum bin in red. Bottom: the 10 to 20 GeV band, signed, with the median and the tail-pulled mean marked. The 64-step, two-stage network at its 18 September 2026 checkpoint.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/Block_E_single_network_chain/E3_Analysis/figures_writeup/case_study_components.png)
# 5. What the endpoint error is made of
Taking the same network apart step by step, with each step measured against the reference **from the state the network was actually given**, so that what is measured is what that step added and not what it inherited:
<table fit-page-width="true" header-row="true">
<tr>
<td>quantity</td>
<td>value</td>
</tr>
<tr>
<td>median endpoint error, radial</td>
<td>166.3 µm</td>
</tr>
<tr>
<td>what one step adds, position</td>
<td>0.415 µm</td>
</tr>
<tr>
<td>what one step adds, slope</td>
<td>0.00160 ×10⁻³</td>
</tr>
<tr>
<td>the position parts of all 64 steps, summed</td>
<td>29.2 µm</td>
</tr>
<tr>
<td>each step's x-slope error times the distance left to z1, summed</td>
<td>120.3 µm, which is 72 per cent of the endpoint error</td>
</tr>
<tr>
<td>coherence of a track's slope increments</td>
<td>0.375, against 0.125 for independent steps</td>
</tr>
<tr>
<td>median signed slope increment</td>
<td>-0.0000307 ×10⁻³, against a median size of 0.00160</td>
</tr>
</table>
**The endpoint error is the slope error of early steps carried the rest of the way.** The positions the steps get wrong sum to 29 µm, a sixth of the total; the slopes they get wrong, each multiplied by the distance that is left, account for 72 per cent of it. The increments are partly aligned, three times more than chance would give, but the median signed increment is 2 per cent of the median size, so this is a mild correlation and not a systematic bend.
**Completing the account with both slopes.** That 72 per cent is what an $`x`$-slope-only model explains while the measure is radial, so the decomposition was redone later with **both** slopes, on the extended networks:
<table fit-page-width="true" header-row="true">
<tr>
<td>network (extended)</td>
<td>endpoint \[µm\]</td>
<td>both slopes explain</td>
<td>x slope alone</td>
<td>positions alone</td>
<td>x part / y part \[µm\]</td>
<td>coherence tx / ty</td>
</tr>
<tr>
<td>64 steps, 2 stages</td>
<td>145.7</td>
<td>98.4 %</td>
<td>73.3 %</td>
<td>7.9 %</td>
<td>108 / 52</td>
<td>0.357 / 0.397</td>
</tr>
<tr>
<td>128 steps, 8 stages</td>
<td>122.2</td>
<td>98.3 %</td>
<td>63.4 %</td>
<td>10.8 %</td>
<td>79 / 59</td>
<td>0.312 / 0.432</td>
</tr>
<tr>
<td>256 steps, 16 stages</td>
<td>147.7</td>
<td>98.9 %</td>
<td>67.3 %</td>
<td>10.0 %</td>
<td>99 / 69</td>
<td>0.411 / 0.453</td>
</tr>
</table>
The two-slope lever-arm sum reproduces the endpoint error to 98.3 to 98.9 per cent, so **there is nothing in the endpoint error but the slope errors of the steps, weighted by how early they were made.** Per step, the slope errors of those three extended networks are 0.00130, 0.00055 and 0.00030 in $`t_x`$ and 0.00072, 0.00032 and 0.00018 in $`t_y`$, all $`\times 10^{-3}`$ in slope.
**What accuracy would be needed.** Inverting the lever-arm sum: a 10 µm endpoint would need the per-step slope error to fall about twelvefold, and a 1 µm endpoint about 120-fold.
![How the endpoint error is built up in the 64-step, two-stage network on the 1,452 test tracks. Left: what each step adds locally, against the z of the step. Middle: each step's slope error multiplied by the distance still to travel, which is what it costs at the fibre-tracker plane. Right: the distribution of the coherence of a track's slope increments, with the value expected for independent steps marked.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/Block_E_single_network_chain/E3_Analysis/figures_writeup/error_anatomy.png)
