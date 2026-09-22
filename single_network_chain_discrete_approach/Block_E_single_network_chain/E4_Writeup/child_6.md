# 1. Convergence, and the lesson
![Training loss, in blue on the left axis, and the validation error at the fibre-tracker plane, in red on the right axis, after every round, one panel per network, to the stop of 18 September 2026.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/Block_E_single_network_chain/E3_Analysis/figures_writeup/convergence_grid.png)
![The same, summarised: every network's training loss against restart on the left, and the validation error it buys on the right. Test tracks are not used here.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/Block_E_single_network_chain/E3_Analysis/figures_writeup/convergence_summary.png)
The two panels of the second figure carry the titles they were given on 18 September 2026, and the second of them is the mistake: *the loss on freshly drawn states stops falling, and the error it buys flattens with it*. The second half was not true. Under the plateau rule of protocol 8, applied to the round logs of that same evening, all eleven runs with twenty or more rounds were still improving by 5 to 24 per cent per ten rounds, and the fifteen extendable runs then improved by a further 12 to 42 per cent. The loss and the error decouple because the rescaling keeps the optimiser productive on the objective long past the point where the objective's remaining decrease buys anything at the endpoint.
Of the sixteen runs as they stand today, **nine satisfy the plateau rule**; six are still drifting down by a few per cent per ten rounds at 1,600 to 1,758 restarts, namely 64 steps with 4 stages, 128 steps with 8 stages and all four 256-step runs; and one, 2 steps with 2 stages, was never extended because it is at the scheme's limit. So the headline numbers are best read as "after 1,000 to 1,758 restarts" rather than as a strict plateau. The residual drift is a few micrometres, an order of magnitude below the differences the study is about.
# 2. No over-training
Every network was carried through the chain on all three splits, and the trained objective was evaluated on 32,000 reference states drawn from each. The endpoint errors in this table are in the **max metric**, $`\max(|\Delta x|, |\Delta y|)`$, which is what this check records; they run 10 to 20 per cent below the radial numbers used elsewhere.
<table fit-page-width="true" header-row="true">
<tr>
<td>network</td>
<td>restarts</td>
<td>training \[µm\]</td>
<td>validation \[µm\]</td>
<td>test \[µm\]</td>
<td>test / training</td>
</tr>
<tr>
<td>2 steps, 2 stages</td>
<td>109</td>
<td>9,632</td>
<td>9,563</td>
<td>9,249</td>
<td>0.960</td>
</tr>
<tr>
<td>2 steps, 4 stages</td>
<td>159</td>
<td>263.5</td>
<td>268.7</td>
<td>260.9</td>
<td>0.990</td>
</tr>
<tr>
<td>2 steps, 8 stages</td>
<td>130</td>
<td>171.6</td>
<td>171.3</td>
<td>168.4</td>
<td>0.982</td>
</tr>
<tr>
<td>2 steps, 16 stages</td>
<td>106</td>
<td>197.0</td>
<td>209.2</td>
<td>201.9</td>
<td>1.025</td>
</tr>
<tr>
<td>64 steps, 2 stages</td>
<td>760</td>
<td>160.9</td>
<td>164.4</td>
<td>152.0</td>
<td>0.945</td>
</tr>
<tr>
<td>64 steps, 4 stages</td>
<td>714</td>
<td>159.0</td>
<td>165.9</td>
<td>161.8</td>
<td>1.017</td>
</tr>
<tr>
<td>64 steps, 8 stages</td>
<td>657</td>
<td>193.4</td>
<td>200.0</td>
<td>192.0</td>
<td>0.993</td>
</tr>
<tr>
<td>64 steps, 16 stages</td>
<td>546</td>
<td>173.3</td>
<td>172.0</td>
<td>172.7</td>
<td>0.997</td>
</tr>
<tr>
<td>128 steps, 2 stages</td>
<td>760</td>
<td>188.7</td>
<td>190.9</td>
<td>189.9</td>
<td>1.006</td>
</tr>
<tr>
<td>128 steps, 4 stages</td>
<td>713</td>
<td>181.6</td>
<td>178.8</td>
<td>174.5</td>
<td>0.961</td>
</tr>
<tr>
<td>128 steps, 8 stages</td>
<td>599</td>
<td>147.5</td>
<td>143.9</td>
<td>144.2</td>
<td>0.978</td>
</tr>
<tr>
<td>128 steps, 16 stages</td>
<td>383</td>
<td>233.0</td>
<td>230.5</td>
<td>226.8</td>
<td>0.974</td>
</tr>
<tr>
<td>256 steps, 2 stages</td>
<td>758</td>
<td>168.0</td>
<td>167.2</td>
<td>169.5</td>
<td>1.009</td>
</tr>
<tr>
<td>256 steps, 4 stages</td>
<td>709</td>
<td>181.7</td>
<td>182.5</td>
<td>177.7</td>
<td>0.978</td>
</tr>
<tr>
<td>256 steps, 8 stages</td>
<td>615</td>
<td>180.5</td>
<td>183.8</td>
<td>176.4</td>
<td>0.977</td>
</tr>
<tr>
<td>256 steps, 16 stages</td>
<td>550</td>
<td>128.8</td>
<td>127.5</td>
<td>122.5</td>
<td>0.952</td>
</tr>
</table>
The test endpoint median is **0.945 to 1.025** times the training median, with a median ratio of 0.980, and the loss on unseen tracks is if anything lower than on training tracks, a ratio of 0.13 to 1.37 with a median of 0.28. A label-free objective with about twenty thousand parameters is not fitting itself to the particles it has seen. The loss ratio is tail-heavy and the training pool has eight times more tracks, so it catches more extreme ones, which is why that ratio sits below one.
# 3. Cost
Microseconds per track for the whole crossing, double precision, one thread, on a shared host:
<table fit-page-width="true" header-row="true" header-column="true">
<tr>
<td>µs per track</td>
<td>q = 2</td>
<td>q = 4</td>
<td>q = 8</td>
<td>q = 16</td>
<td>parameters</td>
</tr>
<tr>
<td>2 steps</td>
<td>26.4</td>
<td>26.0</td>
<td>26.5</td>
<td>28.4</td>
<td>18,956 to 26,180</td>
</tr>
<tr>
<td>64 steps</td>
<td>803.2</td>
<td>813.9</td>
<td>833.4</td>
<td>888.4</td>
<td>same</td>
</tr>
<tr>
<td>128 steps</td>
<td>1,592.3</td>
<td>1,657.6</td>
<td>1,693.4</td>
<td>1,778.1</td>
<td>same</td>
</tr>
<tr>
<td>256 steps</td>
<td>3,262.6</td>
<td>3,311.5</td>
<td>3,324.6</td>
<td>3,546.2</td>
<td>same</td>
</tr>
</table>
Cost is set by the step count, because a crossing is that many forward passes, and barely by the stage count, because stages only widen the last layer: 18,956 parameters at 2 stages against 26,180 at 16, a 38 per cent difference in size that costs 6 to 9 per cent in time. Accuracy at 64 steps and above is flat, so **the cheap chains win**: 64 steps with 4 stages is both the best cell after the extension and four times cheaper than the 256-step chains. These figures are eager PyTorch on one core and are indicative only.
# 4. Where the loss actually goes
For the 64-step, two-stage network, the share of the trained residual carried by each momentum band, beside the endpoint error of the same tracks:
<table fit-page-width="true" header-row="true">
<tr>
<td>momentum band</td>
<td>test tracks</td>
<td>share of the loss</td>
<td>median endpoint error \[µm\]</td>
</tr>
<tr>
<td>1-2 GeV</td>
<td>5</td>
<td>0.19 %</td>
<td>1,468</td>
</tr>
<tr>
<td>2-3 GeV</td>
<td>159</td>
<td>31.8 %</td>
<td>464</td>
</tr>
<tr>
<td>3-5 GeV</td>
<td>341</td>
<td>65.1 %</td>
<td>242</td>
</tr>
<tr>
<td>5-10 GeV</td>
<td>430</td>
<td>2.3 %</td>
<td>124 to 160</td>
</tr>
<tr>
<td>10-50 GeV</td>
<td>487</td>
<td>0.63 %</td>
<td>107 to 146</td>
</tr>
<tr>
<td>50-200 GeV</td>
<td>30</td>
<td>0.03 %</td>
<td>206 to 829</td>
</tr>
</table>
**96.9 per cent of the trained residual comes from the 2 to 5 GeV band**, which holds 500 of the 1,452 test tracks; everything above 5 GeV, which is 947 tracks, contributes 2.9 per cent, and the 10 to 50 GeV band contributes 0.63 per cent. The pooled spread of protocol 4 is the reason: a residual is measured against the spread of the whole population, so the states with the largest residuals dominate, and those are the soft, hard-bending tracks. The loss is not wrong, it is answering a different question from the one the experiment asks, and the study that follows this one changes exactly that.
![Left: the median trained residual per track against momentum, in blue, with the median endpoint error of the same tracks in red. Right: the share of the total loss carried by each momentum band. The 64-step, two-stage network at its 18 September 2026 checkpoint, test tracks.](https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/Block_E_single_network_chain/E3_Analysis/figures_writeup/case_study_loss_vs_p.png)
