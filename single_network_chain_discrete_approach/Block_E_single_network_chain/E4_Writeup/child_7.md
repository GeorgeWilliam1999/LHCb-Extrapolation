Every run, in full. The first two tables are the networks **after the extension**, scored on 22 September 2026; the third is the same runs **as stopped by hand on 18 September 2026**, which is what every figure in this write-up shows.
# 1. Training and stopping, after the extension
The headline is the median validation error over a run's last ten rounds, not its final checkpoint, with the spread of those ten rounds beside it. "Change over the previous ten" is negative when the run was still improving; "finished" is the plateau rule of protocol 8 holding at each of the last three rounds.
<table fit-page-width="true" header-row="true">
<tr>
<td>network</td>
<td>restarts</td>
<td>rounds</td>
<td>training wall \[h\]</td>
<td>validation, last ten rounds \[µm\]</td>
<td>spread</td>
<td>change over the previous ten</td>
<td>finished</td>
</tr>
<tr>
<td>2 steps, 2 stages</td>
<td>109</td>
<td>6</td>
<td>4.5</td>
<td>9,580</td>
<td>0.7 %</td>
<td>not extended</td>
<td>at the scheme's own error</td>
</tr>
<tr>
<td>2 steps, 4 stages</td>
<td>175</td>
<td>23</td>
<td>7.1</td>
<td>271.5</td>
<td>0.7 %</td>
<td>-1.5 %</td>
<td>yes</td>
</tr>
<tr>
<td>2 steps, 8 stages</td>
<td>154</td>
<td>23</td>
<td>4.2</td>
<td>170.3</td>
<td>1.2 %</td>
<td>-3.2 %</td>
<td>yes</td>
</tr>
<tr>
<td>2 steps, 16 stages</td>
<td>166</td>
<td>31</td>
<td>7.2</td>
<td>196.8</td>
<td>2.0 %</td>
<td>-4.4 %</td>
<td>yes</td>
</tr>
<tr>
<td>64 steps, 2 stages</td>
<td>1,260</td>
<td>50</td>
<td>31.2</td>
<td>126.6</td>
<td>17.5 %</td>
<td>-4.2 %</td>
<td>yes</td>
</tr>
<tr>
<td>64 steps, 4 stages</td>
<td>1,714</td>
<td>68</td>
<td>47.8</td>
<td>120.3</td>
<td>11.0 %</td>
<td>-7.1 %</td>
<td>no, still improving</td>
</tr>
<tr>
<td>64 steps, 8 stages</td>
<td>1,157</td>
<td>46</td>
<td>38.7</td>
<td>136.3</td>
<td>9.0 %</td>
<td>+0.2 %</td>
<td>yes</td>
</tr>
<tr>
<td>64 steps, 16 stages</td>
<td>1,046</td>
<td>41</td>
<td>59.3</td>
<td>163.6</td>
<td>10.6 %</td>
<td>+1.5 %</td>
<td>yes</td>
</tr>
<tr>
<td>128 steps, 2 stages</td>
<td>1,260</td>
<td>50</td>
<td>31.9</td>
<td>135.7</td>
<td>11.2 %</td>
<td>+1.0 %</td>
<td>yes</td>
</tr>
<tr>
<td>128 steps, 4 stages</td>
<td>1,213</td>
<td>48</td>
<td>30.9</td>
<td>157.1</td>
<td>20.6 %</td>
<td>+3.0 %</td>
<td>yes</td>
</tr>
<tr>
<td>128 steps, 8 stages</td>
<td>1,599</td>
<td>63</td>
<td>54.7</td>
<td>128.7</td>
<td>12.4 %</td>
<td>-7.2 %</td>
<td>no, still improving</td>
</tr>
<tr>
<td>128 steps, 16 stages</td>
<td>1,383</td>
<td>55</td>
<td>77.0</td>
<td>138.3</td>
<td>9.3 %</td>
<td>-4.1 %</td>
<td>yes</td>
</tr>
<tr>
<td>256 steps, 2 stages</td>
<td>1,758</td>
<td>70</td>
<td>43.0</td>
<td>122.5</td>
<td>11.4 %</td>
<td>-6.0 %</td>
<td>no, still improving</td>
</tr>
<tr>
<td>256 steps, 4 stages</td>
<td>1,709</td>
<td>68</td>
<td>43.8</td>
<td>129.4</td>
<td>14.8 %</td>
<td>-2.6 %</td>
<td>no, still improving</td>
</tr>
<tr>
<td>256 steps, 8 stages</td>
<td>1,615</td>
<td>64</td>
<td>65.5</td>
<td>130.1</td>
<td>9.3 %</td>
<td>-5.2 %</td>
<td>no, still improving</td>
</tr>
<tr>
<td>256 steps, 16 stages</td>
<td>1,050</td>
<td>42</td>
<td>63.2</td>
<td>149.5</td>
<td>9.8 %</td>
<td>-10.9 %</td>
<td>no, still improving</td>
</tr>
</table>
Sixteen runs, 610 hours of single-core training wall time in all. Nine are finished by the rule, six were still drifting down when the farm was emptied, and one was never extended because it sits at the exact scheme's own error.
# 2. The endpoint, after the extension
Radial endpoint error at $`z_1`$ on the 1,452 test tracks. The band is 10 to 50 GeV, 487 tracks; "below 5 GeV" is 500 tracks.
<table fit-page-width="true" header-row="true">
<tr>
<td>network</td>
<td>median \[µm\]</td>
<td>10 to 50 GeV \[µm\]</td>
<td>below 5 GeV \[µm\]</td>
<td>95th percentile \[µm\]</td>
</tr>
<tr>
<td>2 steps, 2 stages</td>
<td>9,250.0</td>
<td>3,630.7</td>
<td>21,960</td>
<td>39,347</td>
</tr>
<tr>
<td>2 steps, 4 stages</td>
<td>284.6</td>
<td>157.5</td>
<td>597.6</td>
<td>1,966.9</td>
</tr>
<tr>
<td>2 steps, 8 stages</td>
<td>183.9</td>
<td>111.8</td>
<td>334.0</td>
<td>1,415.2</td>
</tr>
<tr>
<td>2 steps, 16 stages</td>
<td>203.3</td>
<td>138.1</td>
<td>400.0</td>
<td>1,558.8</td>
</tr>
<tr>
<td>64 steps, 2 stages</td>
<td>145.7</td>
<td>108.1</td>
<td>241.8</td>
<td>897.8</td>
</tr>
<tr>
<td>64 steps, 4 stages</td>
<td>**118.6**</td>
<td>83.3</td>
<td>206.4</td>
<td>807.4</td>
</tr>
<tr>
<td>64 steps, 8 stages</td>
<td>137.9</td>
<td>90.5</td>
<td>270.0</td>
<td>936.4</td>
</tr>
<tr>
<td>64 steps, 16 stages</td>
<td>164.4</td>
<td>104.8</td>
<td>369.5</td>
<td>1,284.2</td>
</tr>
<tr>
<td>128 steps, 2 stages</td>
<td>141.6</td>
<td>106.1</td>
<td>274.8</td>
<td>1,035.1</td>
</tr>
<tr>
<td>128 steps, 4 stages</td>
<td>154.4</td>
<td>119.5</td>
<td>273.4</td>
<td>1,020.8</td>
</tr>
<tr>
<td>128 steps, 8 stages</td>
<td>122.2</td>
<td>**70.5**</td>
<td>240.5</td>
<td>927.7</td>
</tr>
<tr>
<td>128 steps, 16 stages</td>
<td>143.1</td>
<td>98.8</td>
<td>262.9</td>
<td>1,063.5</td>
</tr>
<tr>
<td>256 steps, 2 stages</td>
<td>126.3</td>
<td>70.7</td>
<td>309.8</td>
<td>1,031.5</td>
</tr>
<tr>
<td>256 steps, 4 stages</td>
<td>150.3</td>
<td>100.0</td>
<td>290.5</td>
<td>1,061.6</td>
</tr>
<tr>
<td>256 steps, 8 stages</td>
<td>128.2</td>
<td>85.2</td>
<td>230.8</td>
<td>1,043.7</td>
</tr>
<tr>
<td>256 steps, 16 stages</td>
<td>147.7</td>
<td>89.6</td>
<td>285.9</td>
<td>1,032.1</td>
</tr>
</table>
The two bold cells are the best overall and the best in the band. Both sit inside the round-to-round spread of their neighbours, so they are where the grid happens to have landed rather than a preference for those settings.
# 3. As stopped on 18 September 2026, with the comparators
This is the table every figure in this write-up describes. "Exact scheme" is the same scheme chained at the same step and stage count with no network in it; "per-step chains" is the previous study at the settings the two share, re-scored radially on the same test tracks. The final loss is the unscaled objective at the stop.
<table fit-page-width="true" header-row="true">
<tr>
<td>network</td>
<td>restarts</td>
<td>rounds</td>
<td>final loss</td>
<td>median \[µm\]</td>
<td>95th pct \[µm\]</td>
<td>99th pct \[µm\]</td>
<td>exact scheme \[µm\]</td>
<td>per-step chains \[µm\]</td>
</tr>
<tr>
<td>2 steps, 2 stages</td>
<td>109</td>
<td>6</td>
<td>1.2e-06</td>
<td>9,250.0</td>
<td>39,347</td>
<td>70,104</td>
<td>9,270.9</td>
<td>not run</td>
</tr>
<tr>
<td>2 steps, 4 stages</td>
<td>159</td>
<td>19</td>
<td>9.9e-07</td>
<td>283.8</td>
<td>2,011</td>
<td>5,414</td>
<td>195.4</td>
<td>not run</td>
</tr>
<tr>
<td>2 steps, 8 stages</td>
<td>130</td>
<td>17</td>
<td>1.3e-06</td>
<td>188.1</td>
<td>1,428</td>
<td>4,355</td>
<td>7.47</td>
<td>not run</td>
</tr>
<tr>
<td>2 steps, 16 stages</td>
<td>106</td>
<td>16</td>
<td>1.3e-06</td>
<td>223.6</td>
<td>1,683</td>
<td>3,708</td>
<td>5.17</td>
<td>not run</td>
</tr>
<tr>
<td>64 steps, 2 stages</td>
<td>760</td>
<td>31</td>
<td>1.1e-09</td>
<td>166.3</td>
<td>1,154</td>
<td>3,304</td>
<td>0.147</td>
<td>2,304</td>
</tr>
<tr>
<td>64 steps, 4 stages</td>
<td>714</td>
<td>29</td>
<td>7.1e-10</td>
<td>172.4</td>
<td>1,051</td>
<td>2,915</td>
<td>0.478</td>
<td>2,442</td>
</tr>
<tr>
<td>64 steps, 8 stages</td>
<td>657</td>
<td>27</td>
<td>1.1e-09</td>
<td>204.2</td>
<td>1,249</td>
<td>3,790</td>
<td>0.103</td>
<td>2,346</td>
</tr>
<tr>
<td>64 steps, 16 stages</td>
<td>546</td>
<td>22</td>
<td>7.2e-10</td>
<td>194.2</td>
<td>1,207</td>
<td>3,914</td>
<td>0.072</td>
<td>2,418</td>
</tr>
<tr>
<td>128 steps, 2 stages</td>
<td>760</td>
<td>31</td>
<td>2.7e-10</td>
<td>207.0</td>
<td>1,206</td>
<td>3,779</td>
<td>0.511</td>
<td>2,940</td>
</tr>
<tr>
<td>128 steps, 4 stages</td>
<td>713</td>
<td>29</td>
<td>2.6e-10</td>
<td>195.6</td>
<td>1,246</td>
<td>4,215</td>
<td>0.116</td>
<td>3,296</td>
</tr>
<tr>
<td>128 steps, 8 stages</td>
<td>599</td>
<td>24</td>
<td>1.4e-10</td>
<td>158.6</td>
<td>1,090</td>
<td>4,043</td>
<td>0.011</td>
<td>3,212</td>
</tr>
<tr>
<td>128 steps, 16 stages</td>
<td>383</td>
<td>16</td>
<td>2.6e-10</td>
<td>246.2</td>
<td>1,410</td>
<td>4,117</td>
<td>0.00018</td>
<td>3,827</td>
</tr>
<tr>
<td>256 steps, 2 stages</td>
<td>758</td>
<td>31</td>
<td>7.2e-11</td>
<td>185.3</td>
<td>1,319</td>
<td>3,410</td>
<td>0.228</td>
<td>not run</td>
</tr>
<tr>
<td>256 steps, 4 stages</td>
<td>709</td>
<td>29</td>
<td>5.0e-11</td>
<td>195.6</td>
<td>1,275</td>
<td>4,843</td>
<td>0.030</td>
<td>not run</td>
</tr>
<tr>
<td>256 steps, 8 stages</td>
<td>615</td>
<td>25</td>
<td>4.2e-11</td>
<td>193.6</td>
<td>1,204</td>
<td>3,090</td>
<td>0.00043</td>
<td>not run</td>
</tr>
<tr>
<td>256 steps, 16 stages</td>
<td>550</td>
<td>23</td>
<td>3.0e-11</td>
<td>136.8</td>
<td>1,082</td>
<td>3,606</td>
<td>0.00105</td>
<td>not run</td>
</tr>
</table>
The final losses fall by five orders of magnitude from 2 steps to 256 steps, because a shorter step is an easier equation to satisfy, and they say nothing at all about the endpoint error, which is flat across the same range. That is the decoupling of protocol 8 in one column.
