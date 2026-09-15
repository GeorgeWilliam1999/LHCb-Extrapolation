<!-- TITLE: Checks and cost: convergence, q/p, validation against test, forward cost, the twin, continuity with Block A -->
## Convergence and cost per chain
<table header-row="true" fit-page-width="true">
	<tr><td>N</td><td>q</td><td>all legs converged</td><td>total restarts</td><td>training wall \[s\]</td></tr>
	<tr><td>1</td><td>1</td><td>True</td><td>35</td><td>264</td></tr>
	<tr><td>1</td><td>2</td><td>True</td><td>61</td><td>447</td></tr>
	<tr><td>1</td><td>3</td><td>True</td><td>50</td><td>387</td></tr>
	<tr><td>1</td><td>4</td><td>True</td><td>86</td><td>695</td></tr>
	<tr><td>1</td><td>5</td><td>True</td><td>61</td><td>517</td></tr>
	<tr><td>1</td><td>6</td><td>True</td><td>79</td><td>688</td></tr>
	<tr><td>1</td><td>7</td><td>True</td><td>37</td><td>324</td></tr>
	<tr><td>1</td><td>8</td><td>True</td><td>55</td><td>514</td></tr>
	<tr><td>1</td><td>9</td><td>True</td><td>72</td><td>707</td></tr>
	<tr><td>1</td><td>10</td><td>True</td><td>90</td><td>908</td></tr>
	<tr><td>1</td><td>11</td><td>True</td><td>67</td><td>820</td></tr>
	<tr><td>1</td><td>12</td><td>True</td><td>52</td><td>619</td></tr>
	<tr><td>1</td><td>13</td><td>True</td><td>63</td><td>789</td></tr>
	<tr><td>1</td><td>14</td><td>True</td><td>65</td><td>1,440</td></tr>
	<tr><td>1</td><td>15</td><td>True</td><td>63</td><td>1,382</td></tr>
	<tr><td>1</td><td>16</td><td>True</td><td>71</td><td>939</td></tr>
	<tr><td>1</td><td>17</td><td>True</td><td>84</td><td>1,134</td></tr>
	<tr><td>1</td><td>18</td><td>True</td><td>61</td><td>799</td></tr>
	<tr><td>1</td><td>19</td><td>True</td><td>47</td><td>707</td></tr>
	<tr><td>1</td><td>20</td><td>True</td><td>58</td><td>776</td></tr>
	<tr><td>4</td><td>1</td><td>True</td><td>74</td><td>553</td></tr>
	<tr><td>4</td><td>2</td><td>True</td><td>71</td><td>581</td></tr>
	<tr><td>4</td><td>3</td><td>True</td><td>67</td><td>576</td></tr>
	<tr><td>4</td><td>4</td><td>True</td><td>69</td><td>619</td></tr>
	<tr><td>4</td><td>5</td><td>True</td><td>64</td><td>519</td></tr>
	<tr><td>4</td><td>6</td><td>True</td><td>63</td><td>543</td></tr>
	<tr><td>4</td><td>7</td><td>True</td><td>71</td><td>674</td></tr>
	<tr><td>4</td><td>8</td><td>True</td><td>65</td><td>632</td></tr>
	<tr><td>4</td><td>9</td><td>True</td><td>60</td><td>604</td></tr>
	<tr><td>4</td><td>10</td><td>True</td><td>67</td><td>702</td></tr>
	<tr><td>4</td><td>11</td><td>True</td><td>69</td><td>738</td></tr>
	<tr><td>4</td><td>12</td><td>True</td><td>70</td><td>786</td></tr>
	<tr><td>4</td><td>13</td><td>True</td><td>68</td><td>790</td></tr>
	<tr><td>4</td><td>14</td><td>True</td><td>71</td><td>854</td></tr>
	<tr><td>4</td><td>15</td><td>True</td><td>79</td><td>974</td></tr>
	<tr><td>4</td><td>16</td><td>True</td><td>76</td><td>933</td></tr>
	<tr><td>4</td><td>17</td><td>True</td><td>73</td><td>1,078</td></tr>
	<tr><td>4</td><td>18</td><td>True</td><td>86</td><td>1,142</td></tr>
	<tr><td>4</td><td>19</td><td>True</td><td>81</td><td>1,277</td></tr>
	<tr><td>4</td><td>20</td><td>True</td><td>77</td><td>1,204</td></tr>
	<tr><td>16</td><td>1</td><td>True</td><td>158</td><td>1,716</td></tr>
	<tr><td>16</td><td>2</td><td>True</td><td>145</td><td>1,690</td></tr>
	<tr><td>16</td><td>3</td><td>True</td><td>162</td><td>1,930</td></tr>
	<tr><td>16</td><td>4</td><td>True</td><td>159</td><td>1,463</td></tr>
	<tr><td>16</td><td>5</td><td>True</td><td>160</td><td>2,107</td></tr>
	<tr><td>16</td><td>6</td><td>True</td><td>168</td><td>1,688</td></tr>
	<tr><td>16</td><td>7</td><td>True</td><td>161</td><td>2,187</td></tr>
	<tr><td>16</td><td>8</td><td>True</td><td>174</td><td>1,847</td></tr>
	<tr><td>16</td><td>9</td><td>True</td><td>152</td><td>1,678</td></tr>
	<tr><td>16</td><td>10</td><td>True</td><td>156</td><td>1,777</td></tr>
	<tr><td>16</td><td>11</td><td>True</td><td>162</td><td>1,778</td></tr>
	<tr><td>16</td><td>12</td><td>True</td><td>198</td><td>2,466</td></tr>
	<tr><td>16</td><td>13</td><td>True</td><td>177</td><td>2,301</td></tr>
	<tr><td>16</td><td>14</td><td>True</td><td>179</td><td>2,363</td></tr>
	<tr><td>16</td><td>15</td><td>True</td><td>174</td><td>2,436</td></tr>
	<tr><td>16</td><td>16</td><td>True</td><td>168</td><td>2,322</td></tr>
	<tr><td>16</td><td>17</td><td>True</td><td>179</td><td>2,659</td></tr>
	<tr><td>16</td><td>18</td><td>True</td><td>179</td><td>2,685</td></tr>
	<tr><td>16</td><td>19</td><td>True</td><td>179</td><td>2,828</td></tr>
	<tr><td>16</td><td>20</td><td>True</td><td>190</td><td>2,972</td></tr>
	<tr><td>64</td><td>1</td><td>True</td><td>676</td><td>7,944</td></tr>
	<tr><td>64</td><td>2</td><td>True</td><td>771</td><td>9,844</td></tr>
	<tr><td>64</td><td>3</td><td>True</td><td>723</td><td>7,925</td></tr>
	<tr><td>64</td><td>4</td><td>True</td><td>755</td><td>9,672</td></tr>
	<tr><td>64</td><td>5</td><td>True</td><td>862</td><td>10,193</td></tr>
	<tr><td>64</td><td>6</td><td>True</td><td>854</td><td>11,546</td></tr>
	<tr><td>64</td><td>7</td><td>True</td><td>826</td><td>10,819</td></tr>
	<tr><td>64</td><td>8</td><td>True</td><td>868</td><td>13,944</td></tr>
	<tr><td>64</td><td>9</td><td>True</td><td>860</td><td>11,842</td></tr>
	<tr><td>64</td><td>10</td><td>True</td><td>867</td><td>13,503</td></tr>
	<tr><td>64</td><td>11</td><td>True</td><td>969</td><td>16,372</td></tr>
	<tr><td>64</td><td>12</td><td>True</td><td>939</td><td>17,068</td></tr>
	<tr><td>64</td><td>13</td><td>True</td><td>909</td><td>17,832</td></tr>
	<tr><td>64</td><td>14</td><td>True</td><td>908</td><td>18,244</td></tr>
	<tr><td>64</td><td>15</td><td>True</td><td>924</td><td>16,045</td></tr>
	<tr><td>64</td><td>16</td><td>True</td><td>890</td><td>18,058</td></tr>
	<tr><td>64</td><td>17</td><td>True</td><td>957</td><td>19,294</td></tr>
	<tr><td>64</td><td>18</td><td>True</td><td>900</td><td>21,029</td></tr>
	<tr><td>64</td><td>19</td><td>True</td><td>994</td><td>21,983</td></tr>
	<tr><td>64</td><td>20</td><td>True</td><td>876</td><td>19,301</td></tr>
	<tr><td>128</td><td>1</td><td>True</td><td>2044</td><td>21,818</td></tr>
	<tr><td>128</td><td>2</td><td>True</td><td>2243</td><td>25,073</td></tr>
	<tr><td>128</td><td>3</td><td>True</td><td>2431</td><td>26,930</td></tr>
	<tr><td>128</td><td>4</td><td>True</td><td>2366</td><td>26,662</td></tr>
	<tr><td>128</td><td>5</td><td>True</td><td>2596</td><td>31,714</td></tr>
	<tr><td>128</td><td>6</td><td>True</td><td>2541</td><td>32,269</td></tr>
	<tr><td>128</td><td>7</td><td>True</td><td>2421</td><td>32,351</td></tr>
	<tr><td>128</td><td>8</td><td>True</td><td>2715</td><td>36,253</td></tr>
	<tr><td>128</td><td>9</td><td>True</td><td>2619</td><td>35,236</td></tr>
	<tr><td>128</td><td>10</td><td>True</td><td>2649</td><td>38,743</td></tr>
	<tr><td>128</td><td>11</td><td>True</td><td>2864</td><td>41,078</td></tr>
	<tr><td>128</td><td>12</td><td>True</td><td>2993</td><td>46,585</td></tr>
	<tr><td>128</td><td>13</td><td>True</td><td>2969</td><td>48,318</td></tr>
	<tr><td>128</td><td>14</td><td>True</td><td>2793</td><td>46,175</td></tr>
	<tr><td>128</td><td>15</td><td>True</td><td>3004</td><td>50,525</td></tr>
	<tr><td>128</td><td>16</td><td>True</td><td>3166</td><td>55,646</td></tr>
	<tr><td>128</td><td>17</td><td>True</td><td>3247</td><td>56,057</td></tr>
	<tr><td>128</td><td>18</td><td>True</td><td>3053</td><td>58,148</td></tr>
	<tr><td>128</td><td>19</td><td>True</td><td>3036</td><td>55,660</td></tr>
	<tr><td>128</td><td>20</td><td>True</td><td>3006</td><td>60,118</td></tr>
</table>
## The q/p check: largest change in q/p along the chain
<table header-row="true" fit-page-width="true">
	<tr><td>N</td><td>q</td><td>test</td><td>validation</td></tr>
	<tr><td>1</td><td>1</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>2</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>3</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>4</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>5</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>6</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>7</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>8</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>9</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>10</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>11</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>12</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>13</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>14</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>15</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>16</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>17</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>18</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>19</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>20</td><td>0</td><td>0</td></tr>
	<tr><td>4</td><td>1</td><td>0</td><td>0</td></tr>
	<tr><td>4</td><td>2</td><td>0</td><td>0</td></tr>
	<tr><td>4</td><td>3</td><td>0</td><td>0</td></tr>
	<tr><td>4</td><td>4</td><td>0</td><td>0</td></tr>
	<tr><td>4</td><td>5</td><td>0</td><td>0</td></tr>
	<tr><td>4</td><td>6</td><td>0</td><td>0</td></tr>
	<tr><td>4</td><td>7</td><td>0</td><td>0</td></tr>
	<tr><td>4</td><td>8</td><td>0</td><td>0</td></tr>
	<tr><td>4</td><td>9</td><td>0</td><td>0</td></tr>
	<tr><td>4</td><td>10</td><td>0</td><td>0</td></tr>
	<tr><td>4</td><td>11</td><td>0</td><td>0</td></tr>
	<tr><td>4</td><td>12</td><td>0</td><td>0</td></tr>
	<tr><td>4</td><td>13</td><td>0</td><td>0</td></tr>
	<tr><td>4</td><td>14</td><td>0</td><td>0</td></tr>
	<tr><td>4</td><td>15</td><td>0</td><td>0</td></tr>
	<tr><td>4</td><td>16</td><td>0</td><td>0</td></tr>
	<tr><td>4</td><td>17</td><td>0</td><td>0</td></tr>
	<tr><td>4</td><td>18</td><td>0</td><td>0</td></tr>
	<tr><td>4</td><td>19</td><td>0</td><td>0</td></tr>
	<tr><td>4</td><td>20</td><td>0</td><td>0</td></tr>
	<tr><td>16</td><td>1</td><td>0</td><td>0</td></tr>
	<tr><td>16</td><td>2</td><td>0</td><td>0</td></tr>
	<tr><td>16</td><td>3</td><td>0</td><td>0</td></tr>
	<tr><td>16</td><td>4</td><td>0</td><td>0</td></tr>
	<tr><td>16</td><td>5</td><td>0</td><td>0</td></tr>
	<tr><td>16</td><td>6</td><td>0</td><td>0</td></tr>
	<tr><td>16</td><td>7</td><td>0</td><td>0</td></tr>
	<tr><td>16</td><td>8</td><td>0</td><td>0</td></tr>
	<tr><td>16</td><td>9</td><td>0</td><td>0</td></tr>
	<tr><td>16</td><td>10</td><td>0</td><td>0</td></tr>
	<tr><td>16</td><td>11</td><td>0</td><td>0</td></tr>
	<tr><td>16</td><td>12</td><td>0</td><td>0</td></tr>
	<tr><td>16</td><td>13</td><td>0</td><td>0</td></tr>
	<tr><td>16</td><td>14</td><td>0</td><td>0</td></tr>
	<tr><td>16</td><td>15</td><td>0</td><td>0</td></tr>
	<tr><td>16</td><td>16</td><td>0</td><td>0</td></tr>
	<tr><td>16</td><td>17</td><td>0</td><td>0</td></tr>
	<tr><td>16</td><td>18</td><td>0</td><td>0</td></tr>
	<tr><td>16</td><td>19</td><td>0</td><td>0</td></tr>
	<tr><td>16</td><td>20</td><td>0</td><td>0</td></tr>
	<tr><td>64</td><td>1</td><td>0</td><td>0</td></tr>
	<tr><td>64</td><td>2</td><td>0</td><td>0</td></tr>
	<tr><td>64</td><td>3</td><td>0</td><td>0</td></tr>
	<tr><td>64</td><td>4</td><td>0</td><td>0</td></tr>
	<tr><td>64</td><td>5</td><td>0</td><td>0</td></tr>
	<tr><td>64</td><td>6</td><td>0</td><td>0</td></tr>
	<tr><td>64</td><td>7</td><td>0</td><td>0</td></tr>
	<tr><td>64</td><td>8</td><td>0</td><td>0</td></tr>
	<tr><td>64</td><td>9</td><td>0</td><td>0</td></tr>
	<tr><td>64</td><td>10</td><td>0</td><td>0</td></tr>
	<tr><td>64</td><td>11</td><td>0</td><td>0</td></tr>
	<tr><td>64</td><td>12</td><td>0</td><td>0</td></tr>
	<tr><td>64</td><td>13</td><td>0</td><td>0</td></tr>
	<tr><td>64</td><td>14</td><td>0</td><td>0</td></tr>
	<tr><td>64</td><td>15</td><td>0</td><td>0</td></tr>
	<tr><td>64</td><td>16</td><td>0</td><td>0</td></tr>
	<tr><td>64</td><td>17</td><td>0</td><td>0</td></tr>
	<tr><td>64</td><td>18</td><td>0</td><td>0</td></tr>
	<tr><td>64</td><td>19</td><td>0</td><td>0</td></tr>
	<tr><td>64</td><td>20</td><td>0</td><td>0</td></tr>
	<tr><td>128</td><td>1</td><td>0</td><td>0</td></tr>
	<tr><td>128</td><td>2</td><td>0</td><td>0</td></tr>
	<tr><td>128</td><td>3</td><td>0</td><td>0</td></tr>
	<tr><td>128</td><td>4</td><td>0</td><td>0</td></tr>
	<tr><td>128</td><td>5</td><td>0</td><td>0</td></tr>
	<tr><td>128</td><td>6</td><td>0</td><td>0</td></tr>
	<tr><td>128</td><td>7</td><td>0</td><td>0</td></tr>
	<tr><td>128</td><td>8</td><td>0</td><td>0</td></tr>
	<tr><td>128</td><td>9</td><td>0</td><td>0</td></tr>
	<tr><td>128</td><td>10</td><td>0</td><td>0</td></tr>
	<tr><td>128</td><td>11</td><td>0</td><td>0</td></tr>
	<tr><td>128</td><td>12</td><td>0</td><td>0</td></tr>
	<tr><td>128</td><td>13</td><td>0</td><td>0</td></tr>
	<tr><td>128</td><td>14</td><td>0</td><td>0</td></tr>
	<tr><td>128</td><td>15</td><td>0</td><td>0</td></tr>
	<tr><td>128</td><td>16</td><td>0</td><td>0</td></tr>
	<tr><td>128</td><td>17</td><td>0</td><td>0</td></tr>
	<tr><td>128</td><td>18</td><td>0</td><td>0</td></tr>
	<tr><td>128</td><td>19</td><td>0</td><td>0</td></tr>
	<tr><td>128</td><td>20</td><td>0</td><td>0</td></tr>
</table>
## Validation against test, endpoint median [µm]
<table header-row="true" fit-page-width="true">
	<tr><td>N</td><td>q</td><td>validation</td><td>test</td></tr>
	<tr><td>1</td><td>1</td><td>173,575</td><td>171,524</td></tr>
	<tr><td>1</td><td>2</td><td>4,833</td><td>4,768</td></tr>
	<tr><td>1</td><td>3</td><td>18,121</td><td>17,496</td></tr>
	<tr><td>1</td><td>4</td><td>4,946</td><td>4,642</td></tr>
	<tr><td>1</td><td>5</td><td>471</td><td>456</td></tr>
	<tr><td>1</td><td>6</td><td>338</td><td>324</td></tr>
	<tr><td>1</td><td>7</td><td>330</td><td>326</td></tr>
	<tr><td>1</td><td>8</td><td>230</td><td>226</td></tr>
	<tr><td>1</td><td>9</td><td>226</td><td>222</td></tr>
	<tr><td>1</td><td>10</td><td>231</td><td>226</td></tr>
	<tr><td>1</td><td>11</td><td>241</td><td>221</td></tr>
	<tr><td>1</td><td>12</td><td>297</td><td>281</td></tr>
	<tr><td>1</td><td>13</td><td>264</td><td>264</td></tr>
	<tr><td>1</td><td>14</td><td>236</td><td>230</td></tr>
	<tr><td>1</td><td>15</td><td>226</td><td>233</td></tr>
	<tr><td>1</td><td>16</td><td>217</td><td>211</td></tr>
	<tr><td>1</td><td>17</td><td>225</td><td>217</td></tr>
	<tr><td>1</td><td>18</td><td>297</td><td>285</td></tr>
	<tr><td>1</td><td>19</td><td>287</td><td>282</td></tr>
	<tr><td>1</td><td>20</td><td>257</td><td>266</td></tr>
	<tr><td>4</td><td>1</td><td>4,885</td><td>4,858</td></tr>
	<tr><td>4</td><td>2</td><td>297</td><td>290</td></tr>
	<tr><td>4</td><td>3</td><td>264</td><td>251</td></tr>
	<tr><td>4</td><td>4</td><td>270</td><td>243</td></tr>
	<tr><td>4</td><td>5</td><td>216</td><td>212</td></tr>
	<tr><td>4</td><td>6</td><td>236</td><td>217</td></tr>
	<tr><td>4</td><td>7</td><td>244</td><td>231</td></tr>
	<tr><td>4</td><td>8</td><td>265</td><td>237</td></tr>
	<tr><td>4</td><td>9</td><td>253</td><td>242</td></tr>
	<tr><td>4</td><td>10</td><td>259</td><td>255</td></tr>
	<tr><td>4</td><td>11</td><td>267</td><td>255</td></tr>
	<tr><td>4</td><td>12</td><td>309</td><td>286</td></tr>
	<tr><td>4</td><td>13</td><td>275</td><td>254</td></tr>
	<tr><td>4</td><td>14</td><td>236</td><td>226</td></tr>
	<tr><td>4</td><td>15</td><td>322</td><td>316</td></tr>
	<tr><td>4</td><td>16</td><td>243</td><td>243</td></tr>
	<tr><td>4</td><td>17</td><td>261</td><td>242</td></tr>
	<tr><td>4</td><td>18</td><td>256</td><td>240</td></tr>
	<tr><td>4</td><td>19</td><td>231</td><td>220</td></tr>
	<tr><td>4</td><td>20</td><td>264</td><td>254</td></tr>
	<tr><td>16</td><td>1</td><td>762</td><td>755</td></tr>
	<tr><td>16</td><td>2</td><td>779</td><td>742</td></tr>
	<tr><td>16</td><td>3</td><td>780</td><td>756</td></tr>
	<tr><td>16</td><td>4</td><td>833</td><td>841</td></tr>
	<tr><td>16</td><td>5</td><td>756</td><td>756</td></tr>
	<tr><td>16</td><td>6</td><td>1,047</td><td>1,051</td></tr>
	<tr><td>16</td><td>7</td><td>840</td><td>821</td></tr>
	<tr><td>16</td><td>8</td><td>968</td><td>971</td></tr>
	<tr><td>16</td><td>9</td><td>848</td><td>859</td></tr>
	<tr><td>16</td><td>10</td><td>856</td><td>831</td></tr>
	<tr><td>16</td><td>11</td><td>832</td><td>810</td></tr>
	<tr><td>16</td><td>12</td><td>854</td><td>841</td></tr>
	<tr><td>16</td><td>13</td><td>854</td><td>833</td></tr>
	<tr><td>16</td><td>14</td><td>827</td><td>831</td></tr>
	<tr><td>16</td><td>15</td><td>725</td><td>718</td></tr>
	<tr><td>16</td><td>16</td><td>939</td><td>934</td></tr>
	<tr><td>16</td><td>17</td><td>789</td><td>797</td></tr>
	<tr><td>16</td><td>18</td><td>910</td><td>904</td></tr>
	<tr><td>16</td><td>19</td><td>844</td><td>863</td></tr>
	<tr><td>16</td><td>20</td><td>908</td><td>944</td></tr>
	<tr><td>64</td><td>1</td><td>2,130</td><td>2,069</td></tr>
	<tr><td>64</td><td>2</td><td>2,309</td><td>2,225</td></tr>
	<tr><td>64</td><td>3</td><td>2,392</td><td>2,319</td></tr>
	<tr><td>64</td><td>4</td><td>2,399</td><td>2,364</td></tr>
	<tr><td>64</td><td>5</td><td>2,472</td><td>2,457</td></tr>
	<tr><td>64</td><td>6</td><td>2,418</td><td>2,272</td></tr>
	<tr><td>64</td><td>7</td><td>2,060</td><td>1,983</td></tr>
	<tr><td>64</td><td>8</td><td>2,256</td><td>2,219</td></tr>
	<tr><td>64</td><td>9</td><td>2,403</td><td>2,422</td></tr>
	<tr><td>64</td><td>10</td><td>2,379</td><td>2,340</td></tr>
	<tr><td>64</td><td>11</td><td>2,371</td><td>2,371</td></tr>
	<tr><td>64</td><td>12</td><td>2,469</td><td>2,481</td></tr>
	<tr><td>64</td><td>13</td><td>2,362</td><td>2,308</td></tr>
	<tr><td>64</td><td>14</td><td>2,489</td><td>2,440</td></tr>
	<tr><td>64</td><td>15</td><td>2,434</td><td>2,539</td></tr>
	<tr><td>64</td><td>16</td><td>2,275</td><td>2,288</td></tr>
	<tr><td>64</td><td>17</td><td>2,548</td><td>2,536</td></tr>
	<tr><td>64</td><td>18</td><td>2,574</td><td>2,581</td></tr>
	<tr><td>64</td><td>19</td><td>2,405</td><td>2,400</td></tr>
	<tr><td>64</td><td>20</td><td>2,474</td><td>2,447</td></tr>
	<tr><td>128</td><td>1</td><td>2,810</td><td>2,718</td></tr>
	<tr><td>128</td><td>2</td><td>2,830</td><td>2,772</td></tr>
	<tr><td>128</td><td>3</td><td>2,864</td><td>2,909</td></tr>
	<tr><td>128</td><td>4</td><td>3,225</td><td>3,198</td></tr>
	<tr><td>128</td><td>5</td><td>3,043</td><td>3,010</td></tr>
	<tr><td>128</td><td>6</td><td>2,755</td><td>2,671</td></tr>
	<tr><td>128</td><td>7</td><td>2,549</td><td>2,539</td></tr>
	<tr><td>128</td><td>8</td><td>3,044</td><td>3,116</td></tr>
	<tr><td>128</td><td>9</td><td>2,951</td><td>2,919</td></tr>
	<tr><td>128</td><td>10</td><td>3,352</td><td>3,420</td></tr>
	<tr><td>128</td><td>11</td><td>3,372</td><td>3,357</td></tr>
	<tr><td>128</td><td>12</td><td>3,260</td><td>3,239</td></tr>
	<tr><td>128</td><td>13</td><td>3,165</td><td>3,208</td></tr>
	<tr><td>128</td><td>14</td><td>3,315</td><td>3,265</td></tr>
	<tr><td>128</td><td>15</td><td>3,075</td><td>3,243</td></tr>
	<tr><td>128</td><td>16</td><td>3,544</td><td>3,567</td></tr>
	<tr><td>128</td><td>17</td><td>3,237</td><td>3,203</td></tr>
	<tr><td>128</td><td>18</td><td>3,542</td><td>3,461</td></tr>
	<tr><td>128</td><td>19</td><td>3,454</td><td>3,480</td></tr>
	<tr><td>128</td><td>20</td><td>3,146</td><td>3,105</td></tr>
</table>
## Forward cost per crossing
<table header-row="true" fit-page-width="true">
	<tr><td>N</td><td>q</td><td>networks</td><td>parameters per leg</td><td>parameters per crossing</td><td>µs per track (batch 1452)</td><td>median error \[µm\]</td></tr>
	<tr><td>1</td><td>8</td><td>1</td><td>21,924</td><td>21,924</td><td>13.1</td><td>226</td></tr>
	<tr><td>1</td><td>16</td><td>1</td><td>26,052</td><td>26,052</td><td>12.8</td><td>211</td></tr>
	<tr><td>4</td><td>5</td><td>4</td><td>20,376</td><td>81,504</td><td>47.6</td><td>212</td></tr>
	<tr><td>4</td><td>8</td><td>4</td><td>21,924</td><td>87,696</td><td>47.6</td><td>237</td></tr>
	<tr><td>16</td><td>8</td><td>16</td><td>21,924</td><td>350,784</td><td>188</td><td>971</td></tr>
	<tr><td>16</td><td>15</td><td>16</td><td>25,536</td><td>408,576</td><td>199</td><td>718</td></tr>
	<tr><td>64</td><td>7</td><td>64</td><td>21,408</td><td>1,370,112</td><td>753</td><td>1,983</td></tr>
	<tr><td>64</td><td>8</td><td>64</td><td>21,924</td><td>1,403,136</td><td>754</td><td>2,219</td></tr>
	<tr><td>128</td><td>7</td><td>128</td><td>21,408</td><td>2,740,224</td><td>1,520</td><td>2,539</td></tr>
	<tr><td>128</td><td>8</td><td>128</td><td>21,924</td><td>2,806,272</td><td>1,503</td><td>3,116</td></tr>
</table>
## The supervised twin, test split
<table header-row="true" fit-page-width="true">
	<tr><td>comparator</td><td>pos median \[µm\]</td><td>pos 95th pct \[µm\]</td><td>slope median \[mrad\]</td><td>x \[µm\]</td><td>y \[µm\]</td><td>tx \[mrad\]</td><td>ty \[mrad\]</td><td>x bias \[µm\]</td><td>tx bias \[mrad\]</td></tr>
	<tr><td>against the fine truth</td><td>145</td><td>2,328</td><td>0.0658</td><td>106</td><td>80.9</td><td>0.0414</td><td>0.0343</td><td>+5.99</td><td>+0.00828</td></tr>
	<tr><td>against the real fibre-tracker state</td><td>1,761</td><td>13,454</td><td>0.528</td><td>1,294</td><td>765</td><td>0.363</td><td>0.239</td><td>-33.0</td><td>-4.48e-04</td></tr>
</table>
Parameters 17,796, restarts 17, converged True, wall 113 s.
## Continuity with Block A: the single step
<table header-row="true" fit-page-width="true">
	<tr><td>source</td><td>q</td><td>label-free network, median \[µm\]</td><td>supervised twin \[µm\]</td><td>exact scheme \[µm\]</td><td>straight line \[µm\]</td></tr>
	<tr><td>Block A1 stage sweep, 4x50, magnet-DOWN map, median of 3 seeds</td><td>2</td><td>5,202</td><td>174</td><td>5,198</td><td>520,444</td></tr>
	<tr><td>Block A1 stage sweep, 4x50, magnet-DOWN map, median of 3 seeds</td><td>4</td><td>5,371</td><td>181</td><td>5,401</td><td>520,444</td></tr>
	<tr><td>Block A1 stage sweep, 4x50, magnet-DOWN map, median of 3 seeds</td><td>8</td><td>181</td><td>176</td><td>22.5</td><td>520,444</td></tr>
	<tr><td>Block A1 stage sweep, 4x50, magnet-DOWN map, median of 3 seeds</td><td>16</td><td>187</td><td>170</td><td>19.5</td><td>520,444</td></tr>
	<tr><td>Block A2 size and seed, 4x50, q = 8, magnet-DOWN map, median of 10 seeds</td><td>8</td><td>222</td><td></td><td></td><td></td></tr>
	<tr><td>Block A2 size and seed, 4x200, q = 8, magnet-DOWN map, median of 10 seeds</td><td>8</td><td>115</td><td></td><td></td><td></td></tr>
	<tr><td>Block D, 2x128, magnet-UP map, one seed</td><td>1</td><td>171,524</td><td></td><td></td><td>450,541</td></tr>
	<tr><td>Block D, 2x128, magnet-UP map, one seed</td><td>2</td><td>4,768</td><td></td><td></td><td>450,541</td></tr>
	<tr><td>Block D, 2x128, magnet-UP map, one seed</td><td>3</td><td>17,496</td><td></td><td></td><td>450,541</td></tr>
	<tr><td>Block D, 2x128, magnet-UP map, one seed</td><td>4</td><td>4,642</td><td></td><td></td><td>450,541</td></tr>
	<tr><td>Block D, 2x128, magnet-UP map, one seed</td><td>5</td><td>456</td><td></td><td></td><td>450,541</td></tr>
	<tr><td>Block D, 2x128, magnet-UP map, one seed</td><td>6</td><td>324</td><td></td><td></td><td>450,541</td></tr>
	<tr><td>Block D, 2x128, magnet-UP map, one seed</td><td>7</td><td>326</td><td></td><td></td><td>450,541</td></tr>
	<tr><td>Block D, 2x128, magnet-UP map, one seed</td><td>8</td><td>226</td><td></td><td></td><td>450,541</td></tr>
	<tr><td>Block D, 2x128, magnet-UP map, one seed</td><td>9</td><td>222</td><td></td><td></td><td>450,541</td></tr>
	<tr><td>Block D, 2x128, magnet-UP map, one seed</td><td>10</td><td>226</td><td></td><td></td><td>450,541</td></tr>
	<tr><td>Block D, 2x128, magnet-UP map, one seed</td><td>11</td><td>221</td><td></td><td></td><td>450,541</td></tr>
	<tr><td>Block D, 2x128, magnet-UP map, one seed</td><td>12</td><td>281</td><td></td><td></td><td>450,541</td></tr>
	<tr><td>Block D, 2x128, magnet-UP map, one seed</td><td>13</td><td>264</td><td></td><td></td><td>450,541</td></tr>
	<tr><td>Block D, 2x128, magnet-UP map, one seed</td><td>14</td><td>230</td><td></td><td></td><td>450,541</td></tr>
	<tr><td>Block D, 2x128, magnet-UP map, one seed</td><td>15</td><td>233</td><td></td><td></td><td>450,541</td></tr>
	<tr><td>Block D, 2x128, magnet-UP map, one seed</td><td>16</td><td>211</td><td></td><td></td><td>450,541</td></tr>
	<tr><td>Block D, 2x128, magnet-UP map, one seed</td><td>17</td><td>217</td><td></td><td></td><td>450,541</td></tr>
	<tr><td>Block D, 2x128, magnet-UP map, one seed</td><td>18</td><td>285</td><td></td><td></td><td>450,541</td></tr>
	<tr><td>Block D, 2x128, magnet-UP map, one seed</td><td>19</td><td>282</td><td></td><td></td><td>450,541</td></tr>
	<tr><td>Block D, 2x128, magnet-UP map, one seed</td><td>20</td><td>266</td><td></td><td></td><td>450,541</td></tr>
</table>