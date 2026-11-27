Runs: off (off = ARKN) vs on_nofb (on = network on the UT->T crossing of the second pass)
fitted long tracks (rows): off 810892, on 810892; same events and order: True
fits with a NaN q/p or chi2: off 18, on 18, both 18 (left out below)
events with tracks: 15466

| run | tracks | real | ghosts | chi2/ndf real median / mean | chi2/ndf ghost median / mean | dp/p bias | dp/p sigma (core) |
|---|---|---|---|---|---|---|---|
| off | 810874 | 754917 | 55957 | 1.2901 / 2.2209 | 7.8076 / 15.2290 | -4.82e-04 | 0.0056 |
| on | 810874 | 754917 | 55957 | 1.2903 / 2.2210 | 7.8017 / 15.2481 | -4.82e-04 | 0.0056 |

Momentum resolution of real tracks by true p (core sigma of dp/p, %; bias = median dp/p, %):
| p (GeV) | tracks | off sigma | on sigma | off bias | on bias |
|---|---|---|---|---|---|
| 2-3 | 46719 | 0.496 | 0.496 | -0.0130 | -0.0130 |
| 3-5 | 158540 | 0.467 | 0.467 | -0.0316 | -0.0317 |
| 5-8 | 168437 | 0.444 | 0.444 | -0.0473 | -0.0473 |
| 8-12 | 130269 | 0.439 | 0.439 | -0.0508 | -0.0508 |
| 12-20 | 123613 | 0.483 | 0.483 | -0.0558 | -0.0557 |
| 20-50 | 104754 | 0.940 | 0.940 | -0.1146 | -0.1142 |
| 50-200 | 20180 | 2.847 | 2.847 | -1.2796 | -1.2792 |

Track by track (on - off):
  tracks with any change in q/p, chi2 or ndof: 373038 of 810874 (46.00 %); q/p changed: 6237; ndof changed: 5233
  relative q/p change: 99 % 0.00e+00, largest 6.41e-02
  chi2/ndf change, real: mean +0.0001, median +0.0000, 1 % -0.010, 99 % +0.010; up 181690, down 177245
  chi2/ndf change, ghost: mean +0.0191, median +0.0000, 1 % -0.049, 99 % +0.041; up 7235, down 6868
  tracks heading to |y| > 800 mm at z = 5 m: 22053 (2.72 %), 1.426 per event (events with >= 1: 66.6 %)
    tail    : changed 41.67 %, chi2/ndf change mean -0.0016, 0.1 % -9.147, 99.9 % +8.217, largest |change| 105.86; ndof changed 468 (real tracks 21256)
    not tail: changed 46.13 %, chi2/ndf change mean +0.0015, 0.1 % -0.434, 99.9 % +0.433, largest |change| 1562.11; ndof changed 4765 (real tracks 733661)

RICH2 states against MC truth (TrackResChecker, long tracks): mean and RMS of the residuals / pulls
| state | quantity | off entries | off mean | on mean | off RMS | on RMS |
|---|---|---|---|---|---|---|
| RICH2Beg | txpull | 0 | nan | nan | nan | nan |
| RICH2Beg | txres | 572447 | -2.3e-05 | -2.299e-05 | 0.0009925 | 0.0009925 |
| RICH2Beg | typull | 0 | nan | nan | nan | nan |
| RICH2Beg | tyres | 664489 | 2.586e-06 | 2.421e-06 | 0.0009143 | 0.0009145 |
| RICH2Beg | xpull | 0 | nan | nan | nan | nan |
| RICH2Beg | xres | 53041 | -0.01597 | -0.01596 | 0.2198 | 0.2198 |
| RICH2Beg | ypull | 0 | nan | nan | nan | nan |
| RICH2Beg | yres | 64774 | -0.006455 | -0.006524 | 0.2275 | 0.2276 |
| RICH2End | txpull | 0 | nan | nan | nan | nan |
| RICH2End | txres | 570227 | -2.458e-05 | -2.466e-05 | 0.0009936 | 0.0009938 |
| RICH2End | typull | 0 | nan | nan | nan | nan |
| RICH2End | tyres | 664581 | 2.784e-06 | 2.644e-06 | 0.0009145 | 0.0009147 |
| RICH2End | xpull | 0 | nan | nan | nan | nan |
| RICH2End | xres | 36709 | -0.02483 | -0.02474 | 0.2309 | 0.231 |
| RICH2End | ypull | 0 | nan | nan | nan | nan |
| RICH2End | yres | 50769 | -0.004189 | -0.004067 | 0.2277 | 0.2276 |
