Runs: off (off = ARKN) vs on_rule48 (on = network on the UT->T crossing of the second pass)
fitted long tracks (rows): off 810892, on 810892; same events and order: True
fits with a NaN q/p or chi2: off 18, on 18, both 18 (left out below)
events with tracks: 15466

| run | tracks | real | ghosts | chi2/ndf real median / mean | chi2/ndf ghost median / mean | dp/p bias | dp/p sigma (core) |
|---|---|---|---|---|---|---|---|
| off | 810874 | 754917 | 55957 | 1.2901 / 2.2209 | 7.8076 / 15.2290 | -4.82e-04 | 0.0056 |
| on | 810874 | 754917 | 55957 | 1.2903 / 2.2209 | 7.7992 / 15.2486 | -4.82e-04 | 0.0056 |

Momentum resolution of real tracks by true p (core sigma of dp/p, %; bias = median dp/p, %):
| p (GeV) | tracks | off sigma | on sigma | off bias | on bias |
|---|---|---|---|---|---|
| 2-3 | 46719 | 0.496 | 0.496 | -0.0130 | -0.0131 |
| 3-5 | 158540 | 0.467 | 0.467 | -0.0316 | -0.0317 |
| 5-8 | 168437 | 0.444 | 0.444 | -0.0473 | -0.0473 |
| 8-12 | 130269 | 0.439 | 0.439 | -0.0508 | -0.0508 |
| 12-20 | 123613 | 0.483 | 0.483 | -0.0558 | -0.0557 |
| 20-50 | 104754 | 0.940 | 0.940 | -0.1146 | -0.1143 |
| 50-200 | 20180 | 2.847 | 2.847 | -1.2796 | -1.2792 |

Track by track (on - off):
  tracks with any change in q/p, chi2 or ndof: 373045 of 810874 (46.01 %); q/p changed: 6079; ndof changed: 5111
  relative q/p change: 99 % 0.00e+00, largest 6.41e-02
  chi2/ndf change, real: mean -0.0000, median +0.0000, 1 % -0.009, 99 % +0.009; up 181587, down 177361
  chi2/ndf change, ghost: mean +0.0196, median +0.0000, 1 % -0.046, 99 % +0.039; up 7240, down 6857
  tracks heading to |y| > 800 mm at z = 5 m: 22053 (2.72 %), 1.426 per event (events with >= 1: 66.6 %)
    tail    : changed 41.34 %, chi2/ndf change mean -0.0006, 0.1 % -5.574, 99.9 % +6.091, largest |change| 33.00; ndof changed 340 (real tracks 21256)
    not tail: changed 46.14 %, chi2/ndf change mean +0.0014, 0.1 % -0.441, 99.9 % +0.431, largest |change| 1562.11; ndof changed 4771 (real tracks 733661)

RICH2 states against MC truth (TrackResChecker, long tracks): mean and RMS of the residuals / pulls
| state | quantity | off entries | off mean | on mean | off RMS | on RMS |
|---|---|---|---|---|---|---|
| RICH2Beg | txpull | 0 | nan | nan | nan | nan |
| RICH2Beg | txres | 572447 | -2.3e-05 | -2.298e-05 | 0.0009925 | 0.0009925 |
| RICH2Beg | typull | 0 | nan | nan | nan | nan |
| RICH2Beg | tyres | 664489 | 2.586e-06 | 2.469e-06 | 0.0009143 | 0.0009144 |
| RICH2Beg | xpull | 0 | nan | nan | nan | nan |
| RICH2Beg | xres | 53041 | -0.01597 | -0.01596 | 0.2198 | 0.2198 |
| RICH2Beg | ypull | 0 | nan | nan | nan | nan |
| RICH2Beg | yres | 64774 | -0.006455 | -0.006365 | 0.2275 | 0.2276 |
| RICH2End | txpull | 0 | nan | nan | nan | nan |
| RICH2End | txres | 570227 | -2.458e-05 | -2.466e-05 | 0.0009936 | 0.0009937 |
| RICH2End | typull | 0 | nan | nan | nan | nan |
| RICH2End | tyres | 664581 | 2.784e-06 | 2.662e-06 | 0.0009145 | 0.0009147 |
| RICH2End | xpull | 0 | nan | nan | nan | nan |
| RICH2End | xres | 36709 | -0.02483 | -0.02478 | 0.2309 | 0.231 |
| RICH2End | ypull | 0 | nan | nan | nan | nan |
| RICH2End | yres | 50769 | -0.004189 | -0.004008 | 0.2277 | 0.2277 |
