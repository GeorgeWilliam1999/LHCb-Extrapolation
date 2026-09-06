# Speaker notes

Progress talk, 7 September 2026. One entry per slide, in order; slide numbers
match the page numbers in `main.pdf` (there are no overlays, so one frame is one
page). Written in my own voice, to be spoken rather than read out.

---

**1. Title**

I introduce this as a two-problem talk: an oscillator and the LHCb magnet, one
method between them. I say up front that the surrogate replaces field
propagation only, so nobody spends the next hour wondering where multiple
scattering went. I mention that there are two written companions, both finished,
so the audience knows the numbers are traceable and they can push on anything.

**2. Where this talk goes**

I walk the eight blocks quickly and flag the shape of the argument: the method
question and the product question came out differently, and the talk is built to
make that separation visible. I say that the tables are all in the main line
rather than in backup, because I would rather be interrupted on a number than
hand-wave past it. I say the last block is where I want the discussion.

---

## The problem

**3. What the extrapolator does**

I define the five-component track state and say where the extrapolator is called
from: the fit, the front-to-back matching, and the vertex fit. The number I want
them to hold onto is the $520\,\mu$m median departure from a straight line on
the leg I study, because everything later is a correction to that. I close on
the scope line and the G2 gate, so the boundary of what I am modelling is a
measured number rather than a claim.

**4. Why a fixed cost**

I make the argument that the interesting property is not speed but
predictability: forty-four root-finder residual evaluations whose count depends
on the track, against one forward pass whose cost is identical for every track.
Then I immediately undercut it with the box at the bottom, because I would
rather say the weakness myself than have it found. If someone wants to argue
about the operation count, this is the right slide to be interrupted on.

**5. The two questions**

I emphasise that both meanings were written down before the results, so the
answers could not drift to meet them. I read the three clauses under each
question out loud, because the rest of the talk is scored against exactly those
clauses. I say plainly that question one came out yes and question two did not,
so nobody is waiting for a reveal.

---

## Theory

**6. The two constructions**

I stress that the source paper contains two things that get called by the same
name, and that the difference is structural rather than stylistic. The box at
the bottom is the whole reason I opened the discrete-time column first: the
known start enters the loss as data through every equation, so there is no
interval in which a hand-over can hide. I note their stated limitation, one
initial state and one step, because generalising past it is what my later
experiments do.

**7. The equation of motion**

I write the system out because the audience will want to see where $\kappa$ and
the momentum convention sit. The three properties are the ones that matter
later: non-autonomous, momentum conserved so only four components are predicted,
and a field that is a table rather than a function. I do the ten-GeV
back-of-envelope out loud, because getting four hundred millimetres against a
measured five hundred and twenty is the cheapest reassurance that nothing is
upside down.

**8. Collocation**

I say that implicit here means the stage equations are genuinely coupled, and I
point at the negative entry above the diagonal in the two-stage tableau as the
proof that they cannot be evaluated one at a time. The collocation picture is
the one I keep using for the rest of the talk: one smooth curve, checked at $q$
places. I mention that the closed forms are only used to check the general
construction, which is what actually builds every tableau I run.

**9. The loss**

This is the slide the whole method rests on, so I take it slowly. Each of the
$q+1$ outputs is walked backwards through the scheme and has to land on the
input, which is known; the discrepancies are the entire training signal. I say
explicitly that no correct answer enters anywhere, and that the field must be
differentiable because the gradient runs through the lookup.

**10. The scales**

I explain why the normalisation exists at all: without it the transverse
position would own the loss and the slopes would be invisible. Then I make the
point that the scales are measured from the training states themselves, so
nothing about a reference trajectory sneaks into a label-free loss. The box on
the right is a promissory note: one scale for the whole population is fine on
one geometry and is the thing that breaks later.

**11. The residual parameterisation**

I present this as changing exactly one thing, what the network outputs, and
nothing else in the pipeline. The bending integral is the part worth dwelling
on, because it is a first-order estimate built only from inputs and that is what
keeps the method label free. I make the point that the estimate does not have to
be right, only right to within a factor of a few on every leg type, and that the
gate for choosing the node profile was fixed before I looked at the histograms.

**12. The continuous-time objective**

I write the objective out so the asymmetry is visible: the physics is a mean
over $N$ points and the boundary information is a single number. Then I put the
Jacobian at the origin next to it and let the audience see the problem before I
state it. The origin is repelling and yet it solves the equation exactly, so a
constant output there is a perfect minimiser of the residual term.

**13. The splice family**

I build the splice construction slowly, because the surprise is that the
impostors are exact solutions rather than approximate ones, joined where the
loss cannot look. The $1/T$ arithmetic is worth doing out loud: the price of a
splice falls, the number of places to put one grows, and the price of the truth
rises, all with the same $T$. That is why every resource axis I measured came
back null, and I say so here rather than later.

**14. Pseudo-time stepping**

I explain the relaxed objective as an implicit Euler step in an artificial time,
and the amplification factor $\tau^2/h^2$ as the reason it prices the hidden
layer. I note that I follow the authors' reference code where it disagrees with
their text on the late shrink of $\tau$, because that is the version that
produced their published numbers. The predictions on the right were written
before the runs, and I want them on the record before I show results.

**15. Comparators and metrics**

I say that a number in micrometres means nothing on its own, which is why every
result has three fixed comparators beside it. The ceiling is the important one:
it is the best any network trained on those equations could do, so it is the
honest scorecard. I close on the rule that a single seed is not a measurement,
because that discipline is what makes several later comparisons survive.

---

## The fix literature

**16. Causal weighting**

I describe the mechanism, then give the table: it moves the cliff from one
period to two and no further. The part I care about is not the accuracy gain but
the visibility gain, because these failures declare themselves in the telemetry
while the plain ones report as converged. The budget-extension test is the
clincher: four times the epochs, seven runs, none of them completed the
crossing.

**17. Rohrhofer and the fixed point**

I present this as the diagnosis that got the field moving: fixed points are
global minima with basins that grow with the horizon, and no architecture
repairs it. I confirm it in my own runs, with the classification measured per
run rather than eyeballed. Then I say what it does not cover: the origin is not
the only zero-residual candidate, which is the next two slides.

**18. The regulariser**

I write the term out and explain why it is inert on the true solution by
construction, which is a nice property. My result is that it works exactly as
far as they tested it, two periods, and past that it converts the failure rather
than removing it. That is the sentence I most want the audience to take away:
closing the origin makes the optimiser take the next cheapest member of the
family.

**19. Pseudo-time stepping, measured**

I say this is the only arm that survives past two periods, and I give the
success counts and the error spreads rather than a single headline. I also give
its own wall, between six and ten periods, because a fix with an unstated limit
is not useful to anyone. The $1/T$ confirmation over sixty-five parked runs is
what turns the theory into a measurement.

---

## System and data

**20. Why van der Pol**

I give the three reasons in order: known truth, cheap enough to measure axes to
exhaustion, and isolating so any failure is the method's own. I point out that
the list of things settled there is not decorative, since the stage count, the
optimisation-floor diagnosis, the selection protocol and the chaining decision
all came from the oscillator. If someone thinks the oscillator is a detour, this
is where I would rather argue about it than at the end.

**21. The shared machinery**

I make the point that the reference integrator is verified rather than trusted
on both problems, and that the twin exists so every label-free claim has a
matched supervised number next to it. Then I walk the stall-and-confirm
protocol, which is the least glamorous slide in the talk and the one that saved
the most work. The July lesson is why it exists: a fixed budget gave a
factor-two result that was not real.

**22. The dictionary**

I read down the table and let the audience map the two problems onto one
another. The two rows I linger on are non-autonomous and the measured field,
because those are the genuinely new difficulties, and the last row, because the
difficulty axis changes from one quantity to two. Everything else is a change of
dimension and units.

**23. Van der Pol data**

I give the four verification instruments and their measured values, since the
whole study is scored against this reference. The point is that it is
trustworthy to about $10^{-13}$, six orders below anything a network reaches, so
the yardstick never contaminates the measurement. I mention that nothing is
interpolated: where a network needs a time between stored samples, the
integrator is re-run to stop exactly there.

**24. The local simulation and reconstruction chain**

I say this came first, before any network, and that it is proven end to end: the production checkers print the numbers a real reconstruction prints. I name the one gotcha worth remembering, the extended digitisation, because it cost a day. Then I say why I abandoned home-grown events for the official sample anyway: not because the chain was wrong, but because provenance has to be beyond argument, and a silent configuration error would poison everything downstream.

**25. LHCb data**

I describe the harvest in one pass and then dwell on two choices: splitting by
particle rather than by leg, because legs from one particle are correlated and
splitting by leg would leak, and building leg D but never training on it so
there is a genuine out-of-training test. The population figure is the check that
switching to an official sample changed the provenance and not the physics. The
policy change of 21 July is worth stating plainly: home-grown generation caused
silent problems.

**26. Gates and the fiducial requirement**

I give the four gates quickly and stop on G2, because that is the material
effect and therefore the scope boundary of the project. Then the fiducial
requirement, which is the story I would most want another group to hear: one
percent of the states carried half the physics loss, and the asymmetry between
the two losses is structural rather than accidental. The physics loss evaluates
the field where the network proposes, so it cannot absorb an out-of-map
trajectory the way a supervised loss silently does.

---

## Results, van der Pol

**27. The cliff**

I give the table and the two pictures together, because the phase plane is what
makes the failure legible: the fit abandons the loop and spirals into the
origin, against the direction of the true flow. The number I stress is that the
seeds agree to two decimal places, so this is systematic and not an
initialisation accident. Below one period the reproduction is at or better than
the source work's own reported accuracy, which is the check that the machinery
is built correctly.

**28. The classical anchor at the same step**

I put this in so nobody thinks the networks are being compared only with each other. At a step of 0.8 the explicit order-6 method blows up, the implicit one of the same order is fine at every horizon, and inside the networks' working range the network beats the implicit method. The point I want to land is the last box: fine steps favour classical integration by miles; at a coarse step only implicit structure works, whether solved or learned, and that is the discrete-time column's whole premise.

**29. Causal weighting**

I show the improvement at two periods and then spend most of the time on the one
run that produced a training loss of $9.7\times10^{-6}$ with an error of $0.71$.
That run is the splice theorem in the wild, found before I knew what it was: an
out-of-phase near-solution joined to the truth inside an unsampled interval. I
say that finding it is what sent me to the stability literature.

**30. The resource axes under the original protocol**

This is the 700 runs that earned the right to say the failure is structural. I walk the three axes quickly: size does nothing and depth hurts, density does nothing at the failing horizons, placement does nothing. I dwell on the success map, because the start deciding the outcome is what pointed at the objective. And I state the qualification myself before anyone asks: under the converged protocol density does help at two periods, and I show that later; the four-period wall is what the null results are about.

**31. The supervised control**

I present this as the cleanest possible experiment: change the loss and nothing
else. Twenty-four supervised runs all fit four periods; every physics run at the
same horizon sits between $0.4$ and $2.0$. So the network can express the answer
and the optimiser can find it, and the fault is in what the residual objective
accepts.

**32. Eight objectives**

I explain that a fixed-budget first pass would have given a different and wrong
answer, which is why every run here is trained to a demonstrated plateau. Every
fix works at two periods, which is exactly where the regulariser paper tested
it, so this is not a criticism of that paper. Past two periods only pseudo-time
stepping survives, and I let the column of zeros do the work.

**33. The three failure classes**

I use the phase plane to show that the failures are not noise but three
reproducible shapes. The fact I most want stated is that the polish drives the
flow-following objectives below $10^{-4}$ without moving the error at all, so
loss and error decouple on that branch. That is the silent failure in its fully
converged form, and it is why I do not trust a converged loss on this
construction.

**34. The eight arms in pictures**

I let the residual panel do the talking. Every regulariser variant is quiet across the whole window and pays one enormous spike right after the anchor: that is a splice, placed where the fixed set cannot see it. The parks look different, a moderate residual across the hand-over. I say that the classification on the previous slide is read off this panel, not asserted, and that the summary and trajectory panels on the left are the same 240 runs seen two other ways.

**35. The two resource axes**

I make the honest correction first: an earlier null verdict on density does not
survive training to convergence at two periods, so I have changed that
statement. Past four periods it stands, and pseudo-time stepping's six-period
rate does not move in any direction with density. I say clearly that ten seeds
cannot separate those rates, so I am not claiming a trend I cannot resolve.

**36. The two axes as pictures**

The three panels are the same two axes drawn out, and the right-hand one is the
part I want people to look at. The long-horizon failures are late parks: the
network rides the loop for four to eight periods, drifts off, and then sits at
the origin, and nothing ever wanders off the loop. I flag the reporting
consequence, that the classifier reads a late park as diffuse, so the
periods-survived panel is the honest readout at long horizon rather than the
success rate.

**37. The $1/T$ price**

This is the slide where the theory becomes a measurement: over sixty-five parked
runs, loss times $T$ has a median of $0.85$ and a range of $0.84$ to $0.90$ from
four to fifteen periods. The consequence is the useful part: density cannot move
a resampled method's wall by pricing, because the parked branch is already fully
sampled. What density can do is make the descent to the truth cheaper in more
seeds, which is exactly what the four-period column shows.

**38. The discrete-time floor**

I contrast this immediately with everything before it: the loss is an honest
proxy for the error here, a squared-residual loss of $10^{-5}$ resolving outputs
to roughly its square root. At two stages the network sits on the exactly solved
scheme's own error, and from four stages the scheme runs away to roundoff while
the network stays flat. That flatness is what fixes eight stages for everything
afterwards.

**39. The same floor twice**

I put the stage curve and the architecture map side by side because they are two
views of one statement. More stages cannot help once the scheme's error is below
the network's, and more parameters cannot help either, and in both cases the
reason is how far the optimiser got. I say here that the LHCb architecture grid
then reproduced this independently on a completely different system, which is
the transfer claim in its cleanest form.

**40. Size and seed on van der Pol**

I give the correlation of $+0.974$ as the evidence that the floor is
optimisation and not capacity, and note that there is no capacity-rich,
badly-generalising regime anywhere in the grid. Then the twist: that same loss
ranks six-period chained error at only $+0.35$. So one-step quality and
long-horizon quality are different properties, and the seed decides the second
one.

**41. Selection**

I give the protocol as three steps and the payoff as a factor of three to
ninety-six over the median seed. The far-horizon check is what separated the
front-runners, and I show that the chosen network stayed flat to thirty periods
while its siblings grew by factors of nine to twenty-four. The caveat is real
and I state it: the validation window has to match the horizon the map will
actually be used at.

**42. Routes and the data twin**

I show that chaining a reusable one-step map beats one giant step by a factor of
ten, and that the reason the giant step is limited is nonlinear solvability
rather than linear stability. Then the matched pair: perfect labels are a stable
factor of $2.1$ better and cost seven hundred and ninety-nine integrator steps
per training state, while the physics loss costs zero. I make the point that at
this accuracy the gap is not an information gap, since the two minimisers differ
five orders below either error.

---

## Results, LHCb

**43. The discrete-time column in pictures**

Three pictures for the three claims of the previous table: the head-to-head shows where each formulation wins, the test-set panel shows the error as a distribution rather than one lucky trajectory, and the label bill shows what the physics loss buys. I say plainly that the growth in the middle panel belongs to unselected seeds, and that the selected network's flat line was on the slide before.

**44. Steps 0 and 1 on LHCb**

Before any network I show the function to be learned and the ceiling under it. The field is smooth and uniform across real paths, which is why a one-step method can hope to work; the bending correction falls as one over momentum, which is why the soft tracks are the hard ones. The exact scheme converged on every one of 640 legs, including a single step across the whole magnet, and it floors at about 23 micrometres because of the field map's interpolation, not the scheme. Every network number that follows is measured against this curve on the same states.

**45. The July baseline**

I give the three things that were fixed in July and have not moved since, and
why that leg was chosen: it is the modal cross-magnet plane pair and the hardest
routine step. The point of the figure is the overlap of the two loss families,
which is what the fiducial requirement bought. I say that the factor-two gap the
July experiments chased was a budget artefact, because that is a mistake worth
owning in public.

**46. The stage sweep**

This is the result I am proudest of, so I slow down. At two and four stages the
label-free network lands within $0.1$ and $0.6$ percent of the scheme's own
error, which means it is solving the equations it was given and those equations
are five millimetres wrong on this leg. The twin is thirty times better there
precisely because it never sees the scheme. I also give the two surprises, since
four stages being no better than two is genuinely counterintuitive.

**47. The same sweep as a picture**

I point at where the blue and green curves lie on top of one another and where
they separate, because that is the whole argument in one image. Then I make the
correction about the $29\,\mu$m figure I quoted in July, since a
momentum-stratified population is a harder population and the number is not
comparable. Both curves are in the figure so the difference is visible rather
than argued about.

**48. The architecture grid**

I put the whole grid up because I would rather be asked about a specific cell
than show a summary. Depth four is the optimum and it is interior, which is a
useful thing to know for anyone building one of these. I note that the
unconverged runs are all at width thirty-two or depth six and none of them
diverged: they failed the confirmation pass, which is a different and milder
statement.

**49. The loss-error band**

This is the evidence slide for the optimisation-floor claim, and the picture
does the work: three decades of loss onto one and a half decades of error, all
on one band. If the floor were capacity the small architectures would form their
own branches above it, and they do not. The footnote about the two systems
agreeing to two decimal places is genuinely unexplained and I say so.

**50. Returns and the twin**

I give the diminishing-returns arithmetic, then the result I did not expect: at
the largest size the label-free loss beats its own supervised twin, with a
narrower seed spread. I make the seed discipline explicit, including that
validation selection is safe here and stops being safe the moment the network is
chained. The recommendation at the bottom is what the later experiments should
have followed and only partly did, because they were already running.

**51. The reversed polarity**

I frame this as the strongest available test of the label-free claim: train
where no labelled sample has ever existed. I give the three pre-submission
checks, because the obvious failure mode is a polarity flag that is silently
ignored, and the probe run rules that out. The one genuine asymmetry is in the
tracks rather than the map, and it affects tails and not medians, so I state
which statements need a polarity attached.

**52. The reversed polarity in pictures**

I use the right-hand panel, the two sets of trajectories curving in opposite
directions, as the check that the network learned the sign of the bending rather
than the geometry. The left panel shows the two polarities lying on top of one
another with both ceilings marked. The middle panel is the momentum dependence,
which is the same curve for both.

**53. One network for all legs**

I say the honest thing first: it works mechanically and is nowhere near any of
the ceilings. Making the leg an input costs a factor of fourteen on the magnet
crossing, and on the two short leg types the network is worse than doing
nothing, which is the failure that mattered. The twin being uniformly ahead
here, where on the frozen leg the two were indistinguishable, is the clue that
the gap opens exactly where the geometry varies.

**54. The hypothesis**

I present the four pieces of evidence in the order I actually had them, and I
make the point that this was stated as a hypothesis before it was tested. The
one-part-in-$10^{8}$ arithmetic is the crux: resolving a seven-micrometre
correction on an output whose scale is six hundred millimetres. Then the second
wave separated the parameterisation explanation from the capacity one, and it
came out on the parameterisation side.

**55. Chaining the absolute network**

I give the table and then the diagnosis: growth ratios of about two per leg, in
every architecture and both losses, which is systematic bias rather than
independent random error. Width shifts the curve down and does not change its
slope, which is the sense in which chained error is not bounded. The seed lesson
is the same one the oscillator taught, and it transferred exactly.

**56. Leg D**

I use this as the out-of-training test, since no leg of that geometry was in any
training set. Under absolute outputs, splitting a hard step into two easier ones
makes it worse, and the reason is visible in the first hop being longer than
anything in training. Under the residual design the gap almost closes, so the
earlier conclusion that composite stepping actively destroys accuracy is now
only marginally true, and I would rather retract that cleanly than leave it
standing.

**57. The absolute-output network in pictures**

Three panels, one per result I have just tabulated, for anyone who reads a cloud
faster than a column. On the by-leg panel the thing to see is that on the two
short rows the straight line lies below every network point. On the chain panel
the curves are straight on a log axis over the first four legs, which is
geometric growth rather than square-root growth, and on the leg-D panel the
composite route is worse than the single step at every momentum.

**58. The residual redesign**

I stress that exactly one thing changed and everything else is byte for byte
identical, because that is what makes the comparison mean something. The two
short legs transform by factors of ninety-two and a hundred and ninety and cross
from above the straight line to below it. The magnet crossing improves by only
two and a half, and that is the whole of the remaining problem.

**59. Reading it across**

I go through the five readings and make sure the fourth and fifth land: the twin
is extraordinary on the short legs, and the distance-to-ceiling scorecard says
the network is nowhere near the scheme it is solving on any leg. The
initialisation-check box is the part I would want another group to copy, because
an untrained residual network already beats a straight line by construction. And
the optimisation got easier, not just the answer better, which is the strongest
form of the conditioning claim.

**60. Inside the step, and the tails**

The stage-error table is the clearest single piece of evidence that the
diagnosis was right: flat means an offset, rising means accumulation, and the
profiles change exactly as predicted. Then I turn to the tails and say the
unwelcome thing: a small population got worse while the bulk got thousands of
times better. An extrapolator with a bounded time budget has to survive its
tail, so this is a real limitation and not a presentational one.

**61. The redesign in pictures**

The left panel is the crossing I care about: the residual cloud moves from above
the straight line to below it on the two short leg types, uniformly in momentum.
The right panel is the profile change, flat becoming rising, which is an offset
turning into an honest accumulation. I mention the one cell that is still worse
than a straight line, the stiff seventy-millimetre hop above twenty GeV, because
there is essentially nothing to predict there and the network's own noise floor
sits above it.

**62. Chaining the residual networks**

Two things are true at once here and they pull in opposite directions, so I say
both. The chain is better everywhere, by four to thirteen times, and the whole
distribution moves rather than just the median. But the compounding got worse,
and the reason is that after two or three steps the chain is dominated by the
one leg the redesign barely improved.

**63. The cross-magnet leg on one axis**

I use this as the summary of the product story: making the leg an input costs a
factor of fourteen and the redesign recovers two and a half of it. The
right-hand figure is the gate that had to pass before any farm time was spent,
and I point out that the same label-free expression lands within ten percent of
the truth on three geometries spanning five orders of magnitude. That gap, and
not the output scale, is what the next experiments attack.

**64. The verdict**

I put everything on one page and read the two boxes. Question one is yes on
every one of its pre-stated clauses. Question two is closer than it was and not
there, and I finish on the cost statement with its flag attached, because I do
not want the operation count travelling further than it should.

---

## What is not working

**65. Two ceilings and a floor**

I separate the three limits cleanly: the field map caps the scheme, the
optimiser caps the network a factor of five above that, and one leg caps the
product. The twin also being stuck at five hundred and twelve micrometres on
that leg is the key fact, because it means the problem is not the label-free
objective. And the thirteen-percent share is the uncomfortable irony: fixing the
conditioning handed the optimisation budget to the easy legs.

**66. What I cannot claim**

I go through these deliberately, including that labels are free at LHCb so the
label-free advantage is methodological rather than economic. The harness lessons
are on the right because they cost me a day each and I would like them to cost
somebody else nothing. I say explicitly that no result changes because of the
restart-cap incident, and why.

**67. What would falsify the edge**

I present two tests with pass and fail criteria written before running them, so
the outcome cannot be reinterpreted afterwards. For the throughput test the
important design choice is matching accuracy by tuning the production
integrator's step, so the comparison is not made at different accuracies on each
side. For the cross-magnet test the two outcomes point at different work, which
is exactly why it is worth running first.

---

## Direction

**68. The queue**

I put the field surrogate first because it lifts the ceiling for everyone and
not only for me, and flag that it changes what is being solved, which is a
question for the room. The second-order scale is the tail fix and it stays label
free, which matters for the portability argument. The two small items are
housekeeping and I mention them only so they are on the record.

**69. What the theory predicts, and the thesis framing**

I record the predictions before opening the continuous-time column rather than
after, because that is the only way they are worth anything. There is no
isolated fixed point to park on here, so the expected failure is a
flow-following splice, and the intervention to reach for is pseudo-time stepping
rather than the regulariser. Then the two boxes: the method is validated and the
product edge is unproven, and I say which of those the thesis can carry today.

**70. Feedback**

I read the six questions and say which one I most want answered, which is the
third: I do not have an accuracy or tail requirement to score against, and
having one would change which experiments are worth running. I would also like
the portability argument attacked before it goes into a thesis. Then I stop
talking.

---

## Reference slides

**71. Protocol and gates**

Only if asked. The seven-step protocol and the three package gates, all passing,
including the bitwise reproduction of the July baseline dataset and first
optimiser restart.

**72. Provenance**

Only if asked. Commits, the sample, the field maps and the cluster identifiers,
with the two commits that are still local and pending push flagged as such.

**73. Where the record lives**

Only if asked, or at the end if people want to read further: the two papers, the
five write-ups and the repository.
