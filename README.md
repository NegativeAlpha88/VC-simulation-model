# VC-simulation-model

vc portfolio sim: big checks vs. follow-ons
so i was talking with another vc analyst here in the spanish ecosystem about whether it's better to go big on the first check or just save your cash for the follow-ons. instead of just debating it, i had gemini spit out a model to see what the numbers say.

this is v1.0. it's quick, it's dirty, and it gives some interesting insights.

what the model says (and what it ignores)
the model suggests the 'small initial checks, big follow-ons' strategy comes out ahead, and the result is even "statistically significant."

but before anyone gets too excited, you should know this model is basically a sketch on a napkin and conveniently ignores a few real-world details:

allocation constraints: the model assumes you can invest whatever you want, whenever you want. it completely ignores that with increasing competition, especially for the best performers, you often don't choose your allocation—you take what you can get.

dilution: doesn't track ownership or dilution. a minor detail in venture capital, i'm sure.

multiples: uses a fixed valuation step-up for every round.

exits: pretends every exit is drawn from the same distribution, regardless of how the company actually performed.

true randomness & outliers: the model uses a nice, clean log-normal distribution for exits. real vc returns are far messier, more random, and driven by extreme, black-swan outliers that a simulation like this can't fully capture.

correlation: assumes every startup is an island. no such thing as market downturns affecting everyone.

timing & money: acts like the time value of money isn't a thing.

the "science" part
to make it look official, the simulation runs 10,000 times and uses a mann-whitney u test to compare the results. it's a non-parametric statistical test, meant to tell you if the two piles of numbers (no matter its distribution) are actually different by ranking the results of the simulations.

so... is it useful?
look, for a stylized model, it's fine. it shows a thing. it lets you compare two ideas with transparent rules and uses probabilities and stats to back it up.

i'm working on a v2.0 that's a bit more sophisticated and actually considers all the stuff listed above. stay tuned.
