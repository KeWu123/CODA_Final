# SliceEqOcc-OAAC scale1.50 analysis

Scale1.50 does not replace scale1.25. Its unchanged validation best is lower
by `0.000679`, directly triggering the locked stop rule. The `+0.000099` test
Dice difference is numerically negligible, is accompanied by worse Jaccard,
HD95 and ASD, and is driven by only 2/10 case wins with a negative median.

The appearance magnitude rises from `0.068954` at scale1.25 to `0.082723` at
scale1.50, confirming that the experiment genuinely moved outward in severity.
The absence of selector-compatible improvement therefore brackets the local
response: scale1.25 is the selected OAAC setting among scale1.0/1.25/1.50.

Do not add scale1.125, 1.375, 1.75, per-component bounds, application
probability rescue, or test-checkpoint search. Those searches would optimize
noise in the five-case validation and repeatedly queried ten-case development
split rather than strengthen the paper. The 1.50 result belongs in a parameter
sensitivity table; the final method uses scale1.25.
