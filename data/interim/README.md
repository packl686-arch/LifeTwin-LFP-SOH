# Included Naumann Calendar-Aging Table

`naumann_calendar_observations.csv` is a normalized derivative of the public
Naumann LFP/graphite calendar-aging dataset:

- Dataset: *Data for: Analysis and modeling of calendar aging of a commercial
  LiFePO4/graphite cell*
- Contributor: Maik Naumann
- Dataset DOI: <https://doi.org/10.17632/kxh42bfgtj.1>
- Related paper DOI: <https://doi.org/10.1016/j.est.2018.01.019>
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Normalized file SHA-256:
  `73e7f3c155aed3da7ae637f6b3b91df3eb1fecc5d19f8702af8da810fd62f47c`

## Changes Made by LifeTwin

The upstream measurements were normalized into one row per published checkup
and temperature/SOC condition. LifeTwin added stable condition identifiers,
standard time units, capacity-retention and capacity-loss fields, source and
license metadata, and explicit statistical-unit fields. No individual-cell raw
traces are represented: each trajectory is the published mean of three physical
cells, and the independent analysis unit is the condition trajectory.

Redistribution and reuse of this CSV must preserve the attribution above and
indicate that it is a normalized derivative.

