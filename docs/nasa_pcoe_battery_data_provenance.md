# NASA PCoE Battery Data: Provenance and Use Boundary

Audit date: 2026-08-03

Status: `auxiliary_benchmark_only`; the four local CSV files are not included in
this repository.

## Source record

The upstream dataset is the NASA Ames Prognostics Center of Excellence (PCoE)
**Li-ion Battery Aging Datasets** collection, catalog identifier
`DASHLINK_133`.

- Official Data.gov catalog:
  <https://catalog.data.gov/dataset/li-ion-battery-aging-datasets>
- NASA Open Data Portal record:
  <https://data.nasa.gov/dataset/li-ion-battery-aging-datasets>
- Legacy DASHlink landing page recorded by the catalog:
  <https://c3.nasa.gov/dashlink/resources/133/>
- Related original prognostics paper: Saha and Goebel, *Uncertainty Management
  for Diagnostics and Prognostics of Batteries using Bayesian Techniques*:
  <https://ntrs.nasa.gov/citations/20130010482>

The official catalog describes commercially available, 2 Ah, 18650-size
lithium-ion cells exercised under charge, discharge, and electrochemical
impedance spectroscopy profiles. It reports an approximately 10 Hz acquisition
rate and an end-of-life criterion of 30% rated-capacity fade, from 2 Ah to
1.4 Ah. Some discharge thresholds were deliberately below the OEM-recommended
2.7 V threshold to accelerate deep-discharge aging.

The README distributed with the FY08Q4 files is also preserved in a third-party
mirror:
<https://labinfo.ing.he-arc.ch/gitlab/ticc/16TICc19/nasa-battery-dataset/-/blame/bac7f5812a70d05d0a79e3bca578efc80fb5b59c/data/BatteryAgingARC-FY08Q4/README.txt>.
That README reports room-temperature operation, 1.5 A constant-current charge
to 4.2 V followed by constant-voltage charge to 20 mA, and 2 A constant-current
discharge. Reported discharge cutoffs are 2.7 V, 2.5 V, 2.2 V, and 2.5 V for
B0005, B0006, B0007, and B0018, respectively. These cell-specific values are
attributed to the accompanying README, not to a new LifeTwin measurement.

## Local CSV acquisition

The inspected local bundle contains only four files named `B0005.csv`,
`B0006.csv`, `B0007.csv`, and `B0018.csv`. It contains no README, source URL,
conversion script, conversion version, license file, or upstream file hashes.

The CSV header is `type,temp,time,data`, and each `data` value contains a
serialized mapping of measurement arrays. The documented upstream representation
uses MATLAB files such as `B0005.mat` and a nested `cycle` structure. Therefore,
these CSV files must be described as an **unverified third-party conversion of
the NASA dataset**, not as NASA-authored CSV files. Matching names and plausible
content do not establish byte-level or semantic equivalence to the upstream MAT
files.

The following hashes identify only the exact local files inspected on the audit
date. They are not official NASA checksums.

| File | Bytes | SHA-256 |
|---|---:|---|
| `B0005.csv` | 49,218,466 | `d74b6352fde77fcb55543df48180914ca92d56d36320d70e0ebfcd57696b6105` |
| `B0006.csv` | 49,410,002 | `d544bdcfdf053861cc96736dd25b91a7de99fa91d1a6877aba3e05bb6a5d97c9` |
| `B0007.csv` | 49,943,430 | `251b6a074702fc07991db86c1760db843db967d6648f01f4270337d94461fd80` |
| `B0018.csv` | 26,358,323 | `9ce1516d47b3cb2a4a03d9a6c671fdbc6e703468795cbe5ee772605989ac011f` |

The upstream MAT files are now locally present, but their presence does not
authorize scoring or establish semantic equivalence to these four CSV files.
Before any scored experiment, the project still requires dataset-specific
rights resolution, protocol review, a frozen deterministic conversion, a
completed CSV/MAT crosswalk, and new explicit execution authorization. Until
then, comparing operation counts, timestamps, capacities, cutoff behavior, or
sampled arrays and performing formal scoring remain prohibited.

## 2026-08-06 extracted MAT metadata intake

This intake concerns a different evidence object from the four third-party CSV
files above. The extracted snapshot contains six top-level directories, 38 MAT
files and 10 README/TXT files. Filename identity plus byte-count/SHA-256 duplicate
rules yield 34 unique `Bxxxx` physical-battery IDs and 4 identical duplicate
representation groups: B0025, B0026, B0027, and B0028. The count of 34 is an
identity count for this snapshot, not a claim of 34 independent, same-
distribution, or qualified test cells.

The intake read only file metadata, streaming hashes, 128-byte MATLAB headers,
README/TXT text, and top-level variable names, MATLAB types, and shapes through
`scipy.io.whosmat`. It did not load MAT arrays or capacity values and did not
prepare prefixes, predict, train, or score; all such access and execution counts
are zero. SNL content access was also zero.

The README files expose stopping thresholds, experimental anomalies, and parts
of the outcome/protocol structure. Exposure is therefore fixed as
`development_only_outcomes_and_protocol_structure_exposed`, not outcome-blind or
independent confirmation. No semantic equivalence between the four CSV files and
the 38 MAT files has been established.

Possession and public access do not resolve dataset-specific licensing,
chemistry, redistribution, or aggregate-result publication rights. The chemistry
remains unspecified lithium-ion rather than LFP evidence, and the formal NASA
execution gate remains closed. Rights resolution, protocol review, conversion
freeze, and a new explicit authorization are required before any formal scoring.

## Chemistry and label caveats

The official catalog identifies the cells only as commercial lithium-ion 18650
cells. It does not authoritatively specify the cathode or anode chemistry.
Secondary publications disagree about the chemistry, so LifeTwin does not label
this cohort as LFP, LCO, NCA, or any other specific chemistry without primary
manufacturer or NASA documentation.

The four cells also use different discharge cutoff voltages. Their reported
capacity values are consequently protocol-dependent and must not be pooled as
if they shared an identical measurement boundary. Any later benchmark must keep
cell identity and cutoff protocol explicit, normalize within cell where
appropriate, and use cell-held-out rather than random row splits.

## License status

The official catalog marks the dataset access level as `public` and explicitly
describes prognostic-algorithm development as an intended use, but it does not
provide a dataset-specific license identifier or license URL. The current NASA
portal record likewise does not resolve a license for this collection.

NASA's general science-data guidance distinguishes NASA-led mission data from
other data and asks users to validate source rights when NASA may not be the
original rights holder:
<https://science.data.nasa.gov/about/license>. NASA's STI terms also explain that
U.S. Government employee works are generally not protected by U.S. copyright,
while contractor or third-party material publicly released by NASA may remain
copyrighted:
<https://sti.nasa.gov/disclaimers/>.

Accordingly, LifeTwin does not infer CC0, CC BY, commercial-use rights, or
redistribution rights for this dataset or for the third-party CSV conversion.
The project may inspect the local files for private research, parser testing,
and data-quality assessment, while preserving attribution. It must not commit or
redistribute the CSV files, present them as official NASA CSVs, or imply that
public access alone resolved all downstream rights. Publication or commercial
reuse requires a documented rights review; this statement is a project policy,
not legal advice.

The official catalog lists Dawn McIntosh (`dawn.m.mcintosh@nasa.gov`) as the
dataset contact for clarification.

## Permitted evidence role

After provenance cross-checking, this cohort may serve only as an auxiliary
cycle-aging benchmark for:

- parser and schema validation;
- causal early-prefix SOH or RUL evaluation;
- leave-one-cell-out transfer checks;
- handling measurement noise, local capacity recovery, and non-monotonicity;
- testing uncertainty expansion and refusal behavior under domain shift.

It cannot support claims of LFP-specific accuracy, Hithium product performance,
stationary-storage deployment readiness, calendar-aging validity, or 15-25 year
forecast accuracy. The dataset contains four small cylindrical cells, short
accelerated cycle-aging histories, fixed laboratory profiles, and no multi-year
stationary-storage calendar trajectory. It lacks the temperature, SOC-window,
low-C-rate, rest, maintenance, pack-heterogeneity, and operational context needed
for those claims.

The approved public wording is:

> NASA PCoE data are used only as a cross-cell cycle-aging stress benchmark.
> They do not validate LifeTwin's LFP or 15-25 year stationary-storage claims.

## Local reproduction

Place the four files under an ignored local directory such as
`data/raw/nasa_pcoe/`. Do not copy them into `data/external/` or commit them.
After installing the project dependencies, run:

```powershell
$env:PYTHONPATH='src'
python scripts/run_nasa_pcoe_benchmark.py prepare data/raw/nasa_pcoe

python scripts/run_nasa_pcoe_benchmark.py predict `
  artifacts/nasa-prefix-loco-v1/cycles.csv

python scripts/run_nasa_pcoe_benchmark.py score `
  artifacts/nasa-prefix-loco-v1/cycles.csv `
  artifacts/nasa-prefix-loco-v1/predictions.csv `
  artifacts/nasa-prefix-loco-v1/prediction_manifest.json
```

The ingest rejects files whose byte sizes or SHA-256 identities differ from the
four audited conversions. The frozen benchmark uses target prefixes at cycles
20, 40, 60, and 100 and scores only through the common cycle-132 support. Its
prediction function accepts a physically truncated prefix table; suffix outcomes
are linked only by the separate scorer.

For the post-V1 dynamic-gate development experiment, run:

```powershell
$env:PYTHONPATH='src'
python scripts/run_nasa_dynamic_gate_v2.py run-source data/raw/nasa_pcoe `
  --output-directory artifacts/nasa-dynamic-gate-v2
```

The V2 adapter extracts only within-cycle discharge-curve measurements: current
integration, energy integration, voltage-threshold times, a 3.8-3.4 V duration,
voltages at 0.5 and 1.0 Ah, their mean dV/dQ, and temperature rise. All 636
discharge records in the pinned bundle cover both common windows. These features
do not repair the chemistry, conversion-provenance, licensing, or domain-shift
limitations above. Results and negative findings are recorded in
`reports/nasa_dynamic_gate_v2_development_2026-08-03.md`.
