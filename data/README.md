# Data sources

All files are free, public and aggregated - no patient-level records. Downloaded on **2026-11-11**.
Save each file into `data/raw/` with the name shown, then run `python src/prepare_data.py`.

| File (save as) | What it is | Download |
|---|---|---|
| `physician-shared-patient-patterns-2013-days30.zip`<br>`...-2014-days30.zip`<br>`...-2015-days30.zip` | CMS Physician Shared Patient Patterns (the public DocGraph referral data). One row per pair of providers who treated the same Medicare patient within 30 days. Pairs with fewer than 11 shared patients are suppressed by CMS. | `https://downloads.cms.gov/foia/physician-shared-patient-patterns-<YEAR>-days30.zip` |
| `NPPES_Data_Dissemination_November_2026_V2.zip` | NPPES NPI Registry full monthly file: name, practice address, taxonomy (specialty) for every NPI. | https://download.cms.gov/nppes/NPI_Files.html |
| `OrderReferring_20261109.csv` | CMS Order and Referring: which NPIs may currently order/refer in Medicare. | https://data.cms.gov/provider-characteristics/medicare-provider-supplier-enrollment/order-and-referring |
| `nucc_taxonomy.csv` (reference only) | NUCC taxonomy code descriptions, used to choose specialty codes. | https://www.nucc.org/index.php/code-sets-mainmenu-41/provider-taxonomy-mainmenu-40/csv-mainmenu-57 |

## Raw shared-patient file format

No header. Columns: `npi_from, npi_to, pair_count, beneficiary_count, same_day_count`.
`npi_from` saw the beneficiary first and `npi_to` saw the same beneficiary within the 30-day window,
so the tie has a direction. This project uses `beneficiary_count` as the edge weight ("patients").

## Why 2013-2015?

These are the most recent years CMS published as a free public file. Three consecutive years let us
train on 2013 -> 2014 and test on 2014 -> 2015 - a real time-based hold-out.

## Specialty definitions (NUCC codes)

- **Referrers (the neurologists we score):** primary taxonomy is Neurology 2084N0400X, Child Neurology
  2084N0402X, Clinical Neurophysiology 2084N0600X, Vascular Neurology 2084V0102X, Neuromuscular
  2084N0008X, Behavioral Neurology 2084B0040X, Neurocritical Care 2084A2900X or Neurodevelopmental
  Disabilities 2084P0005X.
- **Targets (the VNS pathway):** any listed taxonomy of Epilepsy 2084E0001X, Neurological Surgery
  207T00000X or hospital Epilepsy Unit 273100000X.
