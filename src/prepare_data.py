"""
Step 1-2: Clean and join the three public files (on NPI) for Texas.

Inputs (data/raw/, see data/README.md for download links):
  - CMS Physician Shared Patient Patterns, 30-day window, 2013/2014/2015  -> the EDGES
  - NPPES NPI Registry (monthly full file)                               -> the NODES
  - CMS Order and Referring                                              -> eligibility flag

Outputs (data/processed/):
  - providers_tx.parquet      one row per Texas NPI: name, city, ZIP, specialty, role
  - edges_tx_<year>.parquet   patient-sharing ties where BOTH providers practise in Texas

The heavy lifting is plain SQL (DuckDB) because the raw files are 1-11 GB each
and do not fit in pandas on a laptop.
"""
import zipfile
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

STATE = "TX"
YEARS = [2013, 2014, 2015]

# --- Specialty definitions (NUCC taxonomy codes) -------------------------------
# Referrers = physicians whose PRIMARY specialty is neurology or a neurology subspecialty.
NEURO_REFERRER_CODES = {
    "2084N0400X": "Neurology",
    "2084N0402X": "Child Neurology",
    "2084N0600X": "Clinical Neurophysiology",
    "2084V0102X": "Vascular Neurology",
    "2084N0008X": "Neuromuscular Medicine",
    "2084B0040X": "Behavioral Neurology",
    "2084A2900X": "Neurocritical Care",
    "2084P0005X": "Neurodevelopmental Disabilities",
}
# Targets = where a VNS candidate is sent: epileptologists, neurosurgeons, hospital epilepsy units.
# Matched on ANY listed taxonomy, because many epileptologists list "Neurology" as primary.
TARGET_CODES = {
    "2084E0001X": "Epilepsy",
    "207T00000X": "Neurological Surgery",
    "273100000X": "Epilepsy Unit (hospital)",
}


def unzip_once(zip_path: Path, prefix: str, suffix: str) -> Path:
    """Extract the one big file we need from a zip (skipped if already extracted)."""
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist()
                    if Path(n).name.startswith(prefix) and n.lower().endswith(suffix)
                    and "fileheader" not in n.lower())
        target = RAW / Path(name).name
        if not target.exists():
            print(f"  extracting {name} ...")
            z.extract(name, RAW)
            if (RAW / name) != target:
                (RAW / name).rename(target)
        return target


def sql_list(codes) -> str:
    return ", ".join(f"'{c}'" for c in codes)


def main():
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")

    # ------------------------------------------------------------------ NODES
    nppes_zip = next(RAW.glob("NPPES_Data_Dissemination_*.zip"))
    nppes_csv = unzip_once(nppes_zip, "npidata_pfile", ".csv")  # ~11 GB
    print(f"NPPES file: {nppes_csv.name}")

    # Primary taxonomy = the slot flagged 'Y'; fall back to slot 1.
    primary = "CASE " + " ".join(
        f"""WHEN "Healthcare Provider Primary Taxonomy Switch_{i}" = 'Y'
                THEN "Healthcare Provider Taxonomy Code_{i}" """ for i in range(1, 16)
    ) + ' ELSE "Healthcare Provider Taxonomy Code_1" END'
    all_codes = "concat_ws('|', " + ", ".join(
        f'"Healthcare Provider Taxonomy Code_{i}"' for i in range(1, 16)) + ")"

    con.execute(f"""
        CREATE TABLE nppes_tx AS
        SELECT
            CAST("NPI" AS VARCHAR)                                   AS npi,
            "Entity Type Code"                                       AS entity_type,
            CASE WHEN "Entity Type Code" = '2'
                 THEN "Provider Organization Name (Legal Business Name)"
                 ELSE trim(concat_ws(' ', "Provider First Name", "Provider Last Name (Legal Name)"))
                      || coalesce(', ' || nullif("Provider Credential Text", ''), '')
            END                                                      AS name,
            upper(trim("Provider Business Practice Location Address City Name")) AS city,
            left("Provider Business Practice Location Address Postal Code", 5)   AS zip5,
            try_strptime("Provider Enumeration Date", '%m/%d/%Y')    AS enumeration_date,
            {primary}                                                AS primary_taxonomy,
            {all_codes}                                              AS all_taxonomies
        FROM read_csv('{nppes_csv.as_posix()}', all_varchar=true, header=true)
        WHERE "Provider Business Practice Location Address State Name" = '{STATE}'
    """)
    # (Deactivated NPIs have blank address fields in NPPES, so the state filter drops them.)
    print("Texas NPPES rows:", con.sql("SELECT count(*) FROM nppes_tx").fetchone()[0])

    # Order & Referring: is this NPI currently allowed to order/refer for Medicare Part B?
    oar_csv = next(RAW.glob("OrderReferring_*.csv"))
    con.execute(f"""
        CREATE TABLE oar AS
        SELECT DISTINCT CAST(NPI AS VARCHAR) AS npi, PARTB = 'Y' AS medicare_eligible_2026
        FROM read_csv('{oar_csv.as_posix()}', all_varchar=true, header=true)
    """)

    ref_case = "CASE " + " ".join(
        f"WHEN primary_taxonomy = '{c}' THEN '{n}'" for c, n in NEURO_REFERRER_CODES.items()
    ) + " END"
    tgt_case = "CASE " + " ".join(
        f"WHEN all_taxonomies LIKE '%{c}%' THEN '{n}'" for c, n in TARGET_CODES.items()
    ) + " END"

    con.execute(f"""
        CREATE TABLE providers AS
        SELECT
            p.npi, p.entity_type, p.name, p.city, p.zip5,
            year(p.enumeration_date)                               AS enumeration_year,
            p.primary_taxonomy,
            CASE WHEN {tgt_case} IS NOT NULL THEN 'target'
                 WHEN {ref_case} IS NOT NULL THEN 'referrer'
                 ELSE 'other' END                                  AS role,
            coalesce({tgt_case}, {ref_case}, 'Other')              AS specialty,
            coalesce(o.medicare_eligible_2026, false)              AS medicare_eligible_2026
        FROM nppes_tx p
        LEFT JOIN oar o USING (npi)
    """)
    n_before = con.sql("SELECT count(*) FROM nppes_tx").fetchone()[0]
    n_after = con.sql("SELECT count(*) FROM providers").fetchone()[0]
    assert n_before == n_after, "Order & Referring join duplicated NPIs"
    con.sql("COPY providers TO '{}' (FORMAT parquet)".format((OUT / "providers_tx.parquet").as_posix()))
    print(con.sql("SELECT role, specialty, count(*) n FROM providers "
                  "WHERE role <> 'other' GROUP BY ALL ORDER BY role, n DESC").df().to_string(index=False))

    # ------------------------------------------------------------------ EDGES
    # Raw file has no header: npi_from, npi_to, pair_count, bene_count, same_day_count.
    # npi_from saw the patient first; npi_to saw the same patient within 30 days.
    for year in YEARS:
        txt = unzip_once(RAW / f"physician-shared-patient-patterns-{year}-days30.zip", "physician", ".txt")
        con.execute(f"""
            CREATE OR REPLACE TABLE raw_edges AS
            SELECT trim(column0) AS from_npi, trim(column1) AS to_npi,
                   CAST(trim(column2) AS INTEGER) AS pair_count,
                   CAST(trim(column3) AS INTEGER) AS patients,
                   CAST(trim(column4) AS INTEGER) AS same_day_count
            FROM read_csv('{txt.as_posix()}', header=false, all_varchar=true,
                          columns={{'column0':'VARCHAR','column1':'VARCHAR','column2':'VARCHAR',
                                   'column3':'VARCHAR','column4':'VARCHAR'}})
        """)
        n_raw = con.sql("SELECT count(*) FROM raw_edges").fetchone()[0]
        con.execute(f"""
            COPY (
                SELECT e.*
                FROM raw_edges e
                JOIN providers a ON a.npi = e.from_npi
                JOIN providers b ON b.npi = e.to_npi
                WHERE e.from_npi <> e.to_npi
            ) TO '{(OUT / f"edges_tx_{year}.parquet").as_posix()}' (FORMAT parquet)
        """)
        n_tx = con.sql(f"SELECT count(*) FROM '{(OUT / f'edges_tx_{year}.parquet').as_posix()}'").fetchone()[0]
        print(f"{year}: {n_raw:,} national ties -> {n_tx:,} ties with both providers in {STATE}")


if __name__ == "__main__":
    main()
