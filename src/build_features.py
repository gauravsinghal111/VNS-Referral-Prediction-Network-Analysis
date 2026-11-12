"""
Step 3-4: Build the patient-sharing graph and one feature row per neurologist per year.

Unit:    a Texas neurologist (role = 'referrer') who appears in that year's Medicare
         patient-sharing data (at least one tie with any Texas provider).
Label:   1 if, in the NEXT year's file, the neurologist has a tie TO an epileptologist,
         neurosurgeon or hospital epilepsy unit (role = 'target'), else 0.
Features: computed ONLY from the current year's file, so nothing from the future leaks in.

Output: data/processed/features.parquet  (years 2013, 2014, 2015; 2015 has no label yet)
"""
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
YEARS = [2013, 2014, 2015]

# ZIP3 prefixes for Texas's large metro areas (simple urban/rural proxy).
METRO_ZIP3 = {
    "Houston": ["770", "772", "773", "774", "775"],
    "Dallas-Fort Worth": ["750", "751", "752", "753", "760", "761", "762"],
    "San Antonio": ["780", "781", "782"],
    "Austin": ["786", "787"],
    "El Paso": ["798", "799"],
}
ZIP3_TO_METRO = {z: m for m, zs in METRO_ZIP3.items() for z in zs}


def load_providers() -> pd.DataFrame:
    p = pd.read_parquet(DATA / "providers_tx.parquet")
    p["zip3"] = p["zip5"].str[:3]
    p["metro"] = p["zip3"].map(ZIP3_TO_METRO).fillna("Non-metro")
    return p


def neuro_graph(edges: pd.DataFrame, roles: pd.Series) -> nx.DiGraph:
    """Directed graph of ties among neurologists and targets only (a few thousand nodes)."""
    keep = edges["from_npi"].map(roles).isin(["referrer", "target"]) & \
           edges["to_npi"].map(roles).isin(["referrer", "target"])
    return nx.from_pandas_edgelist(edges[keep], "from_npi", "to_npi",
                                   edge_attr="patients", create_using=nx.DiGraph())


def features_for_year(year: int, providers: pd.DataFrame) -> pd.DataFrame:
    edges = pd.read_parquet(DATA / f"edges_tx_{year}.parquet")
    roles = providers.set_index("npi")["role"]
    targets = set(providers.loc[providers["role"] == "target", "npi"])

    # --- Volume features: all Texas ties, any specialty --------------------------
    out_v = edges.groupby("from_npi").agg(patients_sent=("patients", "sum"))
    in_v = edges.groupby("to_npi").agg(patients_received=("patients", "sum"))
    partners = pd.concat([edges[["from_npi", "to_npi"]],
                          edges[["to_npi", "from_npi"]].set_axis(["from_npi", "to_npi"], axis=1)])
    n_partners = partners.drop_duplicates().groupby("from_npi").size().rename("n_partners")
    vol = out_v.join(in_v, how="outer").join(n_partners, how="outer").fillna(0)
    vol["total_patients_shared"] = vol["patients_sent"] + vol["patients_received"]

    # --- Existing ties to targets this year (does the doctor already refer?) ----
    to_tgt = edges[edges["to_npi"].isin(targets)]
    prior = to_tgt.groupby("from_npi").agg(n_target_partners=("to_npi", "nunique"),
                                           patients_to_targets=("patients", "sum"))

    # --- Graph features on the neurology network --------------------------------
    G = neuro_graph(edges, roles)
    k = min(300, G.number_of_nodes())  # sampled betweenness: standard shortcut at this size
    graph = pd.DataFrame({
        "in_degree": dict(G.in_degree()),
        "out_degree": dict(G.out_degree()),
        "pagerank": nx.pagerank(G, weight="patients"),
        "betweenness": nx.betweenness_centrality(G, k=k, seed=42),
    })
    print(f"{year}: neuro graph {G.number_of_nodes():,} nodes, {G.number_of_edges():,} ties")

    # --- Assemble one row per active neurologist --------------------------------
    df = providers[(providers["role"] == "referrer") & providers["npi"].isin(vol.index)].copy()
    df = df.set_index("npi").join(vol).join(prior).join(graph)
    df[["n_target_partners", "patients_to_targets"]] = \
        df[["n_target_partners", "patients_to_targets"]].fillna(0)
    df[graph.columns] = df[graph.columns].fillna(0)

    # Geography: how many targets practise in the same ZIP3 (local access to an implant pathway)
    tgt_zip3 = providers.loc[providers["role"] == "target", "zip3"].value_counts()
    df["targets_in_zip3"] = df["zip3"].map(tgt_zip3).fillna(0)
    df["is_metro"] = (df["metro"] != "Non-metro").astype(int)

    # Provider
    df["years_since_npi"] = (year - df["enumeration_year"]).clip(lower=0)
    df["is_child_neurology"] = (df["specialty"] == "Child Neurology").astype(int)
    df["is_general_neurology"] = (df["specialty"] == "Neurology").astype(int)
    df["year"] = year

    # --- Label: tie to a target in NEXT year's file -----------------------------
    nxt = DATA / f"edges_tx_{year + 1}.parquet"
    if nxt.exists():
        e2 = pd.read_parquet(nxt, columns=["from_npi", "to_npi"])
        future_referrers = set(e2.loc[e2["to_npi"].isin(targets), "from_npi"])
        df["label"] = df.index.isin(future_referrers).astype(int)
    else:
        df["label"] = np.nan
    return df.reset_index()


def main():
    providers = load_providers()
    feats = pd.concat([features_for_year(y, providers) for y in YEARS], ignore_index=True)
    feats.to_parquet(DATA / "features.parquet", index=False)
    summary = feats.groupby("year").agg(neurologists=("npi", "size"), positive_rate=("label", "mean"),
                                        already_refer=("n_target_partners", lambda s: (s > 0).mean()))
    print(summary.round(3))


if __name__ == "__main__":
    main()
