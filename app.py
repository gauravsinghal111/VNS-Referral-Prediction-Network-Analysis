"""
Streamlit dashboard for the commercial team.
Run:  streamlit run app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
TEAL, ORANGE, NAVY, GREY = "#0f7b74", "#c9821f", "#12324a", "#8a99a6"

st.set_page_config(page_title="VNS Referral Targets - Texas", layout="wide")


@st.cache_data
def load():
    ranked = pd.read_csv(ROOT / "report" / "target_list_texas.csv", dtype={"npi": str, "zip5": str})
    providers = pd.read_parquet(ROOT / "data" / "processed" / "providers_tx.parquet",
                                columns=["npi", "name", "specialty", "role", "city"])
    edges = pd.read_parquet(ROOT / "data" / "processed" / "edges_tx_2015.parquet")
    edges = edges[edges.from_npi.isin(ranked.npi) | edges.to_npi.isin(ranked.npi)]
    return ranked, providers.set_index("npi"), edges


ranked, providers, edges = load()

st.title("Which Texas neurologists are most likely to refer toward VNS Therapy?")
st.caption(
    "Score = predicted probability that a neurologist forms a Medicare patient-sharing tie with an "
    "epileptologist, neurosurgeon or hospital epilepsy unit in the next year. Patient-sharing is a "
    "**proxy** for referral, not a recorded referral. Built from public, aggregated CMS data "
    "(2013-2015); portfolio project, not a commercial deployment."
)

# ---------------------------------------------------------------- filters
with st.sidebar:
    st.header("Filters")
    metros = st.multiselect("Metro area", sorted(ranked.metro.unique()))
    specs = st.multiselect("Specialty", sorted(ranked.specialty.unique()))
    only_new = st.checkbox("Only new opportunities (no current tie to a target)")
    only_eligible = st.checkbox("Only currently Medicare-eligible to refer (2026)", value=True)
    top_n = st.slider("Show top N", 10, 300, 50, step=10)

view = ranked
if metros:
    view = view[view.metro.isin(metros)]
if specs:
    view = view[view.specialty.isin(specs)]
if only_new:
    view = view[~view.already_refers]
if only_eligible:
    view = view[view.medicare_eligible_2026]

c1, c2, c3 = st.columns(3)
c1.metric("Neurologists in view", f"{len(view):,}")
c2.metric("Already share patients with a target", f"{view.already_refers.mean():.0%}" if len(view) else "-")
c3.metric("Average score (top N)", f"{view.head(top_n).score.mean():.0%}" if len(view) else "-")

st.subheader("Ranked target list")
st.dataframe(
    view.head(top_n)[["rank", "name", "specialty", "city", "metro", "score", "already_refers", "top_drivers"]],
    hide_index=True, width="stretch",
    column_config={"score": st.column_config.ProgressColumn("Score", format="%.2f", min_value=0, max_value=1),
                   "already_refers": st.column_config.CheckboxColumn("Already refers"),
                   "top_drivers": st.column_config.TextColumn("Why this score (top 3 drivers)", width="large")},
)
st.download_button("Download this list (CSV)", view.head(top_n).to_csv(index=False), "vns_targets.csv")

# ---------------------------------------------------------------- physician lookup
st.subheader("Look up a physician")
query = st.text_input("Search by name or NPI", placeholder="e.g. SMITH or 1234567890")
if query:
    hits = ranked[ranked.name.str.contains(query, case=False, na=False) | ranked.npi.str.startswith(query)]
    if hits.empty:
        st.info("No neurologist in the scored list matches that search.")
    else:
        pick = st.selectbox("Select", hits.npi, format_func=lambda n: f"{hits.set_index('npi').at[n, 'name']} "
                            f"({hits.set_index('npi').at[n, 'city']}) - NPI {n}")
        doc = ranked.set_index("npi").loc[pick]
        a, b = st.columns([1, 1.4])
        with a:
            st.markdown(f"### {doc['name']}")
            st.write(f"{doc.specialty} - {doc.city.title()}, TX {doc.zip5}")
            st.metric("Referral score", f"{doc.score:.0%}", f"rank {int(doc['rank'])} of {len(ranked):,}",
                      delta_color="off")
            st.write("**Already shares patients with a target:**", "Yes" if doc.already_refers else "No")
            st.write("**Top drivers of the score:**")
            for d in doc.top_drivers.split("; "):
                st.write(f"- {d}")
        with b:
            e = edges[(edges.from_npi == pick) | (edges.to_npi == pick)].nlargest(25, "patients")
            if e.empty:
                st.info("No patient-sharing ties with Texas providers in 2015.")
            else:
                G = nx.from_pandas_edgelist(e, "from_npi", "to_npi", "patients", create_using=nx.DiGraph())
                role = lambda n: providers.at[n, "role"] if n in providers.index else "other"
                color = [NAVY if n == pick else TEAL if role(n) == "target" else
                         ORANGE if role(n) == "referrer" else GREY for n in G]
                fig, ax = plt.subplots(figsize=(6, 4.5))
                pos = nx.spring_layout(G, seed=1)
                nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color=GREY, arrowsize=8,
                                       width=[0.5 + G[u][v]["patients"] / e.patients.max() * 3 for u, v in G.edges])
                nx.draw_networkx_nodes(G, pos, ax=ax, node_color=color, node_size=[300 if n == pick else 90 for n in G])
                for c, lab in [(NAVY, "This physician"), (TEAL, "Epilepsy / neurosurgery target"),
                               (ORANGE, "Other neurologist"), (GREY, "Other specialty")]:
                    ax.scatter([], [], c=c, label=lab)
                ax.legend(frameon=False, fontsize=8, loc="lower left"); ax.axis("off")
                ax.set_title("Top 25 patient-sharing ties (2015)", fontsize=10)
                st.pyplot(fig)
                partner = e.assign(partner=e.from_npi.where(e.from_npi != pick, e.to_npi),
                                   direction=(e.from_npi == pick).map({True: "sent ->", False: "<- received"}))
                partner["partner_name"] = partner.partner.map(providers["name"])
                partner["partner_specialty"] = partner.partner.map(providers["specialty"])
                st.dataframe(partner[["direction", "partner_name", "partner_specialty", "patients"]],
                             hide_index=True, width="stretch")

with st.expander("How good is the model? (time-based hold-out)"):
    st.markdown((ROOT / "report" / "metrics.md").read_text(encoding="utf-8"))
    st.image(str(ROOT / "figures" / "precision_at_k.png"))
