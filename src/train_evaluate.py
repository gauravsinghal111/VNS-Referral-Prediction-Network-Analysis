"""
Step 5-6: Train, evaluate on a time-based hold-out, explain, and produce the target list.

Time split (no physician-year from the future is ever seen in training):
  TRAIN  features from 2013 -> label = tie to a target in 2014
  TEST   features from 2014 -> label = tie to a target in 2015   (all metrics come from here)
  SCORE  features from 2015 -> predicted ties in 2016 = the ranked target list

Outputs:
  report/metrics.md, report/error_analysis.md, report/target_list_texas.csv, figures/*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA, REPORT, FIGS = ROOT / "data" / "processed", ROOT / "report", ROOT / "figures"
REPORT.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

# Feature name -> plain-English label used in SHAP plots, the memo and the dashboard.
FEATURES = {
    "n_target_partners": "Already shares patients with epilepsy/neurosurgery",
    "patients_to_targets": "Patients sent to epilepsy/neurosurgery",
    "total_patients_shared": "Total patients shared (all doctors)",
    "patients_received": "Patients received from other doctors",
    "patients_sent": "Patients sent to other doctors",
    "n_partners": "Number of distinct partner doctors",
    "in_degree": "Neuro network: incoming ties",
    "out_degree": "Neuro network: outgoing ties",
    "pagerank": "Neuro network: PageRank (influence)",
    "betweenness": "Neuro network: betweenness (bridge role)",
    "targets_in_zip3": "Epilepsy/neurosurgery specialists nearby (ZIP3)",
    "is_metro": "Practises in a major metro",
    "years_since_npi": "Years since NPI issued",
    "is_child_neurology": "Child neurologist",
    "is_general_neurology": "General neurologist",
}
X_COLS = list(FEATURES)
NAVY, TEAL, ORANGE, GREY = "#12324a", "#0f7b74", "#c9821f", "#8a99a6"
K = 100  # the field team's visit capacity -> Precision at 100


def precision_at_k(y, score, k=K):
    top = np.argsort(-score)[:k]
    return float(np.asarray(y)[top].mean())


def recall_at_k(y, score, k=K):
    top = np.argsort(-score)[:k]
    return float(np.asarray(y)[top].sum() / max(np.sum(y), 1))


def metrics_row(name, y, score, prob=None):
    return {
        "Model": name,
        "Precision@100": precision_at_k(y, score),
        "Recall@100": recall_at_k(y, score),
        "ROC-AUC": roc_auc_score(y, score),
        "PR-AUC": average_precision_score(y, score),
        "Brier": brier_score_loss(y, prob) if prob is not None else np.nan,
    }


def make_logreg():
    # log1p tames the heavy-tailed counts; scaling puts features on one footing.
    return make_pipeline(FunctionTransformer(np.log1p), StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0))


def make_xgb():
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss",
                         random_state=42)


def to_md(df: pd.DataFrame, floatfmt=".3f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def main():
    df = pd.read_parquet(DATA / "features.parquet")
    df["years_since_npi"] = df["years_since_npi"].fillna(df["years_since_npi"].median())
    train, test, score = df[df.year == 2013], df[df.year == 2014], df[df.year == 2015].copy()
    Xtr, ytr, Xte, yte = train[X_COLS], train["label"], test[X_COLS], test["label"]
    print(f"train {len(train):,} rows ({ytr.mean():.1%} positive) | "
          f"test {len(test):,} rows ({yte.mean():.1%} positive)")

    # ------------------------------------------------------------ models
    logreg = make_logreg().fit(Xtr, ytr)
    xgb = make_xgb().fit(Xtr, ytr)
    p_lr, p_xgb = logreg.predict_proba(Xte)[:, 1], xgb.predict_proba(Xte)[:, 1]

    rows = [
        metrics_row("Baseline: rank by patient volume", yte, test["total_patients_shared"].values),
        metrics_row("Baseline: rank by current ties to targets", yte,
                    test["patients_to_targets"].values + 1e-6 * test["total_patients_shared"].values),
        metrics_row("Logistic regression", yte, p_lr, p_lr),
        metrics_row("XGBoost", yte, p_xgb, p_xgb),
    ]
    metrics = pd.DataFrame(rows)
    print(metrics.round(3).to_string(index=False))

    # The harder, more commercial question: neurologists with NO tie to a target today.
    new = (test["n_target_partners"] == 0).values
    k_new = 50
    y_new = yte.values[new]
    rng = np.random.default_rng(0)
    boot_idx = [rng.integers(0, len(y_new), len(y_new)) for _ in range(500)]
    new_rows = []
    for name, s in [("Baseline: rank by patient volume", test["total_patients_shared"].values),
                    ("Logistic regression", p_lr), ("XGBoost", p_xgb)]:
        s_new = s[new]
        # Bootstrap 95% interval: only ~15 positives here, so show the uncertainty.
        boots = [average_precision_score(y_new[i], s_new[i]) for i in boot_idx if y_new[i].sum() > 0]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        new_rows.append({"Model": name,
                         "Positives found in top 50": int(y_new[np.argsort(-s_new)[:k_new]].sum()),
                         "ROC-AUC": roc_auc_score(y_new, s_new),
                         "PR-AUC": average_precision_score(y_new, s_new),
                         "PR-AUC 95% CI": f"{lo:.2f} - {hi:.2f}"})
    new_metrics = pd.DataFrame(new_rows)

    base_rate = yte.mean()
    with open(REPORT / "metrics.md", "w", encoding="utf-8") as f:
        f.write("# Model evaluation (time-based hold-out)\n\n")
        f.write("Train: 2013 features -> 2014 label. Test: 2014 features -> 2015 label. "
                "All numbers below are from the test year only.\n\n")
        f.write(f"- Neurologists in train: {len(train):,} ({ytr.mean():.1%} positive)\n")
        f.write(f"- Neurologists in test: {len(test):,} ({base_rate:.1%} positive) - "
                f"so a random list of 100 would contain about {base_rate * 100:.0f} future referrers\n\n")
        f.write("## All active neurologists\n\n" + to_md(metrics) + "\n\n")
        f.write(f"## New-referrer opportunity: neurologists with no tie to a target in 2014 "
                f"(n = {new.sum():,}, {int(y_new.sum())} positives)\n\n" + to_md(new_metrics) + "\n")

    # ------------------------------------------------------------ figures
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], ls="--", c=GREY, lw=1, label="Perfectly calibrated")
    for name, p, c in [("Logistic regression", p_lr, ORANGE), ("XGBoost", p_xgb, TEAL)]:
        frac, mean_p = calibration_curve(yte, p, n_bins=10, strategy="quantile")
        ax.plot(mean_p, frac, marker="o", c=c, label=name)
    ax.set(xlabel="Predicted probability", ylabel="Observed share that formed a tie",
           title="Calibration on the 2015 hold-out")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGS / "calibration.png", dpi=150); plt.close(fig)

    ks = np.arange(10, 401, 10)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, s, c in [("XGBoost", p_xgb, TEAL), ("Logistic regression", p_lr, ORANGE),
                       ("Patient volume only", test["total_patients_shared"].values, GREY)]:
        ax.plot(ks, [precision_at_k(yte, s, k) for k in ks], c=c, lw=2, label=name)
    ax.axhline(base_rate, c=NAVY, ls=":", lw=1, label="Random list")
    ax.set(xlabel="Number of neurologists visited (top-k of the list)",
           ylabel="Share who formed a tie", title="Precision at k, 2015 hold-out", ylim=(0, 1.02))
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(FIGS / "precision_at_k.png", dpi=150); plt.close(fig)

    explainer = shap.TreeExplainer(xgb)
    sv = explainer.shap_values(Xte)
    shap.summary_plot(sv, Xte.rename(columns=FEATURES), show=False, max_display=12, plot_size=(8, 5.5))
    plt.title("What drives the score (SHAP, hold-out)")
    plt.tight_layout(); plt.savefig(FIGS / "shap_summary.png", dpi=150); plt.close()
    importance = pd.Series(np.abs(sv).mean(0), index=X_COLS).sort_values(ascending=False)

    # ------------------------------------------------------------ error analysis
    test = test.assign(p=p_xgb)
    test["rank"] = test["p"].rank(ascending=False, method="first")
    top = test["rank"] <= K
    fp, tp = test[top & (test.label == 0)], test[top & (test.label == 1)]
    fn = test[~top & (test.label == 1)]
    profile_cols = ["total_patients_shared", "n_partners", "n_target_partners",
                    "targets_in_zip3", "pagerank", "is_metro"]
    profile = pd.DataFrame({
        f"True positives in top {K} (n={len(tp)})": tp[profile_cols].median(),
        f"False positives in top {K} (n={len(fp)})": fp[profile_cols].median(),
        f"Missed referrers outside top {K} (n={len(fn)})": fn[profile_cols].median(),
    }).rename(index=FEATURES).reset_index(names="Median value")

    def subgroup(col):
        out = []
        for g, d in test.groupby(col):
            auc = roc_auc_score(d.label, d.p) if d.label.nunique() == 2 else np.nan
            out.append({col: g, "Neurologists": len(d), "Positive rate": d.label.mean(),
                        "ROC-AUC": auc, "PR-AUC": average_precision_score(d.label, d.p)
                        if d.label.sum() else np.nan})
        return pd.DataFrame(out).sort_values("Neurologists", ascending=False)

    with open(REPORT / "error_analysis.md", "w", encoding="utf-8") as f:
        f.write("# Error analysis (XGBoost, 2015 hold-out)\n\n")
        f.write("## Top drivers (mean |SHAP|)\n\n")
        f.write(to_md(importance.rename(index=FEATURES).reset_index()
                      .set_axis(["Feature", "Mean |SHAP|"], axis=1), ".3f") + "\n\n")
        f.write("## Who the model gets right and wrong (medians)\n\n" + to_md(profile, ".4f") + "\n\n")
        f.write("## Performance by specialty\n\n" + to_md(subgroup("specialty")) + "\n\n")
        f.write("## Performance by metro vs non-metro\n\n" + to_md(subgroup("metro")) + "\n")

    # ------------------------------------------------------------ final ranked list
    # Refit on every labelled year, then score 2015 features -> likely ties in 2016.
    labelled = df[df.year.isin([2013, 2014])]
    final = make_xgb().fit(labelled[X_COLS], labelled["label"])
    score["score"] = final.predict_proba(score[X_COLS])[:, 1]
    sv_final = shap.TreeExplainer(final).shap_values(score[X_COLS])
    top3 = np.argsort(-sv_final, axis=1)[:, :3]
    score["top_drivers"] = ["; ".join(FEATURES[X_COLS[j]] for j in row) for row in top3]
    score["already_refers"] = score["n_target_partners"] > 0
    score = score.sort_values("score", ascending=False)
    score["rank"] = np.arange(1, len(score) + 1)
    cols = ["rank", "npi", "name", "specialty", "city", "zip5", "metro", "score", "already_refers",
            "medicare_eligible_2026", "top_drivers"] + X_COLS
    score[cols].to_csv(REPORT / "target_list_texas.csv", index=False)

    # Network picture: Houston neurology network, nodes coloured by role, sized by score.
    providers = pd.read_parquet(DATA / "providers_tx.parquet").set_index("npi")
    e = pd.read_parquet(DATA / "edges_tx_2015.parquet")
    hou = set(score.loc[score.metro == "Houston", "npi"].head(40)) | \
        set(providers.index[(providers.role == "target") & providers.zip5.str[:3].isin(["770"])])
    sub = e[e.from_npi.isin(hou) & e.to_npi.isin(hou)]
    G = nx.from_pandas_edgelist(sub, "from_npi", "to_npi", "patients", create_using=nx.DiGraph())
    sc = score.set_index("npi")["score"]
    colors = [TEAL if providers.at[n, "role"] == "target" else ORANGE for n in G]
    sizes = [60 + 600 * sc.get(n, 0.1) for n in G]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    pos = nx.spring_layout(G, seed=7, k=0.35)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, edge_color=GREY, arrowsize=6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes, linewidths=0)
    ax.scatter([], [], c=ORANGE, label="Neurologist (size = 2016 score)")
    ax.scatter([], [], c=TEAL, label="Epileptologist / neurosurgeon / epilepsy unit")
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Houston: top-40 scored neurologists and nearby targets (2015 ties)")
    ax.axis("off"); fig.tight_layout(); fig.savefig(FIGS / "network_houston.png", dpi=150); plt.close(fig)

    print("\nTop drivers:", ", ".join(FEATURES[c] for c in importance.index[:3]))
    print(f"Saved ranked list of {len(score):,} neurologists -> report/target_list_texas.csv")


if __name__ == "__main__":
    main()
