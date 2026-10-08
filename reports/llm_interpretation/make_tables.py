"""
Builds the LaTeX tables and the forest plot of the LLM-interpretation report from data/xai/results/analysis.json.
Run from the repo root: python reports/llm_interpretation/make_tables.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
A = json.loads((ROOT / "data/xai/results/analysis.json").read_text())
VAR = {"multiomics": "Multi-omics", "l1000": "L1000"}
COND = {"attribution_multiomics": "Attribution agent (multi-omics)", "attribution_l1000": "Attribution agent (L1000)",
        "kg_only": "KG-only agent", "select_attribution_multiomics": "Attribution selector (multi-omics)",
        "select_attribution_l1000": "Attribution selector (L1000)", "select_random": "Random convergent pathway"}


def p(x):
    return "$<$0.001" if x < 0.001 else f"{x:.3f}" if x < 0.1 else f"{x:.2f}"


def m(x, fmt="+.3f"):
    """Signed number in math mode, so negatives get a true minus sign."""
    return f"${x:{fmt}}$"


def ci(c):
    return f"[{m(c[0], '+.2f')}, {m(c[1], '+.2f')}]"


def row(label, r):
    return (f"{label} & {r['n']} & {r['n_cells']} & {m(r['mean_delta'])} & {ci(r['ci95_cluster_bootstrap'])} & "
            f"{m(r['d_z'], '+.2f')} & {p(r['p_mixed'])} & {p(r['p_wilcoxon'])} \\\\")


def write(name, lines):
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "tables" / f"{name}.tex").write_text("\n".join(lines) + "\n")


HEAD = [r"\begin{tabular}{@{}lrrrcrrr@{}}", r"\toprule",
        r"Comparison & $n$ items & Cell lines & Mean $\Delta$ & 95\% CI & $d_z$ & $p_{\text{mixed}}$ & $p_{\text{Wilcoxon}}$ \\",
        r"\midrule"]

# H1: primary, sensitivity, top-3
lines = list(HEAD)
for block, title in (("H1", "Primary: top-1 pathway, items where both agents made a verified claim"),
                     ("H1_sensitivity_imputed", "Sensitivity: abstentions scored as 0 (all 250 items)"),
                     ("H1_top3", "Secondary: mean over the top-3 pathways")):
    lines.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{title}}}}} \\")
    for v in ("multiomics", "l1000"):
        lines.append(row(rf"\quad {VAR[v]}", A[block][v]))
lines += [r"\bottomrule", r"\end{tabular}"]
write("h1", lines)

# H2 and H3
h2, h3 = A["H2"], A["H3"]
write("h2h3", [
    r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
    r"Hypothesis & Variant & $n$ & Estimate & SE / 95\% CI & $p$ \\", r"\midrule",
    *[rf"H2: attribution $\times$ synergy interaction & {VAR[v]} & {h2[v]['n']} & {m(h2[v]['interaction'])} & "
      rf"{h2[v]['se']:.3f} & {p(h2[v]['p'])} \\" for v in ("multiomics", "l1000")],
    rf"H3: $\Delta_{{\text{{L1000}}}} - \Delta_{{\text{{multi-omics}}}}$ & both & {h3['n']} & {m(h3['mean_delta'])} & "
    rf"{ci(h3['ci95_cluster_bootstrap'])} & {p(h3['p_mixed'])} \\",
    r"\bottomrule", r"\end{tabular}"])

# Exploratory decomposition
ex = A["exploratory_deterministic"]
lab = {"select_attribution_multiomics - select_random": "Attribution selector $-$ random (multi-omics)",
       "select_attribution_l1000 - select_random": "Attribution selector $-$ random (L1000)",
       "attribution_multiomics (LLM) - select_attribution_multiomics": "LLM agent $-$ attribution selector (multi-omics)",
       "attribution_l1000 (LLM) - select_attribution_l1000": "LLM agent $-$ attribution selector (L1000)",
       "kg_only (LLM) - select_random": "KG-only LLM agent $-$ random"}
lines = list(HEAD)
lines.append(r"\multicolumn{8}{@{}l}{\emph{What the attributions add (no LLM)}} \\")
lines += [row(rf"\quad {lab[k]}", ex[k]) for k in ("select_attribution_multiomics - select_random",
                                                    "select_attribution_l1000 - select_random")]
lines.append(r"\multicolumn{8}{@{}l}{\emph{What the LLM adds on top of the same input}} \\")
lines += [row(rf"\quad {lab[k]}", ex[k]) for k in ("attribution_multiomics (LLM) - select_attribution_multiomics",
                                                    "attribution_l1000 (LLM) - select_attribution_l1000",
                                                    "kg_only (LLM) - select_random")]
lines += [r"\bottomrule", r"\end{tabular}"]
write("decomposition", lines)

# Agent behaviour and claim descriptives
D = A["descriptives"]
order = ["attribution_multiomics", "attribution_l1000", "kg_only", "select_attribution_multiomics",
         "select_attribution_l1000", "select_random"]
lines = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
         r"Condition & Stratum & Accepted & Abstained & Rejected & First-try valid & Turns & Output tokens \\",
         r"\midrule"]
for c in order[:3]:
    for d in (x for x in D if x["condition"] == c):
        lines.append(rf"{COND[c]} & {d['stratum']} & {d['rate_accepted']:.3f} & {d['rate_abstained']:.3f} & "
                     rf"{d['rate_rejected']:.3f} & {d['first_try_valid']:.3f} & {d['mean_turns']:.1f} & "
                     rf"{d['mean_output_tokens']:,.0f} \\")
    lines.append(r"\addlinespace")
lines[-1:] = [r"\bottomrule", r"\end{tabular}"]
write("behaviour", lines)

lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
         r"Condition & Stratum & Mean spec (top-1) & Common-essential fraction & Pathway diversity & Median pathway size \\",
         r"\midrule"]
for c in order:
    for d in (x for x in D if x["condition"] == c):
        div = "--" if c == "select_random" else f"{d['pathway_diversity']:.2f}"
        lines.append(rf"{COND[c]} & {d['stratum']} & {m(d['mean_spec_top1'])} & {d['mean_common_essential_frac']:.2f} & "
                     rf"{div} & {d['median_pathway_size']:.0f} \\")
    lines.append(r"\addlinespace")
lines[-1:] = [r"\bottomrule", r"\end{tabular}"]
write("pathways", lines)

# Forest plot: every paired comparison on the synergy stratum
rows = [("H1 multi-omics: attribution agent $-$ KG-only", A["H1"]["multiomics"], "#2a78d6"),
        ("H1 L1000: attribution agent $-$ KG-only", A["H1"]["l1000"], "#2a78d6"),
        ("H1 sensitivity, multi-omics (abstain = 0)", A["H1_sensitivity_imputed"]["multiomics"], "#2a78d6"),
        ("H1 sensitivity, L1000 (abstain = 0)", A["H1_sensitivity_imputed"]["l1000"], "#2a78d6"),
        ("H3: L1000 $-$ multi-omics advantage", A["H3"], "#2a78d6"),
        ("Attribution selector $-$ random (multi-omics)", ex["select_attribution_multiomics - select_random"], "#8a8a8a"),
        ("Attribution selector $-$ random (L1000)", ex["select_attribution_l1000 - select_random"], "#8a8a8a"),
        ("LLM agent $-$ attribution selector (multi-omics)",
         ex["attribution_multiomics (LLM) - select_attribution_multiomics"], "#8a8a8a"),
        ("LLM agent $-$ attribution selector (L1000)", ex["attribution_l1000 (LLM) - select_attribution_l1000"], "#8a8a8a"),
        ("KG-only LLM agent $-$ random", ex["kg_only (LLM) - select_random"], "#8a8a8a")]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(6.6, 3.4))
for i, (lab_, r, col) in enumerate(rows[::-1]):
    lo, hi = r["ci95_cluster_bootstrap"]
    ax.plot([lo, hi], [i, i], color=col, lw=1.8)
    ax.plot(r["mean_delta"], i, "o", color=col, ms=5)
    ax.text(0.62, i, f"n={r['n']}", va="center", fontsize=7, color="#555555")
ax.axvline(0, color="#555555", lw=0.8)
ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
ax.set_xlim(-0.3, 0.7)
ax.set_xlabel("mean paired difference in specificity (SD units), 95% CI")
ax.grid(axis="x", color="#e6e6e6", lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "fig_forest.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_forest.png", bbox_inches="tight", dpi=200)
print("tables and figure written to", OUT)
