# README paper figures

These images are exported from the authors' accompanying paper,
[*A Closer Look at Agentic BBO: Benchmarking LLM Agents for Black-Box Optimization*](https://arxiv.org/abs/2610.12183v1),
arXiv v1 (8 October 2026). They reproduce the paper's results, rather than a
live leaderboard or current provider pricing. The figure content is unchanged;
only the empty top margin of Figure 1 is omitted.

Source: the `picture/` directory in the [public TeX source](https://arxiv.org/src/2610.12183v1).
The manuscript and source PDFs are not bundled in this repository.

| Image | Paper figure | Source PDF | Export |
| --- | --- | --- | --- |
| `frontier-model-comparison.png` | 1 | `frontier_model_scores_final_colors_mixed_markers.pdf` | 2400 px wide, omit the first 300 blank rows of the 1350 px render |
| `agentic-bbo-overview.png` | 2 | `Traditional_BO_vs_Agentic_BBO_v4.pdf` | 2000 px wide |
| `task-information.png` | 4(a) | `task_information_bars.pdf` | 1600 px wide |

To reproduce with Poppler, run the following from the repository root with
`PAPER_SOURCE` pointing to the extracted arXiv v1 source:

```bash
pdftoppm -singlefile -scale-to 2400 -y 300 -H 1050 -png \
  "$PAPER_SOURCE/picture/frontier_model_scores_final_colors_mixed_markers.pdf" \
  docs/assets/frontier-model-comparison
pdftoppm -singlefile -scale-to 2000 -png \
  "$PAPER_SOURCE/picture/Traditional_BO_vs_Agentic_BBO_v4.pdf" \
  docs/assets/agentic-bbo-overview
pdftoppm -singlefile -scale-to 1600 -png \
  "$PAPER_SOURCE/picture/task_information_bars.pdf" \
  docs/assets/task-information
```
