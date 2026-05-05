# Results Summary

This document summarizes the saved results included in `result/`.

## Prepared data

Prepared-data metadata is stored in:

```text
artifact/data/prepared/summary.json
```

| Split | Documents | Examples | WikiText examples | Medical examples |
|---|---:|---:|---:|---:|
| Train | 16,837 | 20,368 | 9,882 | 10,486 |
| Validation | 2,105 | 2,561 | 1,175 | 1,386 |
| Test | 2,105 | 2,555 | 1,227 | 1,328 |

Each example uses a 96-token prompt and a 96-token target continuation.

## Overall validation and test performance

Metrics are stored in `result/metrics/`. A compact CSV summary is stored in:

```text
result/tables/metrics_summary.csv
```

| Model | Val loss | Val PPL | Test loss | Test PPL | Test PPL reduction |
|---|---:|---:|---:|---:|---:|
| Frozen base | 2.801 | 16.46 | 2.822 | 16.81 | 0.0% |
| Static write-strength | 2.744 | 15.55 | 2.765 | 15.88 | 5.5% |
| Prompt-conditioned write-strength | 2.741 | 15.51 | 2.762 | 15.84 | 5.8% |
| Static residual-stream re-aggregation | 2.714 | 15.09 | 2.733 | 15.38 | 8.5% |
| Prompt-conditioned residual-stream re-aggregation | 2.705 | 14.95 | 2.723 | 15.23 | 9.4% |

Main takeaway: residual-stream re-aggregation improves more than simple write-strength scaling, while prompt conditioning adds a smaller but consistent gain within each family.

## Per-domain test performance

| Model | WikiText loss | WikiText PPL | Medical loss | Medical PPL |
|---|---:|---:|---:|---:|
| Frozen base | 3.298 | 27.06 | 2.383 | 10.83 |
| Static write-strength | 3.224 | 25.12 | 2.342 | 10.40 |
| Prompt-conditioned write-strength | 3.220 | 25.03 | 2.340 | 10.38 |
| Static residual-stream re-aggregation | 3.178 | 24.01 | 2.322 | 10.19 |
| Prompt-conditioned residual-stream re-aggregation | 3.167 | 23.73 | 2.314 | 10.11 |

The best model improves both domains, so the overall improvement is not driven by a single source.

## Controller analysis

Controller-inspection metrics are stored in:

```text
result/analysis/mlp_inspection.json
```

Test split summary:

| Prompt-conditioned model | Collapse cosine | Domain gap | Mean variance | Mean domain difference |
|---|---:|---:|---:|---:|
| Write-strength | 0.948 | 0.056 | 0.113 | 0.371 |
| Residual-stream re-aggregation | 0.877 | 0.210 | 0.954 | 1.161 |

Main takeaway: prompt-conditioned re-aggregation is less collapsed and more domain-sensitive than prompt-conditioned write-strength scaling.

## Figures

Generated figures are stored under `result/figures/` in both PNG and PDF formats:

```text
01_test_perplexity_comparison
02_test_perplexity_by_source
03_relative_improvement_over_frozen_base
04_mlp_inspection_summary
05_per_layer_mlp_behavior
```

Regenerate all figures with:

```bash
python script/5_make_plots.py
```
