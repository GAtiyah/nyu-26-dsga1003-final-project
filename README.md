# Prompt-Conditioned Residual Re-aggregation for Lightweight Domain Adaptation

This repository contains the code, prepared data, saved metrics, and figures for a DS-GA 1003 final project on lightweight residual adaptation of a frozen language model.

The project studies whether a frozen `Qwen/Qwen3-0.6B` backbone can be adapted on mixed-domain continuation data using small residual controllers. Two intervention families are compared:

1. Layerwise write-strength scaling.
2. Residual-stream re-aggregation.

For each family, the repository evaluates both a static controller and a prompt-conditioned controller. The main empirical finding is that residual-stream re-aggregation is more effective than simple write-strength scaling, while prompt conditioning gives a smaller but consistent additional gain in this setup.

## Authors

- Peng Zhao
- Jinghan Lei
- Yu Gu

Center for Data Science, New York University

## Repository contents

```text
.
├── 1003_project_proposal.md          # Original project proposal
├── design_note.txt                   # Implementation and experiment-design notes
├── example_usuage.txt                # Original stage-by-stage command sketch
├── requirements.txt                  # pip environment
├── environment.yml                   # conda environment
├── REPRODUCIBILITY.md                # Detailed reproduction instructions
├── RESULTS.md                        # Saved results and figure summary
├── script/                           # Stage 0-5 code
├── artifact/data/raw/                # Downloaded raw datasets saved by Hugging Face Datasets
├── artifact/data/prepared/           # Prepared prompt/answer Arrow files
└── result/
    ├── metrics/                      # Stage-3 evaluation JSON files
    ├── analysis/                     # Stage-4 controller-inspection JSON
    ├── tables/                       # Summary CSV
    └── figures/                      # Publication-style plots
```

Large local model and checkpoint artifacts are intentionally not tracked:

```text
artifact/model/
artifact/checkpoints/
*.pt
*.pth
*.bin
*.safetensors
```

The saved metrics and figures are included, so the reported results can be inspected without rerunning training. To fully reproduce training, download the model and rerun the Stage 0-4 scripts.

## Data

The experiment uses continuation examples from:

- WikiText-2 raw (`wikitext`, `wikitext-2-raw-v1`)
- Medical abstracts (`TimSchopf/medical_abstracts`)

Prepared examples use:

- 96 prompt tokens
- 96 target continuation tokens
- next-token cross-entropy only on target tokens
- deterministic seed `42`

Prepared-data summary:

| Split | Examples | WikiText | Medical |
|---|---:|---:|---:|
| Train | 20,368 | 9,882 | 10,486 |
| Validation | 2,561 | 1,175 | 1,386 |
| Test | 2,555 | 1,227 | 1,328 |

The full summary is in `artifact/data/prepared/summary.json`.

## Main results

Test perplexity is lower-is-better.

| Model | Test loss | Test PPL | Test PPL reduction vs frozen |
|---|---:|---:|---:|
| Frozen base | 2.822 | 16.81 | 0.0% |
| Static write-strength | 2.765 | 15.88 | 5.5% |
| Prompt-conditioned write-strength | 2.762 | 15.84 | 5.8% |
| Static residual-stream re-aggregation | 2.733 | 15.38 | 8.5% |
| Prompt-conditioned residual-stream re-aggregation | 2.723 | 15.23 | 9.4% |

The complete metric table is in `result/tables/metrics_summary.csv`.

## Quick setup

Create a clean Python environment. Conda is recommended because the project uses PyTorch, Transformers, Datasets, and PyArrow.

```bash
conda env create -f environment.yml
conda activate residual-reaggregation
```

Alternatively:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The pinned NumPy/PyArrow range avoids common ABI issues from mixing an old PyArrow build with NumPy 2.x.

## Reproducing the pipeline

The full command sequence is:

```bash
python script/0_download_model_and_data.py
python script/1_prepare_data.py

python script/2_train_static.py
python script/2_train_prompt_conditioned.py
python script/2_train_static_reaggregation.py
python script/2_train_prompt_conditioned_reaggregation.py

python script/3_eval_frozen_base.py
python script/3_eval_static.py
python script/3_eval_prompt_conditioned.py
python script/3_eval_static_reaggregation.py
python script/3_eval_prompt_conditioned_reaggregation.py

python script/4_inspect_MLPs.py
python script/5_make_plots.py
```

See `REPRODUCIBILITY.md` for details, expected outputs, hardware notes, and caveats.

## Inspecting saved results without retraining

The saved outputs already included in this repository are enough to inspect the paper results:

```bash
python script/5_make_plots.py
```

This regenerates:

```text
result/tables/metrics_summary.csv
result/figures/*.png
result/figures/*.pdf
```

## Notes for readers

- The repository folder itself may need to be initialized as a Git repository before sharing through GitHub.
- Model weights and trained checkpoints are excluded to keep the repository lightweight.
- The metrics in `result/metrics/` were produced on CUDA with `torch.bfloat16`.
- The write-strength metrics use older result filenames (`static_residual.json` and `prompt_conditioned.json`) for backward compatibility; the README and paper use the clearer names `static write-strength` and `prompt-conditioned write-strength`.
