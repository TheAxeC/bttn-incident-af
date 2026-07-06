# BTTN: interpretable incident-AF prediction from the sinus ECG and clinical risk factors

Code for *"An Interpretable Block-Term Tensor Network for Incident Atrial-Fibrillation Prediction from the Sinus-Rhythm 12-Lead ECG and Clinical Risk Factors"* (Faes and Peeters, Artificial Intelligence in Medicine). The model reads one sinus-rhythm 12-lead ECG plus standard clinical risk factors (age, sex, ECG intervals) and predicts future-onset (incident) atrial fibrillation. It is a glass-box: the ECG evidence routes through interpretable per-block frequency and lead signatures, and the interpretability is measured (a faithfulness probe + control probes + an intervention test), not asserted. The fused model beats a clinical risk-factor baseline and a black-box CNN at competitive accuracy. The algorithm is in `core/bttn.py`.

## What BTTN is

Each ECG becomes an alignment-free spectrogram tensor (frequency x time x lead). The Block-Term Tensor Network factorizes it into K blocks, each with an interpretable frequency factor `A_k` and lead factor `C_k` plus a multi-scale temporal convolution bank, pooled by temporal attention and trained with a Cox partial-likelihood survival loss. The block factors are readable per-block frequency-and-lead signatures, and one survival model gives risk at any horizon. Block-term regression alone is linear and cannot match a CNN; BTTN keeps the interpretable structure while adding the capacity to match it. Clinical risk factors enter through a transparent linear term added to the head log-risk that bypasses the block layer, so the interpretable signatures and the faithfulness probe are unchanged (demographics at no interpretability cost).

## Reproduce (one command, from absolute scratch)

This is an HPC project (credentialed 34 GB MIMIC-IV-ECG waveforms + the MIMIC-IV hospital module for gold ICD-10 labels + GPU SLURM). Given the data and the packed environment, one command builds every cohort, extracts every spectral tensor from the zip, pretrains the SSL encoder, trains every model, and writes every table and figure. Nothing is assumed pre-computed:

```bash
bash run_all.sh            # submit the whole SLURM dependency graph (prerequisites checked up front)
FRESH=1 bash run_all.sh    # additionally wipe existing intermediates first (true from-scratch re-test)
```

`run_all.sh` checks its prerequisites (the waveform zip, the hosp module, the ECG metadata, the packed env) and fails at the top with the exact fix if one is missing; it orders the jobs so no step ever reads a cache another has not built yet. After the graph completes, the tables and figures are in `results/`, and `python paper.py` regenerates the method schematic and prints the figure-to-script map.

## Layout (a Python package; run from this directory with `PYTHONPATH=.`)

```
core/        library + harness: config.py (paths, outcome definition, params), preprocess.py (spectral
             tensorization), data.py (on-demand waveform read from the zip), bttn.py (the BTTN model +
             Cox loss + the clinical-fusion term + the 3-seed ensemble - the algorithm), cnn.py (the CNN
             ceiling), bttr_model.py + survival.py (the linear BTTR baseline), cohort.py +
             cohort_accounting.py (metadata-only cohort builder), bench.py (CV harness + paired
             bootstrap), ssl_pretrain.py (contrastive pretraining), competing_risks.py (gold-ready head)
experiments/ the analyses: multihorizon.py, ablation.py (block-count sweep), ablation_blockstructure.py,
             faithfulness_controls.py, intervention_curve.py, interpret_bttn.py, clin_eval.py,
             external_{code15,ptbxl}.py, compute_table1_cis.py
figures/     the plotting scripts (arch_diagram.py is the method schematic)
bttrlib/     vendored linear block-term regression (the baseline; do not edit)
hpc/         job.slurm (the env-staging GPU runner); run_all.sh submits the graph through it
attic/       dead-ends + dev tools, not part of the paper (see attic/README.md)
results/     generated CSVs + figures (not in the deposit)
paper.py     regenerate the schematic + print the figure/table map
run_all.sh   full from-scratch reproduction (submits the SLURM dependency graph)
```

Paths resolve through `core/config.py` (`AFNET_ECG_ZIP` / `AFNET_HOSP_DIR` / `AFNET_RESULTS` override the data and output locations; data lives under `~/data/`).

## Pipeline (what `run_all.sh` runs, in dependency order)

| step | command | output |
|---|---|---|
| cohorts (gold) | `python -m core.cohort --source icd --horizon 365` (+ `--survival`, `--horizon 1825`) | `results/cohort_icd_*.csv` |
| SSL pretrain | `python -m core.ssl_pretrain --n 30000 --epochs 60` | `results/ssl_encoder.pt` |
| Table I (1-yr) | `python -m core.bench --source icd --horizon 365 --seeds 5 --methods clinical,bttr,cnn,bttn,bttn_ssl,bttn_cov,bttn_ssl_cov --tensor spectral --neg_ratio 4` | `results/bench_spectral_icd_h365.npz` |
| per-row CIs | `python -m experiments.compute_table1_cis` | `results/table1_cis.csv` |
| 5-yr | `python -m core.bench --source icd --horizon 1825 --seeds 5 --methods clinical,bttn,bttn_ssl,bttn_cov,bttn_ssl_cov --tensor spectral --neg_ratio 4` | `results/bench_spectral_icd_h1825.npz` |
| multi-horizon | `python -m experiments.multihorizon --source icd --tensor spectral` | `results/multihorizon_icd_spectral.csv` |
| block-structure ablation | `python -m experiments.ablation_blockstructure --source icd --horizon 365 --neg_ratio 4` | `results/ablation_blockstructure.csv` |
| block-count (K) sweep | `python -m experiments.ablation --mode K --source icd --horizon 365 --neg_ratio 4 --seeds 3` | `results/ablation_K.csv` |
| faithfulness + intervention | `python -m experiments.faithfulness_controls --source icd ...` + `intervention_curve --source icd ...` | `results/faithfulness_controls.csv`, `intervention_curve.csv` |
| substrate map | `python -m experiments.interpret_bttn --source icd --horizon 365 --neg_ratio 4` | `results/bttn_interpret.png` |
| clinical utility | `python -m experiments.clin_eval` | `results/clin_eval_operating_points.csv`, `clin_eval.png` |
| external | `python -m experiments.external_code15 --task afsr` / `external_ptbxl --task detect` | `results/external_*` |

The `bttn_cov` / `bttn_ssl_cov` methods are the clinical-fusion models (BTTN, and BTTN+SSL, with the age/sex/interval covariate term). Each `bench` or experiment builds its spectral tensor cache from the zip on first use, and `run_all.sh` orders the jobs so no step reads a cache another has not built.

## Data

All datasets are public under credentialed access; no raw data is redistributed. MIMIC-IV-ECG (~800k ECGs / 161k patients, the primary cohort, PhysioNet DUA); the gold ICD-10 I48 labels come from the MIMIC-IV hospital module (a separate credentialed agreement). External validation on CODE-15 and PTB-XL (AF identification/detection + substrate transfer, a different and easier task than dated incidence). Waveforms and tensor caches are read from `~/data/` (paths in `core/config.py`); they are not in the repo.

## Conventions

Patient-level cross-validation, patient-bootstrap 95% confidence intervals, paired tests, matched protocol across methods. The vendored `bttrlib/` block-term regression is the linear baseline and is not edited. License: MIT (`LICENSE`).
