"""Rebuild the paper's artifacts + map the reproduction pipeline.

This is an HPC project: the headline numbers need the credentialed 34 GB MIMIC-IV-ECG waveforms + the
MIMIC-IV hosp module (gold ICD-10 labels) + GPU SLURM jobs, so a full rebuild is not local. This entry
point regenerates the locally-runnable method schematic and documents the figure/table map. Run from
this directory.

    python paper.py        # regenerate the method schematic (Fig 1, bttr_vs_bttn.png); print the map

FULL from-scratch reproduction (cluster): ONE command builds every cohort, extracts every spectral
tensor from the zip, pretrains the SSL encoder, trains every model, and writes every table/figure into
code/results/ - nothing is assumed pre-computed:

    bash run_all.sh                # submit the whole SLURM dependency graph (prerequisites checked up front)
    FRESH=1 bash run_all.sh        # additionally wipe existing intermediates first (true from-scratch re-test)

run_all.sh is the single documented entry point; after the graph COMPLETES, `python paper.py` assembles
the schematic and the figures/tables are already in code/results/.

Figure -> script -> manuscript:
    bttr_vs_bttn.png         figures/arch_diagram.py            method schematic (local, no data)
    bttn_interpret.png       experiments/interpret_bttn.py      glass-box substrate map (P-wave 2 Hz / f-wave 4-8 Hz; leads I/II/V1/V5)
    clin_eval.png            experiments/clin_eval.py           calibration + decision curve + risk deciles (true-incidence PPV/NNM)
    beat_saliency*.png       figures/beat_saliency*.py          energy-normalised cardiac-phase saliency
    external_code15_afsr.png experiments/external_code15.py     external substrate transfer (CODE-15)

Table -> source (all committed CSV/npz in code/results/, produced by run_all.sh):
    Table I (1-yr ladder + CIs)   core/bench.py -> bench_spectral_icd_h365.npz ; experiments/compute_table1_cis.py -> table1_cis.csv
    5-yr ladder                   core/bench.py -> bench_spectral_icd_h1825.npz
    multi-horizon + C-index       experiments/multihorizon.py -> multihorizon_icd_spectral.csv
    faithfulness controls         experiments/faithfulness_controls.py -> faithfulness_controls.csv
    block-structure ablation      experiments/ablation_blockstructure.py -> ablation_blockstructure.csv
    intervention curve            experiments/intervention_curve.py -> intervention_curve.csv
    clinical operating points     experiments/clin_eval.py -> clin_eval_operating_points.csv
    cohort accounting             core/cohort_accounting.py
"""
import os
import subprocess
import sys


def main():
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.abspath(__file__)))
    print("== regenerate the method schematic (Fig 1, local, no data) ==", flush=True)
    subprocess.run([sys.executable, "figures/arch_diagram.py"], check=True, env=env)
    print("\nDONE (schematic). The headline numbers + the remaining figures/tables are produced by the "
          "full pipeline: bash run_all.sh (credentialed 34 GB cohort + GPU). See this file's header for the map.")


if __name__ == "__main__":
    main()
