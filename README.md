# cheerios

Neural population decoding pipeline for the DMS temporal waiting task. Uses Ridge regression to decode elapsed time from MSN (medium spiny neuron) population firing rates, and characterizes how the DMS encodes a behavioral clock — including clock-speed extraction, reward history effects, and confound controls.

## Setup

```bash
git clone https://github.com/rebz444/cheerios.git
cd cheerios
conda env create -f environment.yml
conda activate cheerios
```

### Data

Data is not in the repo (gitignored). Copy your data directory to the expected location before running:

```
~/data/neural_data/
    logs/               # CSVs including RZ_msn_waveform.csv
    session_pickles/    # Per-session spike/trial pickle files
    figures/            # Output figures written here
```

If your data lives somewhere else, edit `paths.py` to update `DATA_DIR`.

## Pipeline

Scripts are numbered in run order:

| Step | Script | Description |
|------|--------|-------------|
| 0a | `0a_datajoint_processing_check.ipynb` | DataJoint validation |
| 0b | `0b_neural_data_processing.ipynb` | Neural data ingestion |
| 0c | `0c_neural_data_examining.ipynb` | Data examination |
| 0d | `0d_neural_data_quality_metrics.ipynb/py` | Spike sorting QC |
| 0e–0j | `0e_` … `0j_` | Neuron location matching, waveform diagnostics, cell-type relabeling, region labeling, track deviation |
| 1 | `1a_plot_raster_histo_by_quantile.ipynb` | Raster + PSTH plots |
| 2 | `2_neuron_clustering.ipynb` | Neuron clustering & firing rate analysis |
| 3 | `3a_bg_predict_time_waited.ipynb` | Behavioral prediction from background activity |
| 4 | `4a/4b/4c_encoding_GLM*.ipynb` | GLM encoding models |
| 5 | `5a_timing_analysis.ipynb` | Timing trajectory analysis |
| 6 | `6_trial_firing_rate.ipynb` | Trial-level firing rates |
| — | `population_decoder_v2.py` | Full population decoding pipeline (see below) |

## Population Decoder

`population_decoder_v2.py` is the main analysis script. Run it directly:

```bash
python population_decoder_v2.py
```

It runs on all qualifying sessions (>15 Tier 2 MSN units, >150 trials) and produces per-session and cross-session figures in `~/data/neural_data/population_decoding/results/`.

To regenerate summary plots from existing results without re-running the decoder:

```python
# In population_decoder_v2.py, set:
PLOT_ONLY = True
```

## Key Files

| File | Purpose |
|------|---------|
| `population_decoder_v2.py` | Full decoding pipeline with clock speed, history effects, confound controls |
| `population_decoder.py` | Base decoder utilities and anchor comparison |
| `constants.py` | Animal group assignments, region colors, QC thresholds |
| `paths.py` | Data directory paths — edit this for your machine |
| `utils.py` | Shared helper functions |
