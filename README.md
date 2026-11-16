# Local Intrinsic Dimension during Transformer Pre-Training



## Overview

Code for my bachelor thesis *Analyzing and Intervening on Local Intrinsic Dimensions during Pre-Training of Small Transformer Models*.

Loss and task metrics say little about the geometry of a language model’s representation space during training. This project trains a small decoder-only Transformer (~17M parameters) from scratch on a subset of FineWeb and tracks the **mean local intrinsic dimension (LID)** of its contextual token embeddings throughout pre-training. LID is estimated with a localized TwoNN estimator and needs neither labeled data nor task-specific metrics.

The project addresses three questions:

- How does mean LID develop during pre-training?
- How does it relate to cross-entropy loss?
- Does increasing model depth lower the LID, and how does this affect loss?

The setup follows the LID methodology of [Ruppik et al. (2025)](https://proceedings.neurips.cc/paper_files/paper/2025/file/61c2c6338033da68885e0226881cbe71-Paper-Conference.pdf), applied to the pre-training of small models instead of large models during fine-tuning. The baseline runs use 20,000 steps and 5 seeds; the depth experiments use 3 seeds per configuration.

## Results

1. **The mean local intrinsic dimension follows a non-monotonic, three-phase trajectory**
   
   Across five training seeds, mean LID drops from approximately 116.5 at initialization to 9.2 at step 200, rises to about 16.3 by step 2,400, then settles around 17-18 till the end of training. 

2. **The relationship between LID and loss depends on the training phase**

   Both LID and loss drop in the first phase. Between steps 200 and 2,400, LID rises while loss keeps falling, exhibiting a strong negative correlation. In the late phase both stabilize and the relationship weakens. LID therefore has no monotonic relationship with performance measured through loss during pre-training, and the rise in LID does not indicate overfitting. This differs from findings reported for fine-tuning by [Ruppik et al. (2025)](https://proceedings.neurips.cc/paper_files/paper/2025/file/61c2c6338033da68885e0226881cbe71-Paper-Conference.pdf).

   ![Cross-entropy losses and mean local intrinsic dimension over the course of pre-training. Lines and shaded regions show the mean and standard deviation, respectively, across five baseline seeds. Checkpoints from steps 200 to 20, 000 are shown. LID was estimated using (M, N, L) = (256, 16384, 128).](/assets/baseline_loss_and_lid.png)

4. **Greater depth lowers coincides with lower LID and loss**

   Going from 9 to 25 layers (in steps of 4) reduces validation LID from ~17.94 to ~15.26 and validation loss from ~2.02 to ~1.87, with diminishing returns for additional layers. This is not evidence of a causal link, since depth also changes parameter count, compute cost and long-skip connections.

   ![Cross-entropy loss (top row) and mean local intrinsic dimension (bottom row) during pre-training for models with 9, 13, 17, 21 and 25 layers, evaluated on fixed first shard training (left) and validation (right) samples. Lines and shaded regions show the mean and standard deviation across three training seeds. Checkpoints from step 200 to 20, 000 are shown. LID was estimated using (M, N, L) = (256, 16384, 128).](/assets/depth_loss_and_lid.png)

Takeaway: LID provides information about internal geometry that loss alone does not. It is useful as a complementary diagnostic of pre-training dynamics, not as a sole performance indicator.

## Installation

### Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) package manager

### Setup

Clone the repository and install python dependencies with uv:

```bash
git clone https://github.com/based42/parameter-golf-lid-analysis.git
cd parameter-golf-lid-analysis
uv sync
```

## Usage

### 1. Download the dataset

```bash
uv run python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 105
```

This downloads the first 105 training shards along with populating `./data/datasets/fineweb10B_sp1024/` and `./data/tokenizers/` using the script provided by the [parameter-golf](https://github.com/openai/parameter-golf) repository.

### 2. Train the model

Start a pre-training run:

```bash
uv run python train_gpt.py
```

Checkpoints are saved every `x` steps determined by `CHECKPOINT_EVERY`.

### 3. Compute local intrinsic dimension estimates

Run the LID analysis based on the checkpoints saved from `train_gpt.py`:

```bash
uv run python estimate_analyze_lid.py \
  --num-sampled-sequences 256 \
  --num-sampled-token-vectors 16384 \
  --neighborhood-size 128 \
  --run-id <RUN_ID>
```

Outputs the results to `lid.csv`.

## Project Structure

```bash
runs/<run-id>/
  config.json                         # Training hyperparameters
  log.txt                             # Training log
  metrics_train.csv                   # Current-batch loss
  metrics_val.csv                     # Full-validation loss and bits per byte
  metrics_lr.csv                      # Learning rates
  checkpoints/model_step_<step>.pt    # Model weights at checkpoints
  final_model.pt                      # Final uncompressed model weights
  final_model.int8.ptz                # Upstream quantized export
  analysis/<analysis-id>/
    config.json                       # Analysis arguments
    log.txt                           # Analysis source, progress, runtime versions
    lid.csv                           # step, train_lid, val_lid
    zeroth_shard_sequences_loss.csv   # Fixed training-sample loss
    val_shard_sequences_loss.csv      # Fixed validation-sample loss

```

The main LID results use the unquantized checkpoint files. Fixed-sample losses from the analyzer should be paired with LID when comparing the same evaluation sequences.

## References

Ruppik, B. M., von Rohrscheidt, J., van Niekerk, C., Heck, M., Vukovic, R., Feng, S., Lin, H.-c.,
Lubis, N., Rieck, B., Zibrowius, M., and Gasic, M. (2025). Less Is More: Local Intrinsic Dimen-
sions of Contextual Language Models. Advances in Neural Information Processing Systems. Ed.
by D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen. Vol. 38.
Curran Associates, Inc., pp. 67795–67828.
