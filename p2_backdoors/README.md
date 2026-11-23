# Usage

Run from `p2_backdoors` with Python 3.11 or newer:

```bash
uv sync
```

Prepare the inputs:

```text
models_dir/
  <model_id>.pt          # At least two task model checkpoints
clean_dir/
  labels.csv            # Columns: id,label; labels 0–19
  images/
    <id>.png            # 20 clean images per class (400 total)
```

Run with a **new output directory** for each run:

```bash
uv run python run.py \
  models_dir=/path/to/models_dir \
  clean_dir=/path/to/clean_dir \
  output_dir=/path/to/new_output_dir \
  device=cuda:0
```

The final scores are saved to `output_dir/scores.csv` with columns
`id,c00,...,c19`.
