"""Score files: results/<detector>.csv with columns id,c00,...,c19, one row per model."""
import csv
import os

from model import N_CLASSES

CLASS_COLUMNS = [f"c{c:02d}" for c in range(N_CLASSES)]


def read_scores(path):
    """{model_id: N_CLASSES floats} from a score file."""
    with open(path) as f:
        return {row["id"]: [float(row[column]) for column in CLASS_COLUMNS] for row in csv.DictReader(f)}


def write_scores(path, scores):
    # Write a temporary file and rename it: stopping a run mid-write then leaves the old file, never half a file.
    with open(path + ".tmp", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id"] + CLASS_COLUMNS)
        for model_id, row in scores.items():
            # csv writes floats with str(), which reads back as exactly the same number: no rounding ties.
            writer.writerow([model_id] + row)
    os.replace(path + ".tmp", path)
