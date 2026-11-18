"""Score files: results/<detector>.csv with columns id,c00,...,c19, one row per model."""
import csv
import math
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


def check_submission(path, test_ids):
    """Crash unless the file passes the portal's checks: a rejected file still uses up a daily submission.
    Reads the file as text, so it checks exactly what would be uploaded."""
    with open(path) as f:
        reader = csv.reader(f)
        assert next(reader) == ["id"] + CLASS_COLUMNS, "wrong header"
        ids = []
        for row in reader:
            values = [float(value) for value in row[1:]]
            assert len(values) == N_CLASSES, f"{row[0]}: {len(values)} values instead of {N_CLASSES}"
            assert all(math.isfinite(value) for value in values), f"{row[0]}: nan or inf"
            ids.append(row[0])
    assert len(ids) == len(set(ids)), "an id is repeated"
    assert set(ids) == set(test_ids), f"ids differ from the test models: {len(set(test_ids) - set(ids))} missing, " \
                                      f"{len(set(ids) - set(test_ids))} unknown"
