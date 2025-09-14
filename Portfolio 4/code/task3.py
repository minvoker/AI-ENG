"""Portfolio Assessment 4 – Task 3

This script iterates over a folder of LabelMe JSON files and reassigns
labels for log objects that are smaller than a defined size threshold.  The
new label used is `detected_log`.  All other objects retain their
original labels.  The updated annotation files are written to a
designated output directory.

To use this script, place your images and JSON files in a directory
referred to as `ORIGINAL_DIR` below.  The script copies the image
files unchanged into `UPDATED_DIR` and writes the modified JSON
annotations alongside them.

Adjust the `SIZE_THRESHOLD` to control what counts as a "broken" log.
"""

import json
import os
import shutil
from typing import Tuple

# ---------------------------------------------------------------------------
# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_DIR = os.path.normpath(os.path.join(BASE_DIR, os.pardir, 'extracted-log-labelled'))
UPDATED_DIR = os.path.normpath(os.path.join(BASE_DIR, os.pardir, 'extracted-converted-log-labelled'))

# Any log whose bounding box width or height falls below this value will be
# relabelled as `detected_log`.  Adjust based on your dataset.
SIZE_THRESHOLD = 100


def ensure_output_directories() -> None:
    """Create the destination directory if it doesn't already exist."""
    os.makedirs(UPDATED_DIR, exist_ok=True)


def relabel_logs(json_path: str, output_path: str, size_threshold: int) -> None:
    with open(json_path, 'r') as f:
        data = json.load(f)

    if 'shapes' not in data:
        print(f"Skipping {json_path} because it lacks a 'shapes' field.")
        return

    for shape in data['shapes']:
        label = shape.get('label', '')
        if 'log' in label:
            xs = [p[0] for p in shape['points']]
            ys = [p[1] for p in shape['points']]
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            if width < size_threshold or height < size_threshold:
                shape['label'] = 'detected_log'

    # Write the modified annotation
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def process_all_files() -> None:
    ensure_output_directories()
    for fname in os.listdir(ORIGINAL_DIR):
        src_path = os.path.join(ORIGINAL_DIR, fname)
        dst_path = os.path.join(UPDATED_DIR, fname)
        if fname.lower().endswith(('.json')):
            print(f"Processing annotations for {fname}…")
            relabel_logs(src_path, dst_path, SIZE_THRESHOLD)
        elif fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Copy image files unchanged
            shutil.copy(src_path, dst_path)


def main() -> None:
    process_all_files()
    print(f"Finished processing.  Updated files are in {UPDATED_DIR}")


if __name__ == '__main__':
    main()