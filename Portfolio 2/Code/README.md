# Portfolio 2 ML Pipeline

The goal is to develop a classification model that can
distinguish between two meat processing activities (boning and slicing)
using acceleration data from body worn sensors.

## Requirements

This project uses only common Python packages: `pandas`, `numpy` and
`scikit‑learn`. These are available in most standard Anaconda or
virtualenv installations. If you are running this outside the supplied
environment you can install the dependencies via:

```bash
pip install pandas numpy scikit‑learn
```

## Usage

1. **Obtain the dataset.** Download `Boning.csv` and `Slicing.csv` from
   Canvas or the provided GitHub repository. Save them somewhere on
   your filesystem.

2. **Choose sensors.** Select a digit from 1-9 to determine which two body positions to use.
   Refer to the table below when supplying the `--student_digit`
   argument:

   | Student digit | Sensor set 1      | Sensor set 2      |
   |--------------:|:------------------|:------------------|
   | 0             | Neck             | Head             |
   | 1             | Right Shoulder   | Left Shoulder    |
   | 2             | Right Upper Arm  | Left Upper Arm   |
   | 3             | Right Forearm    | Left Forearm     |
   | 4             | Right Hand       | Left Hand        |
   | 5             | Right Upper Leg  | Left Upper Leg   |
   | 6             | Right Lower Leg  | Left Lower Leg   |
   | 7             | Right Foot       | Left Foot        |
   | 8             | Right Toe        | Left Toe         |
   | 9             | L5               | T12              |

3. **Run the pipeline.** From the `Code` directory, execute the script
   with the appropriate arguments. For example:

   ```bash
   python main.py \
     --boning_csv /path/to/Boning.csv \
     --slicing_csv /path/to/Slicing.csv \
     --student_digit 0 \
     --output_dir ../outputs
   ```

   The script will:

   - Load the two CSV files and extract only the relevant columns for
     your chosen sensors.
   - Create composite signals (RMS, roll and pitch) for each sensor.
   - Merge the boning and slicing data sets and add a class label
     (0 for boning, 1 for slicing).
   - Aggregate all 18 dynamic columns (six raw and twelve composite)
     into non‑overlapping 60‑frame windows. For each window it
     computes six statistics: mean, standard deviation, minimum,
     maximum, area under the curve and number of peaks.
   - Train several machine‑learning models and output a results table
     summarising test and cross‑validation accuracies.

4. **Inspect the outputs.** The aggregated feature table is saved as
   `features.csv` in the output directory and the summary of model
   performance is saved as `results.csv`.

## Outputs

Two artifacts are produced:

* `features.csv` – minute‑level features (108 columns) and the class
  label. This file is useful if you wish to conduct your own
  experiments or visualisations.
* `results.csv` – a concise table reporting both the 70/30 train–test
  accuracy and the mean 10‑fold cross‑validation accuracy for each
  classifier. The script prints this table to the console as well.

## Notes

* The hyperparameter grids used for tuning are intentionally modest to
  ensure reasonable run times. Feel free to expand the grid if you
  have the computational resources.
* The number of peaks is computed using a simple local maxima count.
  Alternative definitions (e.g. thresholded peaks) can be adopted if
  desired.
