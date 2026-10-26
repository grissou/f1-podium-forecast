# Data

The raw CSVs are not committed. Download the Kaggle dataset
[rohanrao/formula-1-world-championship-1950-2020](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
and put the CSV files in `data/raw/`:

```python
import kagglehub
path = kagglehub.dataset_download("rohanrao/formula-1-world-championship-1950-2020")
```

then copy `*.csv` from `path` into `data/raw/`.
