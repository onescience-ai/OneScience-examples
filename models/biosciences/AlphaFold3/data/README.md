This model repository does not upload the AlphaFold3 dataset payload.

Download the required dataset separately:

```bash
modelscope download --dataset OneScience/AlphaFold3_dataset
```

Then link the dataset package data directory into this model package:

```bash
mkdir -p data
ln -s /path/to/AlphaFold3_dataset/data data/alphafold3_dataset
```
