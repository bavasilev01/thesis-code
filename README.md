## Download dataset
- Run from data/mimic-cxr directory
```bash
head -n 10000 IMAGE_FILENAMES | wget -r -N -c -np -nH --cut-dirs=1 --user bvasilyev --ask-password -i - --base=https://physionet.org/files/mimic-cxr-jpg/2.1.0/
```

## Download the pretrained model
```bash
python download_hf.py
```

## Create the dataset (PA images only)
```bash
python mimic_dataset.py
```

## Test LLaVa dataset creation
```bash
python llava_dataset.py
```

## Run the fine-tuning
```bash
python training.py
```
