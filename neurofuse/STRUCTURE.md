# Project structure

## Code from the laptop: `code/batch-04 code/Diffusion and classification/`

This is the layout of the folder as received. Only `app.py`, `convnext-b.py`, `config/diff-if-mif_val.json` and `UPSTREAM_DIFF-IF.md` are stored in this repo. The other files are unchanged upstream Diff-IF files; `UPSTREAM_DIFF-IF.md` explains how to rebuild the exact tree.

```
Diffusion and classification/
├── app.py                 # NeuroFuse Streamlit app: CT+MRI fusion + tumour classification
├── convnext-b.py          # ConvNeXt-B classifier training script
├── infer_ddim.py          # Batch fusion inference with DDIM sampling (Diff-IF)
├── train.py               # Diffusion-model training (Diff-IF)
├── targeted_search.py     # Fusion-knowledge-prior targeted search (Diff-IF)
├── requirements.txt
├── config/                # diff-if-ivf.json, diff-if-ivf_val.json, diff-if-mif.json, diff-if-mif_val.json
├── core/                  # logger.py, metrics.py
├── data/                  # __init__.py, ivf_dataset.py, mif_dataset.py, test_ivf_dataset.py, test_mif_dataset.py, util.py
├── model/                 # __init__.py, base_model.py, model.py, networks.py
│   ├── ddpm_modules/      # diffusion.py, unet.py
│   ├── diffif_modules/    # diffusion.py, finetune_arch.py, loss.py, unet.py
│   └── sr3_modules/       # diffusion.py, unet.py
└── utils/                 # Evaluator.py, img_read_save.py
```

The received folder also had a `.git` directory whose remote is `https://github.com/XunpengYi/Diff-IF.git`, which means the diffusion code is a modified clone of Diff-IF.

Not included, because they are binary or empty:
- images `assert/diffusion.png`, `input/001|002/{CT,MRI}.jpg`, `test_data/{Infrared,Visible}/00004N.png`
- the figures one level up (`batch-04 code/base_archit.png`, `fused_grid.png`, `pairs.png`, `UI.png`)
- `__pycache__/*.pyc`, `.git/`
- the empty `dataset/put_the_dataset_here.md`

Model weights were not part of the received folder and were **not found**.

## Code from email: `code/email_2026-01-30_Shabarish_Palliyana/`

`code&Dataset.docx.txt` is the full text of the Word attachment `code&Dataset.docx`. It contains a dataset layout, a Google Drive dataset link, and a TensorFlow/Keras ResNet50 transfer-learning script for Healthy/Tumor classification, with alternative backbones (MobileNetV2, DenseNet121, EfficientNetB0).

Dataset directory structure, as given in that email attachment:

```
Dataset/
---Training/
----Healthy
----Tumor
---Validation/
----Healthy
----Tumor
```

## Report structure

The 53-page project report PDF (3.2 MB) is not in this repo because it is binary; see `resources/ATTACHMENTS.md`. Its front matter, in order, is: Certificate, Plagiarism Certificate, Approval Certificate, Declaration, Abstract, followed by the chapters.
