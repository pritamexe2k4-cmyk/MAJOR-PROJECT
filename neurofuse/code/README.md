# Code: provenance

The code files included here are exact copies of what was received (line endings converted to LF). No secrets (API keys, tokens or passwords) were found in any file, so nothing was replaced with a placeholder.

## `batch-04 code/Diffusion and classification/`
- **Source:** the "Major Project" folder from Preetam's laptop Desktop, uploaded to the shared workspace on 1–3 Oct 2026. The original files are `batch04_code_only.zip` and its extracted folder `ex/batch-04 code/`; the zip timestamps are 2026-10-01 21:11.
- **What it is:** the NeuroFuse implementation. It contains the Streamlit app `app.py` (page title "NeuroFuse"), the ConvNeXt-B classifier `convnext-b.py`, and diffusion fusion code adapted from Diff-IF (https://github.com/XunpengYi/Diff-IF, MIT License). The received folder was a git clone of that repo with local modifications.
- **Included, verbatim apart from CRLF→LF line endings:** the project-specific files `app.py`, `convnext-b.py` and `config/diff-if-mif_val.json`, under their original names and paths.
- **Not duplicated:** the other 29 source files are byte-identical to upstream Diff-IF commit 19030f5 apart from CRLF line endings. The one other changed file, `model/diffif_modules/unet.py`, differs only by a removed comment line. See [`UPSTREAM_DIFF-IF.md`](batch-04%20code/Diffusion%20and%20classification/UPSTREAM_DIFF-IF.md) for the full list and the exact patch. Images, `__pycache__`, `.git` and an empty placeholder file are also left out. Model weights were not in the folder.

## `email_2026-01-30_Shabarish_Palliyana/code&Dataset.docx.txt`
- **Source:** email "Code and Database" from Shabarish Palliyana <palliyanashabarish004@gmail.com> to preetam.naik3@gmail.com, sent 30 Jan 2026 11:27 IST, Gmail thread 19c0d7a37329b6c2. The email has no body text. Preetam forwarded it on 1 Feb 2026.
- **Original attachment:** `code&Dataset.docx` (13.4 KB Word file). This file is the plain-text content of that document, extracted verbatim. The `.docx` itself is not in this repo because it is binary.
- **What it is:** a dataset folder layout (Training/Validation × Healthy/Tumor), a Google Drive dataset link, and a TensorFlow/Keras ResNet50 transfer-learning training script (256×256 input, Adam 1e-4, EarlyStopping patience 5, ModelCheckpoint `best_tumor_model.h5`, loss/accuracy plots). It ends with alternative backbones to try: MobileNetV2, DenseNet121 and EfficientNetB0.
- This email does not mention "NeuroFuse" by name. It is included because it is the only teammate email with code and it covers tumor-classification code and data for the project.
