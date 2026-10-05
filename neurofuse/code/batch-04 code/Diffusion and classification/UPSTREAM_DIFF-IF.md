# Upstream Diff-IF files in the received folder (not duplicated here)

The received folder `batch-04 code/Diffusion and classification/` was a git clone of **https://github.com/XunpengYi/Diff-IF** (MIT License) at commit **19030f5b7e8b2dffdcc5525742fb77a11ba83027** ("Add files via upload", Xunpeng Yi, 1 Jun 2024) with local changes. `git diff --ignore-cr-at-eol` against that commit shows the following:

- **New project files**, included in this folder verbatim apart from line endings: `app.py`, `convnext-b.py`, and the sample images `input/001`, `input/002` (images not included).
- **Changed from upstream**: `config/diff-if-mif_val.json` (included) and `model/diffif_modules/unet.py`. The only change to `unet.py` is one removed comment line; see the patch below.
- **Deleted locally**: `LICENSE`, `README.md`, `weights/put_the_weights_here.md`.
- **Identical to upstream except for Windows CRLF line endings**, so they are not duplicated here: `train.py`, `infer_ddim.py`, `targeted_search.py`, `requirements.txt`, `config/diff-if-ivf.json`, `config/diff-if-ivf_val.json`, `config/diff-if-mif.json`, `core/logger.py`, `core/metrics.py`, `data/__init__.py`, `data/ivf_dataset.py`, `data/mif_dataset.py`, `data/test_ivf_dataset.py`, `data/test_mif_dataset.py`, `data/util.py`, `model/__init__.py`, `model/base_model.py`, `model/model.py`, `model/networks.py`, `model/ddpm_modules/diffusion.py`, `model/ddpm_modules/unet.py`, `model/diffif_modules/diffusion.py`, `model/diffif_modules/finetune_arch.py`, `model/diffif_modules/loss.py`, `model/sr3_modules/diffusion.py`, `model/sr3_modules/unet.py`, `utils/Evaluator.py`, `utils/img_read_save.py`.

To rebuild the exact received tree, check out upstream commit 19030f5, apply the patch below, and add `app.py`, `convnext-b.py` and `config/diff-if-mif_val.json` from this folder.

The files here use LF line endings; the originals used CRLF.

## Exact diff vs upstream (ignoring CR at end of line)

```diff
diff --git a/config/diff-if-mif_val.json b/config/diff-if-mif_val.json
index 87e89a8..50eb46b 100644
--- a/config/diff-if-mif_val.json
+++ b/config/diff-if-mif_val.json
@@ -1,15 +1,15 @@
 {
     "name": "Diff-IF-mif-val",
-    "phase": "val", // train or val
+    "phase": "val",
     "gpu_ids": [
         0
     ],
-    "path": { //set the path
+    "path": { 
         "log": "logs",
         "tb_logger": "tb_logger",
         "results": "results",
         "checkpoint": "checkpoint",
-        "resume_state":  "weights/MIF_weights"
+        "resume_state":  "Diff-IF\\weights\\MIF_weights_gen.pth"
     },
     "datasets": {
         "train": {
@@ -24,8 +24,8 @@
         "val": {
             "dataset": "Test_mif",
             "name": "test",
-            "dataroot": "dataset/Harvard_FK/test",
-            "data_len": -1 // data length in validation
+            "dataroot": "Diff-IF\\input",
+            "data_len": -1
         }
     },
     "model": {
@@ -44,7 +44,7 @@
             "in_channel": 1,
             "out_channel": 1
         },
-        "beta_schedule": { // use munual beta_schedule for acceleration
+        "beta_schedule": {
             "train": {
                 "schedule": "linear",
                 "n_timestep": 1000,
diff --git a/model/diffif_modules/unet.py b/model/diffif_modules/unet.py
index 3d58e18..3dab906 100644
--- a/model/diffif_modules/unet.py
+++ b/model/diffif_modules/unet.py
@@ -14,7 +14,6 @@ def default(val, d):
         return val
     return d() if isfunction(d) else d
 
-# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
 class PositionalEncoding(nn.Module):
     def __init__(self, dim):
         super().__init__()
```
