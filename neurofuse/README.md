# NeuroFuse: project materials

This folder brings together the NeuroFuse material found in Preetam's Gmail accounts, the project files on his laptop and the project report. It was compiled on 5 Oct 2026. Every statement below comes from a source listed in [SOURCES.md](SOURCES.md). Anything the sources do not cover is marked **not found**.

## What the project is

**NeuroFuse: A Diffusion Based Image Fusion Framework for Tumor Classification** is the research paper that came out of the B.Tech major project *"Multi-Modal Medical Image Fusion for Brain Tumor Classification Using Deep Learning"*. The project was done in the Department of CSE (AIML & IoT), R & AI, VNR Vignana Jyothi Institute of Engineering & Technology, Hyderabad, and the report is dated May 2026.

Abstract, as submitted (ICCIET 2026 / CMT submission 884, 1 Apr 2026):

> Early tumor detection significantly improves patient's survival chances. Single-modalities like Magnetic Resonance Imaging (MRI), Computed Tomography (CT), Positron Emission Tomography (PET) and Single Photon Emission Computed Tomography (SPECT) provide important but limited information. To address the limitation of single modalities, this study proposes a system involving medical image fusion and classification. It is designed to combine complementary information from MRI and CT scans. A diffusion model is used to generate high-quality fused images that preserve MRI's soft-tissue contrast and CT's bone and density details. The diffusion-driven framework overcomes the challenges of ensuring proper feature preservation, reduced noise and improved anatomical visibility. The fusion framework combined with automated classification help clinicians improving decision-making accuracy. After testing with different classification networks, ConvNeXt-B (Transformer inspired CNN) and Vision Transformer (ViT) gave an accuracy of 0.9820 and 0.9888 respectively when trained with fused outputs which are higher compared to their accuracies when trained on CT and MRI. The proposed solution helps contributing to more effective and accessible brain tumor diagnosis systems.

Report keywords: MRI, CT, PET, SPECT, Medical Image Fusion, Diffusion, Classification, ConvNeXt-B, ViT.

## Architecture (summary)

This summary is taken from the project code (`code/batch-04 code/Diffusion and classification/`) and the 53-page project report (the PDF is not in this repo; see [resources/ATTACHMENTS.md](resources/ATTACHMENTS.md)):

1. **Input:** a patient's MRI and CT scan. In the app these live under `input/<patient_id>/CT.jpg` and `MRI.jpg`.
2. **Pre-processing:** the images are converted to YCbCr and the luminance (Y) channel is used. Images are cropped to a multiple of 8 and normalised to [-1, 1] (`preprocess_ddim` in `app.py`).
3. **Diffusion fusion:** a conditional diffusion model adapted from **Diff-IF** (Yi et al., *Information Fusion*, 2024; upstream repo `XunpengYi/Diff-IF`) produces the fused luminance image. Sampling uses DDIM (`load_ddim_model` in `app.py`; upstream `infer_ddim.py` and `model/diffif_modules/`, see `UPSTREAM_DIFF-IF.md`).
4. **Colour merge:** the fused Y channel is recombined with the reference chroma.
5. **Classification:** a fine-tuned **ConvNeXt-B** (`convnext-b.py`) labels the fused image *Tumor* / *Non-tumor*.
6. **UI:** a Streamlit app (`app.py`, page title "NeuroFuse") runs fusion and classification on the selected patient.

According to the report, paired MRI/CT data was not available, so a CycleGAN translated MRI into synthetic CT for training. This point also appears in the ICCIET editorial review comments ([ICCMA.md](ICCMA.md)). The CycleGAN code was **not found** in any source.

See [STRUCTURE.md](STRUCTURE.md) for the file-level structure.

## Team

Authors are listed exactly as they appear in the ICCCMLA / IEEE eCopyright emails, in author order:

Palliyana Shabarish, Kancharagunta Kishan Babu, Sagar Yeruva, Nakka Smitha, Kore Abhijith, Balini Lohith Reddy, Preetam Pujari.

The project report lists the student team as Balini Lohith Reddy, Kore Abhijith, Nakka Smitha, Palliyana Shabarish and Preetam Pujari, with Mr. K. Kishan Babu, Assistant Professor, as guide.

**British teammate:** no email from a British teammate was found in either Gmail account. The only teammate email that contained code came from **Shabarish Palliyana** on 30 Jan 2026 (see [code/README.md](code/README.md)), and nothing in that email indicates nationality.

## Publication / submission status

See [ICCMA.md](ICCMA.md). In short, the paper was accepted at the **2026 IEEE 8th International Conference on Cybernetics, Cognition and Machine Learning Applications (ICCCMLA)**, paper #778636, 5–6 Oct 2026, online. The IEEE copyright transfer was completed on 12 Sep 2026. No conference literally named "ICCMA" was found.

## Contents

| Path | What it is |
|---|---|
| `README.md` | This overview |
| `STRUCTURE.md` | Project / code structure |
| `ICCMA.md` | Publishing and submission notes, quoted from the emails |
| `SOURCES.md` | One line per item: source, date, what it is |
| `code/` | Code exactly as received, with provenance in `code/README.md` |
| `resources/` | Links, and a list of attachments and binaries that are not in this repo |

## Sources checked

Gmail pritam.exe2k4@gmail.com (no hits), Gmail preetam.naik3@gmail.com (all hits), Google Drive (no hits), Notion (no hits), the project files from the laptop, and the major-project repo (empty before this commit). Details are in [SOURCES.md](SOURCES.md).
