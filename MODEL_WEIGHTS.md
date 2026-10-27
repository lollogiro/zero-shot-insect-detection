# Model Weights Download

Most weights need no manual action; only two groups are downloaded by hand.
The `models-cache/` directory is gitignored.

Quickest option: `bash scripts/download_weights.sh` downloads everything below
(skip files that are already present; `--yolo` and `--binary` restrict to one
group). The rest of this file documents the individual commands.

## HuggingFace foundation models (no action needed)

GroundingDINO, SAM3, OWL-ViT and DINOv3 are **not** stored in the repo: they
are downloaded at execution time through the HuggingFace Hub API (the
`from_pretrained` calls in `utils/models.py`) on the first inference run.
Files are cached in the HuggingFace cache (`~/.cache/huggingface/hub`) unless
a `cache_dir` is passed explicitly (e.g. `SAM3Detector(cache_dir="./models-cache")`).
Nothing to download manually. SAM3 and DINOv3 additionally require `HF_TOKEN`
in `.env`; GroundingDINO and OWL-ViT work without it.

## YOLO baselines

Download the file for the detector you want and place it in `models-cache/`
at the repo root (create the directory with `mkdir -p models-cache` if it
does not exist). Keep the exact target file names: they are the defaults
used in `utils/models.py`. Several downloads ship under a different name
(inside a zip, or as `best.pt`), so the rename steps below matter.

| Detector | File expected in `models-cache/` | Source |
|---|---|---|
| Stark2023 | `yolov5_s_best_stark2023.pt` | Direct (GitHub raw) |
| Bjerge2023 | `insect1201-bestF1-1280v5m6_bjerge2023.pt` | Zip (Zenodo) |
| Marin2025 | `marin2025.pt` | Google Drive folder |
| FlatBug | `flat_bug_M.pt` | Direct (erda) |
| InsectsDCT | `insectsDCT_insects6ColorExtend.pt` | Direct (GitHub raw) |
| Arthropod (default) | `yolo11l_ArthroNat+flatbug.pt` | HF Hub (edgaremy/arthropod-detector) |
| Arthropod (variant) | `yolo11l_ArthroNat_mosaic33.pt` | HF Hub (edgaremy/arthropod-detector) |

### Stark2023 (YOLOv5-s)

```bash
wget -O models-cache/yolov5_s_best_stark2023.pt \
  https://raw.githubusercontent.com/stark-t/PAI/main/detectors/trained_weights/yolov5_s_best.pt
```

### Bjerge2023 (zip on Zenodo)

Download and extract the zip, then rename the file (the zip ships without the
`_bjerge2023` suffix). The zip also contains smaller variants
(`insect1201-bestF1-1280v5s6.pt`, `insect1201-bestF1-640v5m.pt`): the repo
uses the `1280v5m6` one.

```bash
wget -O /tmp/YOLOv5models.zip \
  "https://zenodo.org/records/7395752/files/YOLOv5models.zip?download=1"
unzip -o /tmp/YOLOv5models.zip -d models-cache/
mv models-cache/YOLOv5models/insect1201-bestF1-1280v5m6.pt \
  models-cache/insect1201-bestF1-1280v5m6_bjerge2023.pt
```

### Marin2025 (Google Drive folder)

Google Drive cannot be downloaded with `wget`; use `gdown` (a project
dependency, installed by `uv sync` / `pip install -r requirements.txt`) or
download manually from the browser:

```bash
gdown --folder https://drive.google.com/drive/folders/1X3buOu6bhYmx0hesPwmcU_FB693SCfHn
# if the downloaded file is not named marin2025.pt, rename it:
mv <file>.pt models-cache/marin2025.pt
```

### FlatBug

```bash
wget -O models-cache/flat_bug_M.pt \
  https://anon.erda.au.dk/share_redirect/Bb0CR1FHG6/models/flat_bug_M.pt
```

### InsectsDCT (GitHub, ships as `best.pt`)

```bash
wget -O models-cache/insectsDCT_insects8Color.pt \
  https://github.com/kimbjerge/insectDCT/blob/main/runs/detect/insects8Color/weights/best.pt
```

### Arthropod (HF Hub)

Two training-data variants of the same single-class YOLO11-large checkpoint.
The default `yolo11l_ArthroNat+flatbug.pt` generalises best out-of-domain and
on small objects; `yolo11l_ArthroNat_mosaic33.pt` is the alternative. Note
the model card still documents older file names (`yolo11l_arthropod_0.413.pt`
etc.) that 404 on the Hub; the two files below are the ones actually
published. `utils/models.py` (ArthropodDetector) defaults to the `+flatbug`
one, and `examples/arthropod_detector.py` lets you pick either via
`ARTHROPOD_VARIANT`.

```bash
wget -O models-cache/yolo11l_ArthroNat+flatbug.pt \
  https://huggingface.co/edgaremy/arthropod-detector/resolve/main/yolo11l_ArthroNat+flatbug.pt
wget -O models-cache/yolo11l_ArthroNat_mosaic33.pt \
  https://huggingface.co/edgaremy/arthropod-detector/resolve/main/yolo11l_ArthroNat_mosaic33.pt
```

### AutoPollS

Download the model weights for the AutoPollS detector (YOLOv11l) from the Google Drive link provided in the AutoPollS GitHub repository [here](https://github.com/Crall-Lab/AutoPollSAnalysis). The model can be downloaded from [here](https://drive.google.com/file/d/1xsLxBCJhnFi8wejTF1V851t4TskNgAqn/view). From this ZIP folder, the file `AutoPollSAnalysis_models_tmp/AutoPollSAnalysis_models_tmp/AutoPollS_YOLOv11l_800_V0/weights/best.pt` should be renamed to `AutoPollS_YOLOv11l_800_V0.pt` and placed in the `models-cache/` directory.

## Binary classifier weights

The trained insect/background classifiers (SVM/MLP on BioCLIP-2.5/DINOv3
embeddings) are not downloaded automatically. Download them once and place
them in `models-cache/binary_classificator/` at the repo root (create the
directory with `mkdir -p models-cache/binary_classificator` if it does not
exist). Google Drive cannot be downloaded with `wget`; use `gdown` (a
project dependency, installed by `uv sync` / `pip install -r requirements.txt`)
or download manually from the browser:

```bash
gdown --folder -O models-cache/binary_classificator \
  "https://drive.google.com/drive/folders/1Q_SnMAfdvSJaJ1MSYEjf8f0bLwHWLLFd"
```

`gdown --folder -O <dir>` puts every file of the folder directly into `<dir>`
keeping its original name, so the target file names below are kept
automatically.

| File | Classifier |
|---|---|
| `insect_classifier_ecostack_svm_bioclip2_5.joblib` | SVM on BioCLIP-2.5 embeddings |
| `insect_classifier_ecostack_svm_dinov3.joblib` | SVM on DINOv3 embeddings |
| `insect_classifier_ecostack_mlp_bioclip2_5.pth` | MLP on BioCLIP-2.5 embeddings |
| `insect_classifier_ecostack_mlp_dinov3.pth` | MLP on DINOv3 embeddings |

Keep the exact file names: they are the defaults expected by
`load_binary_classifier()` in `utils/inference.py`. Alternatively, train the
classifiers yourself with `binary_classificator/train_svm.ipynb` and
`binary_classificator/train_mlp.ipynb`; they produce the same files.

## Verify

YOLO weight files are loaded when the detector is instantiated, so a wrong
file name or a truncated download fails with an error at that point.
`has_binary_classifier(mode, name)` reports whether a classifier file is
present (without loading), and loading a missing or corrupt file raises a
helpful error from `load_binary_classifier(mode, name)`. Every link above was
checked and the downloaded files match the weights produced by the training
notebooks.
