# Darvin Subnet 323 (`hand_rec`) — 2-Page PPT Summary

---

## Page 1 — Architecture, Data Firewall & Training Design

### Goal & Key Gaps vs. Upstream `phanthymotus`
- **Task & Limits**: 30 static gestures (`23` single-hand + `7` two-hand) + anatomical L/R hand boxes on **Jetson Orin-8G** (`JP5.1.1` / `JP6.1`), strictly **`< 30 MB`** total models and **`< 10%` GPU**.
- **Upstream Blockers**: `55.2 MB` models (over limit), 5 geometric rules (`0.00%` Macro-F1), low forearm-ROI IoU (`0.39`), and no handling of the **`92` visible resting `null` hands** (`11%` of GT boxes) or `xsign` crossed forearms.

### Zero-Leakage Data Strategy
- **Strict Split Firewall**: Official 600-image Feishu dataset is too small for training $\rightarrow$ reserved **strictly** as `val` (`300`, threshold check only) + **locked `test` (`300`, evaluated once)**.
- **100% Open-Source Training**: HaGRID v2 512px RGB + SberCloud 21-landmark annotations (`18,939` train / `1,659` val images, `27,517` hand boxes, `24,322` with 21 kpts, `3,169` resting `no_gesture` hands; **`0` official images in training**).

### 4-Stage `< 30 MB` All-FP32 Pipeline & Progressive Training
| Stage | Model & Size (FP32 ONNX) | Role & Training Design | Open-Source Val |
|---|---|---|---|
| **1. Person Pose** | `YOLO11n-pose` (`11.81 MB`) | COCO-17 face/elbow/wrist anchors (`640×640`, pretrained) | — |
| **2. Hand Det + L/R** | `YOLO11n-hand` (`10.56 MB`) + `0 MB` NumPy Hungarian | Detects *all* visible hands (incl. resting); 20 ep on HaGRID v2; forearm extrapolation + chirality solves `xsign` & touching hands | `mAP50 = 0.995` (`IoU = 0.845` on `val`) |
| **3. Multi-Task Crop** | `MobileNetV3-Small` + FPN + Axis-Marginalized SimCC-21 (`5.80 MB`) | 24 ep on `33,771` crops (`224×224`, union crops + 12% hard negatives); cuts SimCC params `5.3×` to stay native FP32 | `acc = 98.56%`, `kpt_err = 0.029` |
| **4. Dual-Hand Fusion** | Residual MLP (`hidden=144`, `1.48 MB`) | 35 ep on cached Stage 1–3 `(vis + 118-d hand geo + 30-d pair geo)`; 31-way per-hand + 8-way pair head + resting-hand guard | `hand_acc = 99.59%` |
| **Total Bundle** | **`29.66 MB`** (`29,663,786` B `< 30 MB`) | **100% native FP32** (avoids `>10×` FP16 CPU slowdown on ARM) | ORT diff `≤ 2.86e-5` |

---

## Page 2 — Results, Jetson Hardening & Delivery Status

### Benchmark Performance (`CPUExecutionProvider`, `0%` GPU)
| System / Official Split | Macro-F1 (30 cls) | Precision | Recall | Miss Rate | Hand BBox IoU |
|---|---:|---:|---:|---:|---:|
| Upstream Baseline (`val` / `test`, 300 ea.) | `0.00%` / `0.00%` | `0.00%` | `0.00%` | `100.0%` | `0.4018` / `0.3936` |
| **Our 4-ONNX Pipeline (`val`, 300)** | **`99.32%`** | **`99.70%`** | **`99.00%`** | **`1.00%`** | **`0.8476`** |
| **Our 4-ONNX Pipeline (`test`, 300 — locked)** | **`98.63%`** | **`99.09%`** | **`98.33%`** | **`1.67%`** | **`0.8526`** |

- **Highlights on Locked `test`**: `23 / 30` classes at **`1.0000` F1**; **all 7 two-hand classes at `100.0%` F1 (`140/140`)**; only **`3` total FPs** and **`5` total FNs** across `370` positive + `52` resting hands.

### Jetson Orin-8G Deployment Hardening & Verification
- **Multi-Worker Stability (10 Perception + 10 Evaluator Containers)**:
  - Runs strictly on **`CPUExecutionProvider`** (`0%` GPU, zero CUDA context RAM, avoids JP5.1.1 `Provider_SetHost` SIGSEGV with `sherpa-onnx`).
  - Unused plugins (`asr`, `tts`, `vop`, `ocr`, `visual_depth`, `face_recognition`) disabled; `OMP_NUM_THREADS=1`, `MALLOC_ARENA_MAX=2`, ORT arenas disabled; Python `3.8` (`JP5.1.1`) & `3.10` (`JP6.1`) verified.
- **JP5 Build Fix (`log-evaluation-9cffbc0b.zip`)**: Fixed empty Docker `ARG HAND_SUBNET323_BASE_URL=""` shadowing the default URL, published `< 30 MB` ONNX assets to GitHub Release, and passed **`235` tests** + live empty-`ARG` download verification.

### Current Status & Next Step
- **Published on GitHub**: `kiddddd1218/phanthymotus` branch `darvin323-hand-rec-20261125` @ `fa83f11a4ded399c711bf2af934f6fda10c7154c` + Release `darvin323-models-v1` + `configs/Values.yaml`.
- **Next**: Run official Darvin Subnet 323 JP5/JP6 leaderboard evaluation, then mirror weights to ModelScope/COS before upstream PR.
