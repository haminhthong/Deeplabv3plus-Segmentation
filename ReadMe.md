# DeepLabV3+ ResNet50 Semantic Segmentation System

> **Hệ thống phân đoạn ảnh chuẩn hóa từ Dataset Contract, Training Lifecycle, Parity Serving đến Locked Holdout Benchmark trên Pascal VOC 2012.**

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Tầm nhìn & Nguyên lý Kỹ thuật (Engineering Vision)

Dự án này **không nhằm mục đích** chạy đua thêm nhiều kiến trúc (U-Net, FPN, SegFormer, v.v.) hay dựng hạ tầng phân tán quá sớm (Redis, Triton cluster). Thay vào đó, mục tiêu duy nhất là:

> **Xây dựng một hệ thống Semantic Segmentation đúng chuẩn mực ML Engineering, có khả năng kiểm soát dữ liệu chặt chẽ, tái lập 100%, phục vụ suy luận chuẩn xác tại độ phân giải gốc của ảnh và sẵn sàng triển khai thực tế.**

### 5 Nguyên Tắc Cốt Lõi Được Chuẩn Hóa
1. **Một Dữ Liệu - Một Split Contract Duy Nhất (Zero Fallback)**: Xóa bỏ hoàn toàn cơ chế fallback 100 ID mẫu. Toàn bộ ảnh/mặt nạ được xác thực SHA-256 thực tế; phân chia official train thành `dev_train` (85%) và `dev_val` (15%) bằng **multilabel stratification**.
2. **Khóa Chặt Tập Kiểm Thử (Locked Holdout Parity)**: Toàn bộ official VOC `val` (1,449 ảnh) được bảo vệ làm **Locked Holdout** và chỉ được đánh giá **đúng một lần duy nhất** sau khi chốt mô hình cuối. Tuyệt đối không duyệt hay demo tập holdout trên UI.
3. **Artifact Contract 3 Tầng**: Phân định rõ ràng giữa `checkpoints/last.ckpt` (lưu toàn bộ trạng thái RNG, Optimizer, Scheduler, AMP Scaler để resume chính xác), `checkpoints/best.ckpt` (mô hình phát triển tốt nhất trên `dev_val`), và `checkpoints/final_model.pth` (mô hình triển khai tinh gọn, không mang gánh nặng optimizer).
4. **Tính Nhất Quán Giữa Huấn Luyện & Phục Vụ (Training / Serving Parity)**: Đóng gói duy nhất một `Predictor` canonical:
   $$\text{Original Image} \rightarrow \text{Letterbox } 320 \times 320 \rightarrow \text{Model} \rightarrow \text{Crop Padding} \rightarrow \text{Resize Logits to } (H_{\text{orig}}, W_{\text{orig}}) \rightarrow \text{Softmax / Argmax}$$
   Cả Holdout Evaluator, FastAPI và Streamlit đều sử dụng chung một implementation này.
5. **Hệ Thống Thang Đo Chuẩn Xác (Metrics Hierarchy)**: Loại bỏ các phép đo vùng dựa trên tổng pixel class; thay thế Boundary F1 cố định bằng **dung sai thích ứng theo kích thước ảnh gốc** ($r \propto \text{diagonal}$).

---

## 2. Kiến Trúc Toàn Bộ Vòng Đời Hệ Thống (End-to-End ML Lifecycle)

```text
                      PASCAL VOC 2012
                             │
                             ▼
                    ┌─────────────────┐
                    │  DATA INGESTION │
                    └────────┬────────┘
                             │
                             ▼
                    Dataset Integrity Audit
                    ├─ image exists & mask exists
                    ├─ matching dimensions
                    ├─ valid class IDs 0..20 / 255
                    ├─ duplicate ID detection
                    └─ exact duplicate SHA-256 check
                             │
                             ▼
                       DATA MANIFEST
                             │
                             ▼
              ┌─────────────────────────────┐
              │ OFFICIAL VOC TRAIN DATA     │
              └─────────────┬───────────────┘
                            │
                  multilabel stratification
                            │
                   ┌────────┴────────┐
                   ▼                 ▼
             DEV TRAIN          DEV VALIDATION
                   │                 │
                   │                 └─────► model selection
                   │                         hyperparameters
                   │                         best epoch
                   ▼
                TRAIN
                   │
                   ▼
          DeepLabV3+ ResNet50
                   │
                   ▼
             best development config
                   │
                   ▼
              FREEZE CONFIG
                   │
                   ▼
        FINAL FIT ON FULL OFFICIAL TRAIN
                   │
                   ▼
              FINAL MODEL (final_model.pth)
                   │
                   ▼
          OFFICIAL VOC VAL HOLDOUT
           evaluated exactly once
                   │
                   ▼
              FINAL REPORT
                   │
                   ▼
              MODEL RELEASE
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
      FastAPI             Streamlit
         │                   │
         └────── Predictor ───┘
                   │
                   ▼
       Original-Resolution Mask
```

---

## 3. Cấu Trúc Mã Nguồn Chuẩn Hóa

```text
Deeplabv3plus-Segmentation/
│
├── configs/
│   └── deeplabv3plus_resnet50_320.yaml       # Canonical baseline configuration
│
├── src/
│   └── vocseg/
│       ├── __init__.py
│       ├── config.py                         # Typed config & YAML loader
│       ├── constants.py                      # 21 VOC classes, colormap, stats
│       ├── schemas.py                        # Pydantic schemas (artifacts & API)
│       │
│       ├── data/
│       │   ├── audit.py                      # Real SHA-256 audit & stats calculation
│       │   ├── dataset.py                    # VOC segmentation dataset (manifest-based)
│       │   ├── splits.py                     # Multilabel stratified splitting
│       │   └── transforms.py                 # Joint training augmentation & letterbox
│       │
│       ├── models/
│       │   └── deeplabv3plus.py              # Canonical DeepLabV3+ ResNet50 & validator
│       │
│       ├── training/
│       │   ├── losses.py                     # CombinedLoss: CE (ignore 255) + 0.5 * Dice
│       │   ├── reproducibility.py            # Complete RNG state capture/restore
│       │   ├── checkpoint.py                 # last.ckpt, best.ckpt, final_model.pth
│       │   └── trainer.py                    # Training engine (AMP, CosineAnnealing)
│       │
│       ├── evaluation/
│       │   ├── metrics.py                    # mIoU, Dice, PixelAcc, adaptive Boundary F1
│       │   ├── development.py                # Fast letterbox evaluation during epochs
│       │   └── holdout.py                    # Locked original-resolution evaluator
│       │
│       ├── inference/
│       │   ├── predictor.py                  # Canonical end-to-end original-resolution Predictor
│       │   └── visualization.py              # Colormap overlay, entropy uncertainty map
│       │
│       └── api/
│           └── app.py                        # Lightweight FastAPI service (/health, /segment)
│
├── scripts/
│   ├── prepare_data.py                       # Audit & stratified dev/holdout split generation
│   ├── train.py                              # Development training (DEV TRAIN -> DEV VAL)
│   ├── final_fit.py                          # Final fit on full official train -> final_model.pth
│   ├── evaluate_holdout.py                   # Locked holdout evaluation (single run)
│   └── predict.py                            # CLI inference using canonical Predictor
│
├── demo/
│   └── streamlit_app.py                      # Interactive demo UI (pure image upload & inspect)
│
├── experiments/
│   └── architectures.py                      # Archived U-Net & FPN for ablation research
│
└── tests/
    ├── unit/                                 # Unit tests for transforms, splits, metrics, checkpoint
    ├── integration/                          # Synthetic mini VOC end-to-end lifecycle test
    └── api/                                  # FastAPI endpoint contract tests
```

---

## 4. Hướng Dẫn Vận Hành Hệ Thống (Workflow & Commands)

### Bước 1: Chuẩn Bị Dữ Liệu & Audit Chống Rò Rỉ
```bash
python scripts/prepare_data.py --data-root data/VOC2012_train_val/VOC2012_train_val
```
- Phân chia official train thành `dev_train` (85%) và `dev_val` (15%) có phân tầng đa nhãn.
- Khóa official val thành `holdout.txt`.
- Xuất `artifacts/data/dataset_manifest.json` và `artifacts/data/audit.json` chứa mã băm SHA-256 thực tế.

### Bước 2: Huấn Luyện Giai Đoạn Phát Triển (Development Training)
```bash
python scripts/train.py --config configs/deeplabv3plus_resnet50_320.yaml
```
- Huấn luyện trên `dev_train`, đánh giá sau mỗi epoch trên `dev_val`.
- Tự động lưu `checkpoints/last.ckpt` (đầy đủ RNG states để resume) và `checkpoints/best.ckpt` (khi `val_miou_all` đạt đỉnh mới).

### Bước 3: Huấn Luyện Mô Hình Cuối Cùng (Final Fit)
```bash
python scripts/final_fit.py --best-checkpoint checkpoints/best.ckpt
```
- Đọc `best_epoch` từ `best.ckpt`, đóng băng cấu hình.
- Khởi tạo trọng số ImageNet mới, huấn luyện lại đúng `best_epoch` trên toàn bộ tập official train (`dev_train + dev_val`).
- Xuất artifact triển khai tinh gọn: `checkpoints/final_model.pth`.

### Bước 4: Đánh Giá Trên Tập Kiểm Thử Bị Khóa (Locked Holdout Evaluation)
```bash
python scripts/evaluate_holdout.py --checkpoint checkpoints/final_model.pth
```
- Đánh giá mô hình release trên tập official VOC `val` tại độ phân giải gốc của ảnh.
- Chạy đúng **1 lần duy nhất** để kết xuất báo cáo chuẩn mực `outputs/final_holdout_report.json`.

### Bước 5: Dự Đoán Bằng Giao Diện Dòng Lệnh (CLI Predict)
```bash
python scripts/predict.py --image path/to/image.jpg --checkpoint checkpoints/final_model.pth
```

### Bước 6: Khởi Chạy API Phục Vụ Suy Luận (FastAPI Serving)
```bash
uvicorn vocseg.api.app:app --host 0.0.0.0 --port 8000
```
- `GET /health`: Kiểm tra trạng thái máy chủ và thiết bị tính toán.
- `POST /segment`: Nhận file ảnh và trả về metadata (kích thước, danh sách lớp, tỷ lệ phủ %, độ bất định entropy, latency) kèm mặt nạ PNG base64.

### Bước 7: Trực Quan Hóa Tương Tác (Streamlit Interactive Demo)
```bash
streamlit run demo/streamlit_app.py
```
- Giao diện trực quan thuần túy cho phép người dùng kéo thả ảnh bất kỳ, quan sát Mặt nạ phân đoạn, Ảnh phủ màu (Overlay), Bản đồ bất định (Normalized Entropy Map) và Bảng tỷ lệ diện tích các lớp VOC.

---

## 5. Phân Cấp Hệ Thống Đo Lường (Metric Hierarchy)

| Nhóm | Metric | Vai trò trong hệ thống |
| :--- | :--- | :--- |
| **Primary** | `mIoU (All 21 classes)` | Tiêu chí chính chọn model checkpoint theo chuẩn Pascal VOC |
| **Diagnostic** | `mIoU (No Background)` | Đánh giá năng lực phát hiện 20 lớp đối tượng tiền cảnh |
| **Diagnostic** | `Per-class IoU` | Định lượng điểm mạnh / điểm yếu của từng lớp cụ thể |
| **Supporting** | `Mean Dice` | Đo lường độ trùng lặp tập hợp |
| **Supporting** | `Pixel Accuracy` | Tỷ lệ pixel được phân lớp chính xác tổng thể |
| **Boundary** | `Adaptive Boundary F1` | Đánh giá chất lượng đường biên với bán kính dung sai thích ứng theo kích thước ảnh ($0.5\% \times \text{diagonal}$) |
| **Error Analysis** | `Confusion Pairs` | Phân tích các cặp lớp nhầm lẫn nhiều nhất off-diagonal |
| **System** | `Latency (Mean / p50 / p95)` | Đo lường độ trễ suy luận tính bằng mili-giây |
| **System** | `FPS` | Tốc độ thông lượng xử lý ảnh |

---

## 6. Điểm Nổi Bật Trong Hồ Sơ AI Engineer (CV Value)

Dự án này chứng minh năng lực toàn diện của một **AI / Machine Learning Engineer**:

- **Data Engineering**: Data audit nghiêm ngặt, băm SHA-256 chống rò rỉ dữ liệu, phân tầng đa nhãn (multilabel stratification) cân bằng phân bố 20 lớp.
- **Deep Learning**: Kiến trúc DeepLabV3+ ResNet50 với Atrous Spatial Pyramid Pooling (ASPP), transfer learning, joint augmentation, loss kết hợp CE + 0.5 Dice, Automatic Mixed Precision (AMP).
- **ML Engineering**: Tách bạch 2 chặng (Development vs Final Fit), quản lý artifact 3 tầng, resume 100% tái lập (RNG capture), bảo vệ tuyệt đối Locked Holdout, Training-Serving Parity.
- **Computer Vision**: Khôi phục mặt nạ tại độ phân giải gốc của ảnh, ước lượng độ bất định qua Normalized Entropy map, đánh giá đường biên thích ứng (Adaptive Boundary F1).
- **Software Engineering**: Cấu trúc module chuẩn (`src/vocseg`), REST API với FastAPI, Interactive demo với Streamlit, bộ kiểm thử tự động (Unit, API, Integration E2E).
