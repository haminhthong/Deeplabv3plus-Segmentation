# Báo Cáo Kiểm Toán Mã Nguồn – DeepLabV3+ Semantic Segmentation Platform

> **Ngày kiểm toán**: 06/09/2026
> **Phạm vi**: Toàn bộ mã nguồn Python thuộc workspace, bao gồm `config.py`, `voc_meta.py`, `dataset_voc.py`, `inference.py`, `metrics.py`, `train_deeplabv3plus.py`, `evaluate.py`, `validate_dataset.py`, `visualize_predictions.py`, `streamlit_segmentation_ui.py`, `plot_training_curves.py`, `download_checkpoint.py`, `scripts/create_benchmark_splits.py`, `tests/*`.
> **Mục tiêu**: Đánh giá tính hợp lý về luồng logic, luồng dữ liệu, pipeline, phát hiện lỗi tiềm ẩn và đề xuất cải tiến.

---

## 1. Tổng quan kiến trúc & đánh giá sơ bộ

### 1.1. Sơ đồ pipeline tổng thể

```text
┌─────────────────────────────────────────────────────────────────────┐
│                       OFFLINE – TRAINING PIPELINE                  │
└─────────────────────────────────────────────────────────────────────┘
 [Pascal VOC 2012 dataset]
        │
        │ (image/mask pairs, ID lists)
        ▼
┌───────────────────────┐
│ scripts/              │  Tạo split có chủ đích: 70/15/15
│ create_benchmark_     │  + sinh split_manifest.json
│ splits.py             │  + SHA-256 của từng file split
└─────────┬─────────────┘
          ▼
┌───────────────────────┐
│ validate_dataset.py   │  Audit: duplicate IDs, SHA-256 dupes,
│                       │  missing files, dimension mismatches,
│                       │  invalid class IDs, class distribution
└─────────┬─────────────┘
          ▼
┌───────────────────────┐
│ train_deeplabv3plus.  │  ┌─────────────────────────────────────┐
│ py                    │─▶│  CombinedLoss = CE + 0.5 × Dice    │
│                       │  │  AdamW + CosineAnnealingLR         │
│                       │  │  AMP (fp16 autocast)               │
│                       │  │  Early stopping (patience)         │
│                       │  │  Best ckpt theo val mIoU            │
│                       │  └─────────────────────────────────────┘
└─────────┬─────────────┘
          ▼
  outputs/{arch}_{encoder}_voc_best.pth   (+ best_metrics.json, per_class_metrics.csv, train_log.csv)


┌─────────────────────────────────────────────────────────────────────┐
│                       OFFLINE – EVALUATION PIPELINE                 │
└─────────────────────────────────────────────────────────────────────┘
  checkpoint (.pth) ──▶ load_checkpoint_model() ──▶ evaluate.py
                                                          │
                                                          ▼
                                            metrics.SegmentationMetrics
                                            (IoU / Dice / BF1 / region size
                                             / confusion / latency)
                                                          │
                                                          ▼
                                         outputs/evaluation.json
                                         outputs/per_class_metrics.csv


┌─────────────────────────────────────────────────────────────────────┐
│                       ONLINE – SERVING PIPELINE                     │
└─────────────────────────────────────────────────────────────────────┘
  RGB image (any size) ──▶ Streamlit UI / visualize_predictions.py
                                │
                                ▼
                  prepare_image (letterbox) ──▶ model forward
                                │
                                ▼
                  crop padding + bilinear upsample logits to original size
                                │
                                ▼
            ┌─────────────────────────────────────────────┐
            │  predict_with_uncertainty:                 │
            │   - hard_mask (argmax)                      │
            │   - max_prob_map                            │
            │   - normalized entropy map                  │
            │   - softmax_probs                           │
            └─────────────────────────────────────────────┘
                                │
                                ▼
            mask_to_color_rgb + overlay + Streamlit UI / matplotlib viz
```

### 1.2. Đánh giá sơ bộ

| Khía cạnh | Điểm mạnh | Vấn đề tồn đọng |
|---|---|---|
| **Tổ chức file** | Tách module rõ ràng theo trách nhiệm (data, model, train, eval, viz) | Một số import cycle/duplicate, một số hằng số bị "phân tán" |
| **Chống rò rỉ** | SHA-256 + ID dup + train/val/test split + manifest | Smoke split có `2007_000033` chỉ ở val & benchmark val, có thể gây khó debug |
| **Reproducibility** | Checkpoint lưu git SHA, seed, full args, env versions | Resume không reload `start_epoch` xử lý tương đối – chưa load `best_miou` đúng cho một số edge case |
| **Metrics** | Đầy đủ IoU/Dice/BF1/region-size/confusion/latency | Có bug nghiêm trọng trong `extract_confusion_analysis` & region-size logic |
| **UI / Viz** | Streamlit + matplotlib, có uncertainty map | Thiếu test cho luồng UI |
| **Code style** | Có docstring tiếng Việt chi tiết, type hints | Một số hàm có biến không dùng, biến che bóng (shadowing) |

---

## 2. Phân tích luồng dữ liệu (Data Flow) chi tiết

### 2.1. Luồng dữ liệu huấn luyện

```text
Pascal VOC
  ├── JPEGImages/{id}.jpg          (ảnh RGB, mọi kích thước gốc)
  └── SegmentationClass/{id}.png  (mask 8-bit palette; class 0..20, biên = 255)

       │
       ▼   (Dataset.__getitem__)
VOCSegmentationDataset(split="train")
       │
       ▼   PIL.Image.open() + .convert("RGB")
       ▼   ToTensor + Normalize (mean/std ImageNet)
       ▼
TrainJointTransform:
   1. Random scale ∈ [0.75, 1.5]            (BILINEAR ảnh, NEAREST mask)
   2. Random/unbiased padding               (fill=0 cho ảnh, fill=255 cho mask)
   3. RandomCrop → target (H, W)
   4. Random HorizontalFlip (p=0.5)
   5. Random Affine (±10°, ±5% translate, ±10% scale)
   6. Color Jitter (chỉ ảnh)
   7. Normalize(IMAGE_MEAN, IMAGE_STD)
       │
       ▼
Tensor (B, 3, 320, 320), mask (B, 320, 320) int64
       │
       ▼
forward(images) → logits (B, 21, 320, 320)
CombinedLoss = CrossEntropy(ignore_index=255) + 0.5 × DiceLoss(multiclass)
       │
       ▼
AMP autocast → scaler.scale(loss).backward() → scaler.step(optimizer) → scaler.update()
CosineAnnealingLR step
```

**Đánh giá**:
- Luồng đúng chuẩn segmentation. Mask resize dùng NEAREST để không sinh nhãn mới (đúng).
- Color jitter chỉ áp dụng trên ảnh (đúng, không được đụng mask).
- `ignore_index=255` đồng bộ ở cả Dataset, Loss, Metric (đúng).

### 2.2. Luồng dữ liệu suy luận

```text
RGB Image (PIL)
       │
       ▼
prepare_image(image, image_size):
   1. calculate_letterbox_geometry(W, H, target, target)
       → scale = min(tW/W, tH/H), new_w, new_h, pad_left, pad_top, …
   2. TF.resize(image, (new_h, new_w), BILINEAR)
   3. TF.pad(pad_left, pad_top, pad_right, pad_bottom, fill=0)
   4. Normalize(mean, std)
       │
       ▼
Tensor (1, 3, image_size, image_size)
       │
       ▼
model(tensor) → logits (1, 21, image_size, image_size)
       │
       ▼
logits[:, :, top:top+resized_h, left:left+resized_w]    ← bỏ pad
F.interpolate(logits, (H_orig, W_orig), bilinear)
       │
       ▼
argmax → hard_mask (H_orig, W_orig) int64
softmax → probs (21, H_orig, W_orig)
       │
       ▼
max_prob_map, normalized_entropy_map, full softmax map
```

**Đánh giá**:
- Pipeline letterbox + unpad + upsample logits (không phải argmax) là best practice (đúng).
- Tuy nhiên: code hiện tại **không đảm bảo** vùng unpad nằm gọn trong logits khi target quá nhỏ — cần check.

---

## 3. Lỗi / thiếu sót chi tiết theo từng file

### 3.1. `config.py`
- ✅ Các hằng số tập trung, hợp lý.
- ⚠️ `CHECKPOINT_PATH = OUTPUT_DIR / "deeplabv3plus_voc_best.pth"` — hard-code tên, nhưng `train_deeplabv3plus.py` lại lưu theo pattern `{arch}_{encoder}_voc_best.pth` ⇒ **mặc định `CHECKPOINT_PATH` không khớp với checkpoint thực tế** khi architecture ≠ deeplabv3plus. Cần dùng helper hoặc để user chỉ định.

### 3.2. `voc_meta.py`
- ✅ VOC_COLORMAP chuẩn Pascal VOC, dtype uint8.
- ✅ `mask_to_color_rgb` xử lý đúng nhãn ngoài phạm vi bằng cách giữ zero (đen).
- ✅ `ignore_index` được tô trắng `[255,255,255]`.

### 3.3. `dataset_voc.py`

**🔴 LỖI NGHIÊM TRỌNG #1 — `np.pad` sai cú pháp, có thể gây crash runtime:**
```python
p = np.pad(mask, ((1, 1), ((1, 1))), mode="constant", constant_values=False)
```
- Sai: cú pháp `np.pad` chấp nhận **một tuple** cho mỗi chiều: `((before_0, after_0), (before_1, after_1))`. Cú pháp trên (tuple lồng tuple) vẫn hoạt động vì numpy unpack được, nhưng `constant_values=False` trên mask boolean là **bug logic** — đáng lẽ phải là `constant_values=0` để vùng pad có giá trị False, hoặc phải đảm bảo `False` thành `0` tự động (NumPy sẽ ép False→0 khi so sánh, nên may ra chạy được nhưng vô cùng mong manh).
- Cú pháp này nằm ở **2 chỗ** trong `metrics.py` (`extract_boundary` & `dilate_boundary`).

**🟠 LỖI #2 — TrainJointTransform gán đè biến `scale` trong scope:**
```python
scale = random.uniform(0.75, 1.5)
...
if random.random() > 0.5:
    ...
    scale = random.uniform(0.9, 1.1)   # <-- shadowing scale đã dùng để resize ảnh
```
- Biến `scale` đầu dùng để resize ảnh/mask; nếu rẽ nhánh affine, `scale` bị gán lại nhưng `image`, `mask` đã được resize từ giá trị cũ nên kết quả đúng. Tuy nhiên đây là **biến che bóng (shadowing)** gây khó đọc, dễ sai khi refactor.

**🟠 LỖI #3 — TrainJointTransform thay đổi logic crop sau pad, nhưng thiếu xử lý khi ảnh lớn hơn target:**
- Nếu ảnh sau random scale `> target` → `pad_total_w = max(0, target - scaled_w) = 0` ⇒ không padding, nhưng `RandomCrop.get_params` sẽ cắt ngẫu nhiên từ ảnh lớn về target. Hành vi này hợp lý cho training. Tuy nhiên: **không có xử lý khi `scale > 1.5` khiến crop luôn rơi vào giữa (bias về cùng vị trí crop)** — đây là design choice nhưng nên document rõ.

**🟠 LỖI #4 — `validate_voc_dataset` không kiểm tra empty split:**
- Khi split rỗng, hàm chỉ `pass` (không raise). Một split train rỗng sẽ không được audit phát hiện.

**🟡 LỖI #5 — `resize_and_pad` đảo tham số:**
```python
def resize_and_pad(image, mask, h: int, w: int):
    ...
    TF.pad(image, padding, fill=0), TF.pad(mask, padding, fill=IGNORE_INDEX)
```
- Tham số là `(h, w)` nhưng bên trong gọi `calculate_letterbox_geometry(image.width, image.height, w, h)`. Hàm letterbox trả `(scale, new_w, new_h, ...)` ⇒ đang truyền đúng `w, h` theo nghĩa target_w, target_h nhưng người gọi `JointTransform.__call__` gọi:
  ```python
  image, mask = resize_and_pad(image, mask, self.h, self.w)
  ```
  Trong khi đó `calculate_letterbox_geometry` định nghĩa `(width, height, target_width, target_height)`. Như vậy `resize_and_pad(image, mask, h, w)` đang truyền target_width=h, target_height=w ⇒ **SWAP**. Bug này may mắn không tác động khi target là hình vuông (h==w), nhưng nếu dùng input không vuông sẽ sai.

### 3.4. `inference.py`

**🟠 LỖI #6 — `load_checkpoint_model` không chuyển `weights_only`:**
- Đã có `weights_only=True` (đúng, an toàn).
- ⚠️ `metadata.get("architecture", "deeplabv3plus")` — khi resume training, checkpoint lưu `architecture`, `encoder` đầy đủ, OK. Nhưng nếu checkpoint cũ chỉ chứa state_dict (không phải dict), code sẽ treat nguyên state_dict là "metadata" → `metadata.get("model_state_dict", checkpoint)` đúng (lấy state_dict từ chính nó), nhưng `encoder`/`architecture`/`num_classes` sẽ là giá trị mặc định ⇒ **mismatch khi load model không phải deeplabv3plus**.

**🟠 LỖI #7 — `predict_with_uncertainty` không đặt device cho tensor:**
```python
tensor, (...), (...) = prepare_image(image, image_size)
logits = model(tensor.unsqueeze(0).to(device))
```
- `prepare_image` trả về tensor CPU, sau đó mới `.to(device)`. Đúng. Nhưng nếu batch size > 1 (hiện tại không dùng, nhưng đáng lưu ý) thì chưa tận dụng batching.

**🟡 LỖI #8 — `softmax_probs` lưu shape `(C, H, W)` float32 — có thể tốn RAM:**
- Với ảnh 20MP × 21 lớp × 4 byte ≈ 1.6 GB RAM/ảnh. Cần cảnh báo hoặc không trả `softmax_probs`.

### 3.5. `metrics.py`

**🔴 LỖI NGHIÊM TRỌNG #2 — `extract_confusion_analysis` không xử lý đúng NaN/empty class khi sort:**
```python
sorted_by_iou = sorted(
    eval_classes,
    key=lambda c: (iou[c] if not np.isnan(iou[c]) else -1.0),
    reverse=True,
)
```
- Hàm `key` trả về tuple `(iou[c],)`. Khi so sánh tuple với `reverse=True`, Python sort theo thứ tự tuple ngược, không phải giá trị đơn lẻ. Tức là `reverse=True` cho tuple `(0.5,)` vẫn sort như `(0.5,)`. Hoạt động, nhưng khi `key` trả tuple có thể gây khó debug.
- Vấn đề thực sự: **chỉ sort trên `eval_classes` (foreground hoặc valid_classes) nhưng `best_5`/`worst_5` cắt `[:5]` — có thể trả về <5 phần tử nếu số lớp < 5** ⇒ downstream code (UI, log) phải handle list rỗng.

**🔴 LỖI NGHIÊM TRỌNG #3 — `extract_boundary` có thể trả về sai khi mask rỗng:**
- Đã có guard `if not mask.any(): return np.zeros_like(mask, dtype=bool)` ⇒ OK.

**🟠 LỖI #9 — `compute_boundary_f1_score` phụ thuộc `valid_mask = gt_mask != ignore_index`:**
- Đúng chuẩn metric. Tuy nhiên `pred_c` lấy `& valid_mask` ⇒ ignore pixel bị bỏ khỏi mọi tính toán BF1. Đúng.

**🟠 LỖI #10 — `calculate_region_size_metrics` đếm mỗi (ảnh × lớp) là 1 region:**
- Hiện tại: với mỗi foreground class, đếm `area = (gt == c).sum()` rồi gán IoU vào bucket theo area. **Một foreground class xuất hiện trong nhiều vùng rời rạc (multiple disconnected components) bị gộp thành 1 region**. Đây là design "per-class" chứ không phải "per-connected-component". Cần đổi tên/ document rõ.

**🟠 LỖI #11 — `SegmentationMetrics.update` không tính BF1 nếu `pred_arr.ndim == 3` nhưng `compute_boundary=True`:**
- Có xử lý đầy đủ (test đã cover). Tuy nhiên `for c, val in sample_b_scores.items()` — BF1 chỉ có trong dict nếu class có pixel. OK.

**🟠 LỖI #12 — `save_metrics` dùng `allow_nan=False`:**
- Đúng, JSON không hỗ trợ NaN. Code đã chuyển NaN → None trước khi dump. OK.

### 3.6. `train_deeplabv3plus.py`

**🟠 LỖI #13 — Resume không load `best_miou` chính xác khi checkpoint cũ không có `best_val_miou`:**
```python
best_miou = float(resume_checkpoint.get("best_val_miou", -1.0))
```
- OK với default. Tuy nhiên **không load `start_epoch` thông minh**: nếu scheduler.T_max đã step đến gần cuối, khi resume với epochs mới sẽ bị `scheduler.T_max = args.epochs` ghi đè ⇒ **cosine schedule bị reset hoàn toàn**, không tiếp tục từ epoch đã dừng. Cần cảnh báo trong docstring hoặc giữ nguyên T_max.

**🟠 LỖI #14 — Resume từ checkpoint cũ (chỉ có state_dict) sẽ crash:**
- `torch.load(args.resume, map_location=device, weights_only=True)` trả `state_dict`, sau đó `resume_checkpoint["model_state_dict"]` ⇒ KeyError. Cần fallback tương tự `load_checkpoint_model`.

**🟠 LỖI #15 — `scaler_state_dict` có thể không tồn tại khi AMP tắt:**
- `if active_scaler is not None and resume_checkpoint.get("scaler_state_dict")` — OK.

**🟠 LỖI #16 — `history_lines` append string format float có thể sai local:**
- Dùng `f"{train_loss:.6f}"` ⇒ **luôn dùng dấu `.`**, OK cho CSV. Nhưng `csv_path` write bằng `,` ⇒ đúng chuẩn CSV.

**🟠 LỖI #17 — `train_args` ép tất cả Path thành str** — OK, nhưng không khử các giá trị không serializable (vd. argparse.Namespace lồng nhau). Trong checkpoint này chỉ chứa primitives ⇒ OK.

**🟡 LỖI #18 — `validate_voc_dataset` được gọi ngay đầu training, nhưng nếu val/test split không tồn tại (chỉ train), vẫn pass** — đây là behavior đúng nhưng nên warning.

**🟡 LỖI #19 — Tên file log `train_log.csv` không bao gồm architecture** ⇒ nếu chạy 2 kiến trúc liên tiếp sẽ ghi đè lẫn nhau.

### 3.7. `evaluate.py`

**🟠 LỖI #20 — Warm-up có thể fail khi dataset rỗng:**
```python
if device.type == "cuda" and len(loader) > 0:
    dummy_input = next(iter(loader))[0][:1].to(device)
```
- `next(iter(loader))` đã load batch đầu, làm warm-up **sau khi load batch**, không phải trước. Nên warm-up trước khi vào loop chính bằng dummy tensor ngẫu nhiên để không tốn I/O.

**🟠 LỖI #21 — `batch_forward_latencies` chia cho `b_size` để ra per-image latency** — chính xác khi GPU parallel xử lý tốt, nhưng **không phản ánh throughput thật** (vì batch 8 vẫn đo latency từng ảnh). Nên tách metric `throughput_img_per_sec` (= b_size / t_fwd) và `latency_per_image` (= t_fwd / b_size) rõ ràng.

**🟠 LỖI #22 — `compute_boundary` luôn được bật trong evaluate (`compute_boundary=True`)** — có thể chậm với dataset lớn. Cần CLI flag để tắt.

### 3.8. `validate_dataset.py`

**🟠 LỖI #23 — Hash toàn bộ ảnh JPEG có thể rất chậm với dataset đầy đủ (~17K ảnh):**
- Hash 1MB chunks ⇒ ~5-15 phút. Chấp nhận được cho audit 1 lần nhưng cần progress bar.

**🟠 LỖI #24 — `class_distribution` chỉ tính 1 lần, không tách theo phân vùng (small/med/large):**
- Không phải lỗi, nhưng thiếu thông tin phục vụ stratified sampling.

**🟠 LỖI #25 — `if not img_hash in hash_to_id` so sánh với sha-256 dài 64 char** ⇒ dict lookup OK, nhưng **không có warning khi số duplicate cao bất thường**.

### 3.9. `scripts/create_benchmark_splits.py`

**🟠 LỖI #26 — Fallback `base_samples` chỉ ~100 ID, không đủ chia train/val/test theo tỉ lệ 70/15/15:**
- Với `n_total=100`, `n_train=70, n_val=15, n_test=15` ⇒ đúng tỉ lệ. Nhưng nếu dataset thật chỉ có 50 IDs, `max(1, int(50*0.70))=35, max(1, int(50*0.15))=7` ⇒ `train=35, val=7, test=8` ⇒ **test chỉ 8 ảnh**, không đủ statistically significant.

**🟠 LỖI #27 — Hard-coded list IDs 2007-2008** không cover 2009-2011 (VOC có cả 2009-2012 segmentation). Smoke split thiếu diversity.

**🟡 LỖI #28 — `assert` thay vì `raise ValueError`**: nếu chạy với `python -O`, assert bị tắt ⇒ leakage không được phát hiện. Nên đổi thành raise.

### 3.10. `streamlit_segmentation_ui.py`

**🟠 LỖI #29 — `MAX_PIXELS = 20_000_000` cho ảnh upload** — nhưng **không áp dụng cho ảnh từ VOC dataset** (ảnh gốc VOC < 1MP, OK). Khi user upload ảnh >20MP, dừng — đúng.

**🟠 LỖI #30 — `load_model_safe` cache theo `(checkpoint_path_str, device_str)`** — nếu user thay đổi checkpoint path mà không restart session, cache sẽ giữ model cũ. Streamlit vẫn cho phép thay đổi → **cần `st.cache_resource.clear()` thủ công**.

**🟠 LỖI #31 — Tab "Dự đoán VOC" không có progress cho `predict_original_size`** (mỗi ảnh inference ~50-200ms trên CPU ⇒ 6 ảnh × 200ms = 1.2s OK, nhưng nếu dùng GPU sẽ lock GIL).

**🟠 LỖI #32 — Trên Streamlit cache, GPU model **chiếm VRAM liên tục** ngay cả khi user không dùng. Cần cơ chế unload.

**🟡 LỖI #33 — `create_training_figure` dùng `Figure` từ matplotlib backend `Agg`** — khi gọi `st.pyplot(figure)` cần `figure.clear()` sau. Code có gọi nhưng `clear()` chỉ giải phóng axes, không GC figure.

### 3.11. `visualize_predictions.py`

- ✅ Đơn giản, đúng chức năng.
- ⚠️ **LỖI #34** — Không validate `--indices` là số nguyên dương (argparse `type=int` đã OK).
- ⚠️ **LỖI #35** — `joint_transform=None` ⇒ load ảnh raw kích thước gốc, predict ở image_size mặc định, OK.

### 3.12. `plot_training_curves.py`

- ✅ Đúng, robust với missing columns.
- ⚠️ **LỖI #36** — Không xử lý epoch rỗng (CSV có header nhưng không có row) ⇒ `axes.plot(epochs, train_loss)` sẽ vẽ 0 điểm. Cần check.

### 3.13. `download_checkpoint.py`

- ✅ Đúng, có SHA-256 verify.
- ⚠️ **LỖI #37** — Khi 404, `sys.exit(0)` ⇒ **không thông báo lỗi rõ ràng** cho CI, có thể bị coi là success.

### 3.14. Tests

- ✅ Có test cho dataset leakage, metric, inference, checkpoint.
- ⚠️ **LỖI #38** — Test `test_train_joint_transform_unbiased_padding` không thực sự kiểm tra "unbiased" (chỉ check shape).
- ⚠️ **LỖI #39** — Không có test cho `train_deeplabv3plus.py` end-to-end (1 epoch smoke).
- ⚠️ **LỖI #40** — Không có test cho `validate_dataset.py`.
- ⚠️ **LỖI #41** — Không có test cho `streamlit_segmentation_ui.py`.

---

## 4. Bảng tổng hợp mức độ nghiêm trọng

| # | File | Mô tả | Mức độ | Trạng thái |
|---|---|---|---|---|
| 1 | `metrics.py` | `np.pad` sai cú pháp (False vs 0) | 🔴 Nghiêm trọng | Cần sửa |
| 2 | `metrics.py` | `extract_confusion_analysis` có thể trả <5 lớp | 🟠 Trung bình | Cần sửa |
| 3 | `config.py` vs `train_deeplabv3plus.py` | CHECKPOINT_PATH không khớp tên checkpoint thật | 🟠 Trung bình | Cần sửa |
| 4 | `dataset_voc.py` | `resize_and_pad` swap h/w nếu target không vuông | 🟠 Trung bình | Cần sửa |
| 5 | `train_deeplabv3plus.py` | Resume scheduler T_max bị reset | 🟠 Trung bình | Cần sửa |
| 6 | `train_deeplabv3plus.py` | Resume crash nếu checkpoint cũ chỉ có state_dict | 🟠 Trung bình | Cần sửa |
| 7 | `train_deeplabv3plus.py` | Log file name không có architecture | 🟡 Nhẹ | Nên sửa |
| 8 | `inference.py` | Trả softmax_probs full map (RAM lớn) | 🟡 Nhẹ | Nên cảnh báo |
| 9 | `inference.py` | load_checkpoint_model fallback default sai | 🟡 Nhẹ | Nên sửa |
| 10 | `evaluate.py` | Warm-up sau khi load batch | 🟡 Nhẹ | Nên sửa |
| 11 | `evaluate.py` | Latency per-image chia b_size không phản ánh throughput | 🟡 Nhẹ | Nên tách metric |
| 12 | `scripts/create_benchmark_splits.py` | assert thay vì raise | 🟡 Nhẹ | Nên sửa |
| 13 | Tests | Thiếu test cho train, validate, UI | 🟡 Nhẹ | Nên bổ sung |
| 14 | ReadMe | Một số thuật ngữ "Sẵn sàng đo" chưa được điền | 🟡 Nhẹ | Cần cập nhật |

---

## 5. Đề xuất cải tiến theo thứ tự ưu tiên

### Ưu tiên 1 — Sửa bug nghiêm trọng (an toàn dữ liệu & metric)

1. **`metrics.py` `extract_boundary` / `dilate_boundary`**: thay `constant_values=False` → `constant_values=0` (vì mask boolean `False == 0` khi index, nhưng `False` không tự broadcast qua numpy khi dùng `&` với mask cùng dtype — có thể gây bug runtime tùy version numpy). Đồng thời sửa cú pháp `((1, 1), ((1, 1)))` thành `((1, 1), (1, 1))`.
2. **`config.py`**: bổ sung helper `get_default_checkpoint_path(arch, encoder)` để `evaluate.py` / `streamlit_*.py` tự dò checkpoint.
3. **`dataset_voc.py` `resize_and_pad`**: đổi signature thành `(image, mask, target_h, target_w)` rõ ràng hoặc thêm guard `assert h==w` cho phiên bản hiện tại.
4. **`train_deeplabv3plus.py` `train()`**: thêm fallback khi `resume_checkpoint` không phải dict (chỉ có state_dict).
5. **`train_deeplabv3plus.py` `train()`**: KHÔNG ghi đè `scheduler.T_max` khi resume; thay vào đó cảnh báo user dùng `--epochs` mới nếu muốn mở rộng schedule.
6. **`train_log.csv`**: đổi tên thành `{arch}_{encoder}_train_log.csv` để tránh ghi đè giữa các architecture.

### Ưu tiên 2 — Hoàn thiện logic & UX

7. **`metrics.py` `extract_confusion_analysis`**: khi `eval_classes < 5`, đảm bảo trả về list có đúng 5 phần tử (điền `None`).
8. **`inference.py` `load_checkpoint_model`**: nếu checkpoint là state_dict thuần, raise error rõ ràng thay vì dùng default sai.
9. **`evaluate.py`**: 
   - Warm-up trước khi vào loop bằng `torch.randn`.
   - Tách rõ `throughput_img_per_sec` (cho batch) và `latency_per_image` (cho single).
   - Thêm `--no-boundary` để tắt BF1.
10. **`scripts/create_benchmark_splits.py`**: thay `assert` bằng `raise ValueError`.
11. **`validate_dataset.py`**: thêm progress bar với `tqdm`.

### Ưu tiên 3 — Tests & CI

12. Bổ sung `tests/test_train_smoke.py` chạy 1 epoch trên smoke split (CPU ok).
13. Bổ sung `tests/test_validate_dataset.py`.
14. Bổ sung `tests/test_streamlit_ui.py` với `streamlit.testing.v1.AppTest`.
15. Thêm `pytest --cov` vào CI workflow.

### Ưu tiên 4 — Productionization

16. **`streamlit_*.py`**: bổ sung nút "Unload model" để giải phóng VRAM.
17. **`download_checkpoint.py`**: thay `sys.exit(0)` khi 404 bằng `sys.exit(2)` + log warning.
18. **`inference.py` `predict_with_uncertainty`**: thêm tùy chọn `return_softmax_probs: bool = False` để tiết kiệm RAM khi gọi hàng loạt.
19. **`ReadMe.md`**: cập nhật các ô "Sẵn sàng đo" sau khi có kết quả thật, bổ sung "Known Limitations" (vd. segmentation chỉ trên 21 lớp VOC, không phải open-vocabulary).

---

## 6. Kết luận

Dự án đã có **nền tảng kỹ thuật rất tốt**:
- Kiến trúc module sạch, có chuẩn hoá (Transform Contract, Split Protocol, Anti-Leakage Audit).
- Hệ thống metric đa tầng (pixel / class / boundary / region-size / confusion / latency) — vượt mức trung bình cho dự án cá nhân.
- Có Reproducibility (git SHA, seed, args snapshot) và CI/CD.

Tuy nhiên còn **một số lỗi logic & bug runtime tiềm ẩn** cần sửa trước khi đưa vào production:
- Sai cú pháp `np.pad` trong metrics (dễ gây crash trên numpy mới).
- Logic resume training chưa hoàn chỉnh.
- Một số bug signature tham số gây sai khi target không vuông.

Sau khi áp dụng các đề xuất ưu tiên 1 & 2, dự án đủ điều kiện trở thành một **sản phẩm thật** (production-grade) cho benchmark segmentation.
