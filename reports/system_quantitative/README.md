# Copilot Kinect 系統量化報表

產生日期：2026-06-10  
資料來源：`history/`、`clustering/`、`data/video_tuning/`、`data/embeddings/`、`tests/`

## 先講結論

| 項目 | 量化結果 | 解讀 |
| --- | ---: | --- |
| 單元測試 | 5 / 5 passed | 目前 hand-raise 規則測試通過 |
| 課堂匯出資料 | 157,835 筆 CSV rows / 6 天 | 系統已有足夠的課堂 metric 輸出可做分析 |
| 匯出學生數 | 9 位 | `history/` 目前涵蓋小型課堂子集 |
| Metric keys | 11 種 | presence、focus、posture、fatigue、distance、stillness、hand raise、attention、score、attendance 等 |
| Face embeddings | 8 筆 / 512 維 | 目前辨識資料庫可辨識 8 位已建立 embedding 的身份 |
| Identity-flow 最低 TPR | 97.50% | 既有標註 tuning set 上 recall 表現高 |
| Identity-flow 最低 precision | 100.00% | 既有標註 tuning set 未出現錯誤接受 |
| People-count 穩定度 | 98.69% - 99.30% | 人數追蹤在影片測試中大多落在預期 ±1 人 |
| K-means silhouette | 0.3594 | 分群可解釋但屬中等分離，不是強分類器 |

## 圖表索引

| 圖 | 說明 |
| --- | --- |
| [01_rows_by_metric.png](01_rows_by_metric.png) | 各課堂指標資料筆數 |
| [02_metric_means.png](02_metric_means.png) | 各數值指標平均值 |
| [03_identity_flow_scores.png](03_identity_flow_scores.png) | 身份辨識 TPR / Precision / 人數穩定度 |
| [04_cluster_size_score.png](04_cluster_size_score.png) | K-means 群集人數與平均成績 |
| [05_feature_score_correlation.png](05_feature_score_correlation.png) | 行為特徵與作業成績相關係數 |

## 1. 課堂資料匯出覆蓋率

| 指標 | 數值 |
| --- | ---: |
| CSV 檔案 | 6 |
| JSON 檔案 | 6 |
| 總資料列 | 157,835 |
| 課程數 | 1 |
| 課程名稱 | 人工智慧導論 |
| 學生數 | 9 |
| Metric keys | 11 |
| 日期 | 2026-04-23, 2026-04-29, 2026-04-30, 2026-05-12, 2026-06-03, 2026-06-04 |

### 各 metric 資料量

| Metric Key | 資料列數 |
| --- | ---: |
| `presence` | 40,015 |
| `fatigue` | 14,681 |
| `desk-distance` | 14,681 |
| `head-stability` | 14,681 |
| `posture-angle` | 14,681 |
| `hand-raise` | 14,681 |
| `focus-ratio` | 14,681 |
| `shared-attention` | 14,681 |
| `stillness` | 14,681 |
| `attendance-rate` | 186 |
| `assignment-score` | 186 |

## 2. 身份辨識與追蹤

### Face database 覆蓋率

| 項目 | 數值 |
| --- | ---: |
| 本機學生照片資料夾 | 16 |
| 訓練照片 | 24 |
| 已建立 embedding 身份 | 8 |
| Embedding 維度 | 512 |
| Embedding DB 對應照片數 | 24 |

### Identity-flow 驗證

| 測試資料 | 人數 | 秒數 | 驗證樣本 | 閾值 | TPR/Recall | Precision | FAR | Wrong Accept | 人數 MAE | ±1 穩定度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rgb_nir_side_by_side_20260420_174707_v2` | 5 | 306.60 | 761 | 0.32 | 97.50% | 100.00% | 0.00% | 0.00% | 0.3088 | 99.30% |
| `rgb_nir_side_by_side_20260421_225112_v2` | 2 | 15.33 | 33 | 0.25 | 100.00% | 100.00% | 0.00% | 0.00% | 0.0565 | 99.13% |
| `identity_flow_smoke` | 5 | 306.60 | 96 | 0.35 | 98.96% | 100.00% | 0.00% | 0.00% | 0.2418 | 98.69% |

這裡的 TPR / Precision / FAR 是根據 `data/video_tuning/` 內已有標註或配對資訊的 tuning JSON 計算。沒有 ground truth 的 snapshot compare CSV 只能當作 similarity / threshold 診斷，不拿來宣稱 accuracy。

## 3. Hand-raise / Pose-depth 邏輯

| 項目 | 數值 |
| --- | ---: |
| Hand-raise validation 圖片 | 7 |
| 檔名判斷 positive assets | 2 |
| 檔名判斷 negative assets | 5 |
| 現有 hand-raise unit tests | 5 / 5 passed |

目前這一塊有規則測試與圖片素材，但還沒有「每張圖的人工標註 + 模型輸出表」，所以不能誠實宣稱 image-level accuracy。下一步可建立 `image_path,true_label,pred_label` 表來算 precision / recall / F1。

## 4. K-means 學習行為分群

| 指標 | 數值 |
| --- | ---: |
| 學生 rows | 200 |
| 行為特徵數 | 8 |
| 群數 | 4 |
| Silhouette | 0.3594 |
| Calinski-Harabasz | 137.48 |
| Davies-Bouldin | 1.0898 |
| Inertia | 515.42 |

### 群集摘要

| Cluster | 學生數 | 平均成績 | 中位數 | 成績排名 |
| --- | ---: | ---: | ---: | ---: |
| C0 | 57 | 84.24 | 84.67 | 3 |
| C1 | 46 | 86.18 | 86.28 | 2 |
| C2 | 50 | 90.24 | 90.52 | 1 |
| C3 | 47 | 78.04 | 77.01 | 4 |

### 行為特徵與作業成績關係

| 特徵 | 與作業成績相關係數 |
| --- | ---: |
| 專注度平均 | +0.6000 |
| 身體前傾投入度平均 | +0.5220 |
| 共同注意力平均 | +0.4741 |
| 舉手比例平均 | +0.3819 |
| 頭部穩定度平均 | +0.3195 |
| 頭與桌距離平均 | -0.5486 |
| 疲勞度平均 | -0.5653 |
| 發呆指數平均 | -0.5659 |

解讀：專注度、身體前傾投入度、共同注意力與成績呈正相關；疲勞度、發呆指數、頭與桌距離與成績呈負相關。Silhouette 0.3594 表示分群有可解釋性，但分離程度中等，適合用於探索與報表，不宜直接當成強監督式分類器。

## 5. 目前不能宣稱 accuracy 的部分

| 系統部分 | 現況 | 還需要什麼資料 |
| --- | --- | --- |
| Kinect RGB/depth 對齊 | 有 calibration samples / profile | RGB-depth 對齊點 ground truth，才能算 pixel error |
| YOLO pose detection | 有模型與 runtime pipeline | 人工標註 bbox/keypoint benchmark |
| 課堂行為 metric correctness | 有大量匯出資料 | per-student classroom event ground truth |
| Hand-raise 圖片準確率 | 有素材與 unit tests | 每張圖 true label + pred label |
| 完整 attendance accuracy | 有 identity-flow tuning 指標 | 每堂課逐時間 attendance ground truth |

## 6. 附檔

- [summary_metrics.csv](summary_metrics.csv)：報表核心數值摘要
- `01_*.png` ~ `05_*.png`：視覺化圖表
