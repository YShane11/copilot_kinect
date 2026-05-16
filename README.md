# Copilot Kinect

以 Kinect + YOLO + Flask 做出的課堂學習行為分析系統。專案會擷取 RGB / 深度影像，進行姿態偵測、人臉辨識、出席紀錄與課堂指標統計，並提供網頁儀表板查看即時與歷史資料。

## 主要功能

- Kinect v1 / v2 影像擷取與 RGB、深度畫面校正
- YOLO pose detection，用於舉手、專注、姿態與課堂互動指標
- InsightFace 人臉辨識與本地學生臉部資料庫
- Flask dashboard 顯示課程、學生、出席與分析資料
- 課堂紀錄匯出成 CSV / JSON
- 可選擇將課堂指標上傳到 Power Automate webhook
- `k-means.py` 用於課堂指標分群與分析輸出

## 專案結構

```text
app.py                         Flask 主程式與 API
src/vision/                    Kinect、辨識、姿態與指標核心邏輯
templates/                     Dashboard 與首頁模板
static/                        前端靜態檔與示範圖
scripts/                       校正、評估、調參與重建資料庫工具
reels/                         Kinect 錄影工具
data/administrators.example.json  帳號/課程設定範例
data/hand_raise_validation/    舉手偵測驗證用公開圖片
models/yolo/                   本地 YOLO 權重放置位置
tests/                         單元測試
```

## 安裝

建議使用 Windows + Python 3.10，並在專案根目錄建立虛擬環境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` 預設使用 CUDA 12.6 的 PyTorch wheel。如果電腦沒有 NVIDIA GPU 或 CUDA 版本不同，需要改成符合環境的 `torch` / `torchvision` 安裝方式。

## 本地設定

建立本地管理員與課程設定：

```powershell
Copy-Item data\administrators.example.json data\administrators.json
```

如果要啟用 Power Automate 指標上傳，設定環境變數：

```powershell
$env:POWER_AUTOMATE_UPLOAD_URL="https://..."
```

沒有設定 `POWER_AUTOMATE_UPLOAD_URL` 時，系統仍可正常在本地產生資料，只是不會上傳到外部 webhook。

## 模型與資料

YOLO 權重檔請放在 `models/yolo/`，例如：

```text
models/yolo/yolo26x-pose.pt
models/yolo/yolo26l-pose.pt
models/yolo/yolo26m-pose.pt
models/yolo/yolo26s-pose.pt
models/yolo/yolo26n-pose.pt
```

以下內容刻意不提交到 git：

- `.venv/`
- Kinect 錄影與測試影片
- YOLO `.pt` 權重檔
- 學生人臉資料與 embedding database
- 出席紀錄、歷史課堂指標與分群輸出
- `data/administrators.json` 等本地帳號設定

## 執行

啟動 Flask app：

```powershell
.\.venv\Scripts\python.exe app.py
```

常用頁面：

```text
http://127.0.0.1:5000/
http://127.0.0.1:5000/dashboard
```

## 常用工具

重建人臉資料庫：

```powershell
.\.venv\Scripts\python.exe scripts\rebuild_face_db.py
```

檢查 GPU / ONNX Runtime：

```powershell
.\.venv\Scripts\python.exe scripts\check_gpu_runtime.py
```

執行 K-Means 課堂分群：

```powershell
.\.venv\Scripts\python.exe k-means.py
```

## 測試

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```

目前測試主要覆蓋姿態與舉手判斷邏輯；硬體相關流程需要接上 Kinect 後在本機驗證。

## 推上 GitHub 前注意

這個 repo 已經透過 `.gitignore` 排除大型檔案與個資資料。若要重新確認將被提交的內容，可以執行：

```powershell
git status --short
git ls-files
```

不要把 webhook URL、學生照片、帳號密碼、影片與模型權重直接提交到公開 GitHub。
