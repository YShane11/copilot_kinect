# Kinect v2 + Python（VS Code）環境建置流程

本文說明如何在 Windows + VS Code 環境中，讓本專案使用 Kinect v2 擷取 RGB / Depth 影像，並接到 Flask 課堂分析系統。

## 目標

- 使用 Kinect v2 作為主要影像來源。
- 透過 `PyKinect2` 讀取 Color 與 Depth frame。
- 在本專案中啟動 `/kinect/color_feed` 與 `/kinect/depth_feed` 預覽。
- 讓出席辨識、姿態偵測、深度距離與課堂指標可以共用 Kinect v2 畫面。

## 技術架構

```text
Kinect v2
  -> Kinect for Windows Runtime / SDK 2.0
  -> PyKinect2 + comtypes
  -> src/vision/kinect_service.py
  -> Flask routes
  -> Dashboard / Attendance / Metrics
```

本專案的 Kinect v2 支援集中在：

- `src/vision/kinect_service.py`
- `src/vision/rgb_depth_alignment.py`
- `data/kinect_alignment_profiles.json`

## 一、硬體與系統需求

Kinect v2 對 USB 與電源比較挑剔，建議先確認：

- Windows 10 / Windows 11。
- Kinect for Windows v2 感測器。
- Kinect v2 專用電源供應器與 USB 3.0 轉接器。
- 主機具備可穩定供電的 USB 3.0 port。
- 已安裝 NVIDIA GPU / CUDA 12.6 對應的 PyTorch wheel（若要跑 YOLO / InsightFace）。

若 Kinect v2 反覆斷線，優先換 USB 3.0 port，不要接 USB hub。

## 二、安裝 Kinect v2 Runtime / SDK

1. 安裝 Microsoft Kinect for Windows Runtime 2.0。
2. 建議一併安裝 Kinect for Windows SDK 2.0，方便使用 Kinect Studio 或 SDK Browser 測試硬體。
3. 插上 Kinect v2 電源與 USB。
4. 在 Windows 裝置管理員確認 Kinect v2 裝置正常出現。

Kinect v2 與 Kinect v1 不同，通常不需要用 Zadig 改成 libusbK。若你之前為 Kinect v1 調過驅動，請不要把 Kinect v2 裝置改成 libusb/libusbK。

## 三、建立 Python 環境

從專案根目錄執行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` 已包含本專案需要的 Kinect v2 相關套件：

```text
comtypes
pykinect2 @ git+https://github.com/Kinect/PyKinect2.git@...
```

如果安裝 `pykinect2` 時因網路或 GitHub 存取失敗，重新執行 `pip install -r requirements.txt` 即可；若公司或學校網路有限制，請改用可連 GitHub 的網路。

## 四、VS Code 設定

在 VS Code 中選擇專案虛擬環境：

```text
Ctrl + Shift + P
Python: Select Interpreter
選擇 D:\copilot_kinect\.venv\Scripts\python.exe
```

建議用 VS Code 內建終端機執行指令，避免用到全域 Python。

## 五、指定 Kinect v2 後端

本專案會自動嘗試 Kinect v2，若 Kinect v2 暖機逾時，會 fallback 到 Kinect v1。若你要強制使用 Kinect v2，可在 PowerShell 設定：

```powershell
$env:KINECT_BACKEND="v2"
.\.venv\Scripts\python.exe app.py
```

可接受的值：

```text
v2
kinect2
kinect_v2
```

若要恢復自動模式，關掉該 PowerShell 視窗，或執行：

```powershell
Remove-Item Env:\KINECT_BACKEND
```

## 六、啟動專案

```powershell
.\.venv\Scripts\python.exe app.py
```

開啟：

```text
http://127.0.0.1:5000/
http://127.0.0.1:5000/dashboard
http://127.0.0.1:5000/kinect/color_feed
http://127.0.0.1:5000/kinect/depth_feed
```

成功時，`/api/kinect/status` 會回傳類似：

```json
{
  "status": "connected",
  "message": "Kinect connected (v2)",
  "source_mode": "kinect_v2_preview",
  "kinect_backend": "kinect_v2"
}
```

進入出席或課堂分析流程後，`source_mode` 可能會從 `kinect_v2_preview` 變成 `kinect_v2`。

## 七、Kinect v2 影像行為

Kinect v2 原始 Color frame 通常是 1920 x 1080 BGRA，本專案會轉成 BGR 給 OpenCV 使用。

Depth frame 通常是 512 x 424，單位是毫米。預覽模式會直接顯示原始 depth 的色彩化結果；出席分析模式則可將 depth 對齊到 color frame，讓姿態 keypoint 可以取到較合理的深度值。

相關環境變數：

```powershell
$env:KINECT_COLOR_MAX_WIDTH="1280"
$env:KINECT_IDLE_PREVIEW_FPS="15"
$env:KINECT_ATTENDANCE_FPS="12"
$env:KINECT_STREAM_PREVIEW_MAX_WIDTH="1280"
$env:KINECT_STREAM_JPEG_QUALITY="74"
$env:KINECT_V2_ALIGN_DEPTH_IN_ATTENDANCE="1"
$env:KINECT_V2_DEPTH_ALIGN_INTERVAL="0.85"
```

常用調整：

- 如果 GPU / CPU 負載過高，先降低 `KINECT_COLOR_MAX_WIDTH`，例如 `960`。
- 如果網頁串流延遲明顯，降低 `KINECT_STREAM_PREVIEW_MAX_WIDTH` 或 `KINECT_STREAM_JPEG_QUALITY`。
- 如果深度對齊造成出席流程卡頓，可暫時設為 `$env:KINECT_V2_ALIGN_DEPTH_IN_ATTENDANCE="0"`。

## 八、RGB / Depth 對齊設定

Kinect v2 對齊設定在：

```text
data/kinect_alignment_profiles.json
```

預設 v2 profile：

```json
{
  "enabled": true,
  "shift_x": 0.0,
  "shift_y": 0.0,
  "scale_x": 1.0,
  "scale_y": 1.0,
  "mirror_x": false,
  "mirror_y": false,
  "hole_fill_kernel": 5,
  "prefer_native_mapper": true
}
```

`prefer_native_mapper` 會優先使用 Kinect v2 SDK 的座標映射能力，把 depth map 到 color space。`hole_fill_kernel` 用來補一些映射後的小洞。

如果要用錄影做離線校正，可參考：

```powershell
.\.venv\Scripts\python.exe scripts\calibrate_quad_alignment.py --write
```

## 九、快速驗證 PyKinect2

可先測試套件是否能 import：

```powershell
.\.venv\Scripts\python.exe -c "import comtypes; from pykinect2 import PyKinectRuntime, PyKinectV2; print('PyKinect2 OK')"
```

再測試 runtime 是否能初始化：

```powershell
.\.venv\Scripts\python.exe -c "from pykinect2 import PyKinectRuntime, PyKinectV2; rt=PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth); print(rt.color_frame_desc.Width, rt.color_frame_desc.Height, rt.depth_frame_desc.Width, rt.depth_frame_desc.Height); rt.close()"
```

若第二個指令失敗，通常是 Kinect Runtime / SDK、USB 3.0、裝置供電或驅動狀態問題。

## 十、常見問題

### `Kinect v2 modules failed to load`

代表 Python 找不到或無法載入 `pykinect2` / `comtypes`。

處理方式：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "from pykinect2 import PyKinectRuntime, PyKinectV2; print('ok')"
```

請確認 VS Code interpreter 也是 `.venv`。

### `Kinect v2 runtime failed to initialize`

代表套件有載入，但 Kinect Runtime 無法啟動感測器。

檢查：

- Kinect v2 電源是否接好。
- 是否使用 USB 3.0 port。
- 裝置管理員是否顯示 Kinect v2。
- Kinect Studio / SDK Browser 是否能看到畫面。
- 是否有其他程式正在占用 Kinect。

### 一直顯示 `Kinect v2 is warming up`

Kinect v2 啟動 color / depth frame 有時需要數秒。本專案自動模式下，若 v2 在暖機時間內沒有 frame，會暫時 fallback 到 v1。

若你確定只要 v2：

```powershell
$env:KINECT_BACKEND="v2"
.\.venv\Scripts\python.exe app.py
```

### 深度畫面有洞或與 RGB 不完全對齊

這是 Kinect v2 的常見現象，尤其在邊緣、反光表面、遠距離或遮擋處會更明顯。

可調整：

- `data/kinect_alignment_profiles.json` 的 `hole_fill_kernel`。
- `KINECT_V2_DEPTH_ALIGN_INTERVAL`。
- Kinect 擺放角度與距離。

### 畫面卡頓

Kinect v2 的 1080p color frame 加上 YOLO / InsightFace 會吃不少資源。

優先調整：

```powershell
$env:KINECT_COLOR_MAX_WIDTH="960"
$env:KINECT_IDLE_PREVIEW_FPS="10"
$env:KINECT_ATTENDANCE_FPS="8"
```

若只是要測 UI，可使用影片模式：

```powershell
$env:KINECT_TEST_VIDEO="reels\recordings\your_test_video.mp4"
.\.venv\Scripts\python.exe app.py
```

## 十一、建議測試順序

1. 用 Kinect Studio 或 SDK Browser 確認硬體能正常出 color / depth。
2. 用 `python -c` 確認 `PyKinect2` 可以 import。
3. 用 `PyKinectRuntime` 初始化 color + depth。
4. 啟動 `app.py`。
5. 打開 `/api/kinect/status` 看 `kinect_backend` 是否為 `kinect_v2`。
6. 打開 `/kinect/color_feed` 與 `/kinect/depth_feed`。
7. 進入 dashboard 或 attendance 流程，確認辨識與深度指標正常。

## 十二、與 Kinect v1 文件的差異

Kinect v1 主要依賴 `libfreenect`、`libusb`、`vcpkg`、CMake 與 Python wrapper 編譯。

Kinect v2 主要依賴 Microsoft Kinect Runtime / SDK 2.0 與 `PyKinect2`，通常不需要自己編譯 C/C++ wrapper，也不需要 Zadig 改驅動。建置重點會從「編譯 libfreenect」變成「確認 SDK / USB 3.0 / PyKinect2 runtime 初始化」。
