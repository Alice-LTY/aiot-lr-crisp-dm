# 🛠️ 開發指南 (Development Guide)

本文件提供開發者詳細的開發環境設定和開發流程說明。

## 📋 目錄

- [環境需求](#環境需求)
- [環境設定](#環境設定)
- [專案結構](#專案結構)
- [開發工作流程](#開發工作流程)
- [測試指南](#測試指南)
- [部署指南](#部署指南)
- [常見問題](#常見問題)

---

## 💻 環境需求

### 必要軟體

- **Python**: 3.8+ (建議 3.12)
- **pip**: 最新版本
- **Git**: 2.0+
- **作業系統**: macOS / Linux / Windows

### 推薦工具

- **IDE**: VS Code / PyCharm / Jupyter Lab
- **Terminal**: iTerm2 (macOS) / Windows Terminal (Windows)
- **Git GUI**: GitHub Desktop / GitKraken (可選)

---

## 🔧 環境設定

### 1. Clone 專案

```bash
# HTTPS
git clone https://github.com/Alice-LTY/aiot-lr-crisp-dm.git

# SSH（建議）
git clone git@github.com:Alice-LTY/aiot-lr-crisp-dm.git

# 進入專案目錄
cd aiot-lr-crisp-dm
```

### 2. 建立虛擬環境

#### macOS / Linux

```bash
# 使用 venv
python3 -m venv aiot_env

# 啟動虛擬環境
source aiot_env/bin/activate

# 確認 Python 版本
python --version
```

#### Windows

```powershell
# 使用 venv
python -m venv aiot_env

# 啟動虛擬環境 (PowerShell)
.\aiot_env\Scripts\Activate.ps1

# 或 CMD
.\aiot_env\Scripts\activate.bat

# 確認 Python 版本
python --version
```

#### 使用 Conda（可選）

```bash
# 建立環境
conda create -n aiot_env python=3.12

# 啟動環境
conda activate aiot_env

# 確認安裝
conda list
```

### 3. 安裝依賴套件

```bash
# 升級 pip
pip install --upgrade pip

# 安裝專案依賴
pip install -r requirements.txt

# 安裝開發依賴（如果有）
pip install -r requirements-dev.txt

# 驗證安裝
pip list
```

### 4. 設定 VS Code（可選）

安裝推薦的 VS Code 擴充功能：

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-toolsai.jupyter",
    "streetsidesoftware.code-spell-checker",
    "eamodio.gitlens"
  ]
}
```

設定 Python 直譯器：
1. 按 `Cmd+Shift+P` (macOS) 或 `Ctrl+Shift+P` (Windows)
2. 輸入 "Python: Select Interpreter"
3. 選擇 `aiot_env` 環境

---

## 📁 專案結構

```
aiot-lr-crisp-dm/
│
├── .streamlit/                    # Streamlit 配置
│   ├── config.toml                # 主要主題配置
│   └── config_dark.toml           # 深色主題備份
│
├── notebooks/                     # Jupyter 筆記本
│   └── linear_regression_analysis.ipynb
│
├── aiot_env/                      # 虛擬環境（不提交到 Git）
│
├── __pycache__/                   # Python 快取（不提交到 Git）
│
├── .gitignore                     # Git 忽略規則
├── CHANGELOG.md                   # 版本更新日誌
├── CONTRIBUTING.md                # 貢獻指南
├── DEVELOPMENT.md                 # 開發指南（本文件）
├── LICENSE                        # MIT 授權
├── README.md                      # 專案說明
│
├── requirements.txt               # Python 依賴套件
│
├── linear_regression.py           # 核心線性迴歸實作
├── streamlit_app.py              # Streamlit 標準版
├── streamlit_app_simple.py       # Streamlit 簡化版
├── streamlit_app_optimized.py    # Streamlit 優化版（推薦）
├── demo.py                       # 命令列演示
└── theme_switcher.py             # 主題切換工具
```

### 核心模組說明

| 檔案 | 說明 | 用途 |
|------|------|------|
| `linear_regression.py` | 核心線性迴歸實作 | 資料生成、模型訓練、評估 |
| `streamlit_app_optimized.py` | 優化版應用 ⭐ | 完整功能、最佳實踐 |
| `streamlit_app_simple.py` | 簡化版應用 | 快速展示、教學用途 |
| `streamlit_app.py` | 標準版應用 | 原始完整版本 |
| `demo.py` | 命令列演示 | 非互動式展示 |
| `theme_switcher.py` | 主題切換 | 快速切換 Streamlit 主題 |

---

## 🔄 開發工作流程

### 日常開發流程

```bash
# 1. 啟動虛擬環境
source aiot_env/bin/activate  # macOS/Linux
# 或
.\aiot_env\Scripts\activate   # Windows

# 2. 更新專案到最新版本
git pull origin master

# 3. 建立功能分支
git checkout -b feature/你的功能名稱

# 4. 進行開發
# 編輯檔案...

# 5. 測試變更
python linear_regression.py
streamlit run streamlit_app_optimized.py

# 6. 提交變更
git add .
git commit -m "feat: 新增某功能"

# 7. 推送分支
git push origin feature/你的功能名稱

# 8. 在 GitHub 上建立 Pull Request
```

### 功能開發最佳實踐

#### 1. 小步提交

```bash
# 好的範例：小而專注的提交
git commit -m "feat: 新增殘差分析函數"
git commit -m "test: 新增殘差分析測試"
git commit -m "docs: 更新殘差分析文件"

# 避免：大而雜亂的提交
git commit -m "更新很多東西"
```

#### 2. 定期同步

```bash
# 定期從主分支同步
git checkout master
git pull origin master
git checkout feature/你的功能名稱
git merge master
```

#### 3. 程式碼審查

- 提交 PR 前自我審查
- 確保所有測試通過
- 更新相關文件
- 回應審查意見

---

## 🧪 測試指南

### 手動測試

#### 測試核心功能

```bash
# 測試線性迴歸核心
python linear_regression.py

# 預期輸出：
# - 資料生成成功
# - 模型訓練完成
# - 評估指標顯示
# - 視覺化圖表
```

#### 測試 Streamlit 應用

```bash
# 測試優化版
streamlit run streamlit_app_optimized.py

# 測試項目：
# ✓ 應用正常啟動
# ✓ 參數調整功能正常
# ✓ 圖表即時更新
# ✓ 所有分頁可切換
# ✓ 下載功能正常
# ✓ 無錯誤訊息
```

#### 測試主題切換

```bash
# 測試所有主題
python theme_switcher.py light
streamlit run streamlit_app_optimized.py
# 檢查淺色主題

python theme_switcher.py dark
streamlit run streamlit_app_optimized.py
# 檢查深色主題

python theme_switcher.py blue
streamlit run streamlit_app_optimized.py
# 檢查藍色主題
```

### 自動化測試（未來計劃）

```bash
# 執行所有測試
pytest

# 執行特定測試
pytest tests/test_linear_regression.py

# 查看測試覆蓋率
pytest --cov=. --cov-report=html

# 開啟覆蓋率報告
open htmlcov/index.html  # macOS
```

### 測試檢查清單

執行以下測試確保品質：

- [ ] 核心功能測試
  - [ ] 資料生成正確
  - [ ] 模型訓練成功
  - [ ] 評估指標準確
  - [ ] 視覺化正常顯示

- [ ] Streamlit 應用測試
  - [ ] 應用啟動無錯誤
  - [ ] 所有參數滑桿正常
  - [ ] 圖表互動正常
  - [ ] 分頁切換流暢
  - [ ] 下載功能正常

- [ ] 相容性測試
  - [ ] Python 3.8+ 相容
  - [ ] macOS 測試
  - [ ] Linux 測試
  - [ ] Windows 測試

- [ ] 效能測試
  - [ ] 大資料集 (500+ 點) 測試
  - [ ] 回應時間 < 2 秒
  - [ ] 記憶體使用合理

---

## 🚀 部署指南

### 本地部署

```bash
# 啟動應用（推薦優化版）
streamlit run streamlit_app_optimized.py

# 啟動簡化版
streamlit run streamlit_app_simple.py

# 啟動標準版
streamlit run streamlit_app.py

# 指定端口
streamlit run streamlit_app_optimized.py --server.port 8080

# 開啟瀏覽器（預設會自動開啟）
# 預設 http://localhost:8501
```

### 本地網路分享

如果需要在區域網路內分享應用：

```bash
# 允許外部連接
streamlit run streamlit_app_optimized.py --server.address 0.0.0.0

# 取得本機 IP
# macOS/Linux
ifconfig | grep "inet "
# Windows
ipconfig

# 其他裝置可透過 http://你的IP:8501 訪問
```

### 效能優化

#### Streamlit 快取

```python
import streamlit as st

# 快取資料載入
@st.cache_data
def load_data():
    return pd.read_csv('data.csv')

# 快取資源（例如模型）
@st.cache_resource
def load_model():
    return LinearRegression()
```

### 程式碼優化

```python
# ❌ 避免：迴圈中重複計算
for i in range(n):
    result = expensive_function()
    process(result)

# ✅ 推薦：快取結果
result = expensive_function()
for i in range(n):
    process(result)
```

---

## ❓ 常見問題

### Q1: 虛擬環境啟動失敗

**A**: 
```bash
# macOS/Linux: 賦予執行權限
chmod +x aiot_env/bin/activate

# Windows: 調整 PowerShell 執行政策
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q2: 套件安裝失敗

**A**:
```bash
# 升級 pip
pip install --upgrade pip

# 清除快取
pip cache purge

# 重新安裝
pip install -r requirements.txt --no-cache-dir
```

### Q3: Streamlit 無法啟動

**A**:
```bash
# 檢查 Streamlit 版本
streamlit --version

# 重新安裝 Streamlit
pip uninstall streamlit
pip install streamlit

# 檢查端口佔用
lsof -i :8501  # macOS/Linux
netstat -ano | findstr :8501  # Windows
```

### Q4: Git 推送失敗

**A**:
```bash
# 檢查遠端連接
git remote -v

# 更新遠端 URL
git remote set-url origin https://github.com/Alice-LTY/aiot-lr-crisp-dm.git

# 強制推送（謹慎使用）
git push -f origin master
```

### Q5: Jupyter Notebook 無法開啟

**A**:
```bash
# 安裝 Jupyter
pip install jupyter

# 啟動 Jupyter Lab
jupyter lab

# 或啟動 Jupyter Notebook
jupyter notebook
```

---

## 📚 參考資源

### 官方文件

- [Python 文件](https://docs.python.org/3/)
- [Streamlit 文件](https://docs.streamlit.io/)
- [scikit-learn 文件](https://scikit-learn.org/stable/)
- [NumPy 文件](https://numpy.org/doc/)
- [Pandas 文件](https://pandas.pydata.org/docs/)
- [Plotly 文件](https://plotly.com/python/)

### 學習資源

- [CRISP-DM 方法論](https://www.datascience-pm.com/crisp-dm-2/)
- [線性迴歸原理](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)
- [Streamlit 教學](https://docs.streamlit.io/library/get-started)
- [Git 教學](https://git-scm.com/book/zh-tw/v2)

### 社群資源

- [Stack Overflow](https://stackoverflow.com/questions/tagged/python)
- [Streamlit Forum](https://discuss.streamlit.io/)
- [GitHub Discussions](https://github.com/Alice-LTY/aiot-lr-crisp-dm/discussions)

---

## 🤝 獲得幫助

如需協助，請：

1. 查閱本文件
2. 搜尋 [GitHub Issues](https://github.com/Alice-LTY/aiot-lr-crisp-dm/issues)
3. 開啟新的 Issue
4. 參考 [CONTRIBUTING.md](CONTRIBUTING.md)

---

<div align="center">

**📅 最後更新: 2025年1月4日**

Happy Coding! 🎉

</div>
