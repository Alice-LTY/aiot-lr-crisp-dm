# 簡單線性迴歸 CRISP-DM 專案

這個專案展示如何使用 CRISP-DM 方法論來建立簡單線性迴歸模型。

## 專案結構
```
AIOT/
├── requirements.txt          # 套件依賴
├── linear_regression.py      # 主要實作
├── streamlit_app.py         # Streamlit 網頁應用
├── demo.py                  # 演示腳本
├── theme_switcher.py        # 主題切換工具
├── deployment_check.py      # 部署檢查工具
├── .gitignore              # Git 忽略檔案
├── README.md               # 專案說明
├── .streamlit/
│   ├── config.toml          # Streamlit 主題配置
│   └── config_dark.toml     # 深色主題備份
└── notebooks/
    └── linear_regression_analysis.ipynb  # Jupyter 分析筆記本
```

## 安裝套件
```bash
pip install -r requirements.txt
```

## 執行方式

### 1. 基本分析
```bash
python linear_regression.py
```

### 2. Streamlit 網頁應用
```bash
streamlit run streamlit_app.py
```

#### 主題配置
使用內建的主題切換工具：

```bash
# 淺色主題（預設）
python theme_switcher.py light

# 深色主題
python theme_switcher.py dark

# 藍色專業主題
python theme_switcher.py blue
```

#### 可用主題
1. **淺色主題 (light)**: 
   - 主要色: `#1f77b4` (經典藍)
   - 背景色: `#ffffff` (白色)
   - 文字色: `#262730` (深灰)

2. **深色主題 (dark)**: 
   - 主要色: `#00d4ff` (科技藍)
   - 背景色: `#0e1117` (深黑)
   - 文字色: `#fafafa` (淺白)

3. **藍色專業主題 (blue)**:
   - 主要色: `#2E86AB` (專業藍)
   - 背景色: `#F8FBFF` (淺藍背景)
   - 文字色: `#1A365D` (深藍文字)

## 🚀 Streamlit Cloud 部署

### 步驟 0: 部署前檢查（推薦）

使用內建的部署檢查工具：
```bash
python deployment_check.py
```

此工具會檢查：
- ✅ 必要檔案是否存在
- ✅ streamlit_app.py 語法是否正確
- ✅ requirements.txt 是否完整
- ✅ Git 儲存庫狀態

### 步驟 1: 準備 GitHub 儲存庫

1. 將專案推送到 GitHub：
```bash
git init
git add .
git commit -m "Initial commit: Simple Linear Regression CRISP-DM"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

### 步驟 2: 在 Streamlit Cloud 建立應用

1. 前往 [Streamlit Cloud](https://streamlit.io/cloud)
2. 使用 GitHub 帳號登入
3. 點擊 **"New app"**
4. 填寫部署設定：

#### 🔧 部署配置
- **Repository**: `<your-username>/<your-repo-name>`
- **Branch**: `main` (或您選擇的分支)
- **Main file path**: `streamlit_app.py`
- **App URL**: 將自動生成為 `https://<appname>-<username>.streamlit.app`

#### 📝 範例配置
```
Repository: john-doe/linear-regression-crisp-dm
Branch: main
Main file path: streamlit_app.py
App URL: https://linear-regression-crisp-dm-john-doe.streamlit.app
```

### 步驟 3: 高級設定（可選）

在 GitHub 儲存庫中建立 `.streamlit/secrets.toml` 來存放敏感資訊：
```toml
# .streamlit/secrets.toml
[database]
host = "your-database-host"
username = "your-username"
password = "your-password"
```

### 步驟 4: 部署狀態檢查

部署過程中，Streamlit Cloud 會：
1. ✅ 克隆您的 GitHub 儲存庫
2. ✅ 安裝 `requirements.txt` 中的套件
3. ✅ 執行 `streamlit_app.py`
4. ✅ 提供公開 URL

### 🔄 自動部署

每當您推送新的 commit 到指定分支時，應用會自動重新部署：

```bash
git add .
git commit -m "Update: improved visualizations"
git push origin main
```

### 🌐 分享應用

部署成功後，您可以：
- 📱 分享 URL 給任何人使用
- 🔗 嵌入到網站或部落格
- 📊 在簡報中展示

### ⚠️ 注意事項

1. **免費帳戶限制**：
   - 最多 3 個公開應用
   - 共享資源（CPU/記憶體）
   - 應用閒置後會休眠

2. **檔案大小限制**：
   - 單一檔案最大 100MB
   - 整個儲存庫最大 1GB

3. **隱私設定**：
   - 免費應用為公開可見
   - 付費帳戶可設定私人應用

### 3. Jupyter 筆記本分析
開啟 `notebooks/linear_regression_analysis.ipynb`

## CRISP-DM 流程

1. **Business Understanding (業務理解)**
   - 建立簡單線性迴歸模型來理解變數間的線性關係
   - 評估不同噪音水平對模型效能的影響

2. **Data Understanding (資料理解)**
   - 使用人工生成的資料 y = ax + b + noise
   - 可調整參數：斜率(a)、截距(b)、噪音大小、資料點數量

3. **Data Preparation (資料準備)**
   - 使用 numpy 生成合成資料
   - 建立 pandas DataFrame 進行資料管理

4. **Modeling (建模)**
   - 使用 scikit-learn LinearRegression
   - 訓練模型並取得係數

5. **Evaluation (評估)**
   - R² 決定係數
   - 均方誤差 (MSE)
   - 視覺化比較

6. **Deployment (部署)**
   - Streamlit 互動式網頁應用
   - 即時參數調整和結果視覺化
   - Streamlit Cloud 線上部署
