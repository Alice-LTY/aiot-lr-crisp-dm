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
