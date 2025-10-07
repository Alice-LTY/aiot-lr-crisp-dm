# 簡單線性迴歸 CRISP-D
```

## 🚀 快速開始

### 安裝使用 **CRISP-DM 方法論**建立的簡單線性迴歸模型互動式應用。

## ✨ 主要特色

- 🎯 互動式參數調整（斜率、截距、噪音）
- 📊 多維度視覺化（迴歸線圖、殘差分析、噪音分佈）
- 📈 完整效能評估（R²、MSE、RMSE、MAE）
- 🔍 參數比較分析
- 📚 內建 CRISP-DM 說明和 FAQ

## 📁 專案結構

```
AIOT/
├── requirements.txt              # 套件依賴
├── linear_regression.py          # 核心實作
├── streamlit_app_optimized.py    # 主應用 (推薦)
└── README.md                     # 說明文件
```

## � 快速開始

### 安裝

```bash
# 1. 克隆專案
git clone https://github.com/Alice-LTY/aiot-lr-crisp-dm.git
cd aiot-lr-crisp-dm

# 2. 建立虛擬環境
python3 -m venv aiot_env
source aiot_env/bin/activate

# 3. 安裝依賴
pip install -r requirements.txt
```

### 執行

```bash
streamlit run streamlit_app_optimized.py
```

應用將在瀏覽器中開啟 `http://localhost:8501`

## 🎨 功能亮點

### 互動式視覺化
- 📈 即時更新的迴歸線圖
- 🔍 殘差分析圖（檢查模型假設）
- 📊 噪音分佈直方圖

### 效能評估指標
- **R²** (決定係數): 模型解釋變異的比例
- **MSE/RMSE**: 預測誤差大小
- **MAE**: 平均絕對誤差
- **殘差統計**: 均值和標準差

### 參數分析
- 真實參數 vs 預測參數對比表
- 絕對誤差和相對誤差計算
- 自動參數估計品質評估

## 📊 CRISP-DM 方法論

本專案實踐完整的 **CRISP-DM** 資料科學流程：

| 階段 | 說明 | 本專案實作 |
|-----|------|-----------|
| 🎯 **業務理解** | 確定專案目標 | 建立線性迴歸模型，探索噪音對效能的影響 |
| 📚 **資料理解** | 探索資料特性 | 合成資料：`y = ax + b + noise` |
| 🔧 **資料準備** | 清理和轉換 | NumPy/Pandas 生成與管理資料 |
| 🤖 **建模** | 訓練模型 | scikit-learn LinearRegression (OLS) |
| ✅ **評估** | 效能評估 | R²、MSE、RMSE、MAE、殘差分析 |
| 🚀 **部署** | 實際應用 | Streamlit 互動式網頁應用 |

## 📈 效能指標

| 指標 | 說明 | 理想值 |
|------|------|--------|
| **R²** | 模型解釋變異的比例 | 接近 1 |
| **MSE** | 預測誤差的平方平均 | 接近 0 |
| **RMSE** | MSE 的平方根 | 接近 0 |
| **MAE** | 預測誤差絕對值平均 | 接近 0 |

**R² 評級**: ≥0.9 優秀 | 0.7-0.9 良好 | 0.5-0.7 中等 | <0.5 較差

## 🛠️ 技術棧

- Python 3.12 | NumPy, Pandas | scikit-learn
- Plotly 視覺化 | Streamlit 網頁框架

## 📝 相關文件

- [DEPLOYMENT.md](DEPLOYMENT.md) - 部署指南（固定網址設定）
- [REPORT.md](REPORT.md) - 專案報告與成果
- [STEPS.md](STEPS.md) - Vibe Coding 開發步驟
- [APP.md](APP.md) - 應用程式使用指南
- [CHANGELOG.md](CHANGELOG.md) - 版本更新日誌
---
*Built with Streamlit | 展示 CRISP-DM 方法論*
