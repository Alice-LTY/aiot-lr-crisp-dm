# Streamlit App Deployment Guide

## 🚀 部署到 Streamlit Cloud

### 建議使用的主檔案

**推薦**: `streamlit_app_optimized.py` 
- ✅ 完全自包含，無外部依賴
- ✅ 功能完整，包含所有增強功能
- ✅ 最適合 Streamlit Cloud 部署

### 在 Streamlit Cloud 設定

1. **Main file path**: `streamlit_app_optimized.py`
2. **Python version**: 3.9 或更高
3. **Requirements**: 使用專案中的 `requirements.txt`

### 本地測試

```bash
# 啟動虛擬環境
source aiot_env/bin/activate

# 測試優化版（推薦）
streamlit run streamlit_app_optimized.py

# 或測試重構版（需要 linear_regression.py）
streamlit run streamlit_app.py
```

### 版本比較

| 版本 | 檔案 | 特點 | 部署建議 |
|------|------|------|----------|
| **優化版** | `streamlit_app_optimized.py` | 自包含、功能完整 | ⭐ **推薦用於部署** |
| 重構版 | `streamlit_app.py` | 使用外部模組、Clean Code | 適合本地開發 |
| 標準版 | `streamlit_app_simple.py` | 基本功能 | 適合演示 |

### 故障排除

如果部署失敗，檢查：
1. **主檔案路徑**: 確保指向 `streamlit_app_optimized.py`
2. **依賴套件**: 確保 `requirements.txt` 包含所有必要套件
3. **Python 版本**: 建議使用 Python 3.9+