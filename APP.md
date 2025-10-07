# 📱 應用程式說明 (Application Guide)

## 簡單線性迴歸 CRISP-DM Streamlit 應用

本文件詳細說明三個版本的 Streamlit 應用程式的功能、使用方式和技術細節。

---

## 📋 目錄

- [應用版本概覽](#應用版本概覽)
- [streamlit_app_optimized.py - 優化增強版](#優化增強版-streamlit_app_optimizedpy)
- [streamlit_app_simple.py - 簡化版](#簡化版-streamlit_app_simplepy)
- [streamlit_app.py - 標準版](#標準版-streamlit_apppy)
- [使用指南](#使用指南)
- [常見問題](#常見問題)
- [技術說明](#技術說明)

---

## 🎯 應用版本概覽

### 版本比較表

| 特性 | 優化增強版 ⭐ | 簡化版 | 標準版 |
|------|--------------|--------|--------|
| **檔案名稱** | `streamlit_app_optimized.py` | `streamlit_app_simple.py` | `streamlit_app.py` |
| **推薦度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **程式碼行數** | ~550 行 | ~230 行 | ~250 行 |
| **功能完整度** | 100% | 60% | 80% |
| **視覺化圖表** | 4 種 | 1 種 | 2 種 |
| **評估指標** | 4 個 | 3 個 | 3 個 |
| **UI 組織** | 分頁式 (4 tabs) | 單頁 | 單頁 |
| **教育內容** | 豐富 | 基本 | 中等 |
| **Type Hints** | ✅ 完整 | ❌ 無 | ❌ 無 |
| **Docstrings** | ✅ 詳細 | ⭐ 基本 | ⭐ 基本 |
| **適用場景** | 教學、展示、分析 | 快速演示 | 日常使用 |

### 選擇建議

- **教學展示**: 選擇 **優化增強版** ⭐
- **快速演示**: 選擇 **簡化版**
- **日常學習**: 選擇 **標準版**

---

## 優化增強版: streamlit_app_optimized.py

### 🌟 核心特色

1. **分頁式 UI 設計** (4 個 tabs)
2. **完整的視覺化分析** (迴歸線圖、殘差圖、噪音分佈、參數比較)
3. **豐富的評估指標** (R², MSE, RMSE, MAE + 殘差統計)
4. **教育導向內容** (CRISP-DM 說明、FAQ)
5. **專業的程式碼品質** (Type Hints, Docstrings, 錯誤處理)

### 📊 功能模組

#### 1. 資料生成模組

```python
def generate_data(slope: float, intercept: float, noise_std: float, 
                  n_points: int, x_range: tuple) -> pd.DataFrame
```

**功能**: 生成合成線性迴歸資料

**參數**:
- `slope`: 真實斜率
- `intercept`: 真實截距
- `noise_std`: 噪音標準差
- `n_points`: 資料點數量
- `x_range`: x 值範圍 (min, max)

**回傳**: 包含 x, y, y_true, noise 的 DataFrame

**特點**:
- 使用 `np.random.seed(42)` 確保可重現性
- 支援自訂參數範圍
- 自動計算噪音值

#### 2. 模型訓練模組

```python
def train_model(data: pd.DataFrame) -> tuple
```

**功能**: 訓練線性迴歸模型

**參數**:
- `data`: 包含 x 和 y 欄位的資料框

**回傳**: (model, y_pred) 訓練好的模型和預測值

**特點**:
- 使用 scikit-learn LinearRegression
- 自動提取特徵和目標變數
- 回傳模型和預測值方便後續使用

#### 3. 評估指標模組

```python
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_with_noise: np.ndarray) -> dict
```

**功能**: 計算完整的評估指標

**指標包含**:
- **R²**: 模型解釋變異的比例
- **MSE**: 均方誤差
- **RMSE**: 均方根誤差
- **MAE**: 平均絕對誤差
- **殘差統計**: 殘差平均值和標準差

**特點**:
- 同時使用含噪音和無噪音資料
- 提供完整的殘差資訊
- 便於診斷模型效能

#### 4. 視覺化模組

##### 迴歸線圖

```python
def create_scatter_plot(data: pd.DataFrame, y_pred: np.ndarray, 
                       model: LinearRegression, true_slope: float, 
                       true_intercept: float) -> go.Figure
```

**功能**: 建立散點圖和迴歸線

**包含元素**:
- 🔵 藍色散點: 實際資料點（含噪音）
- 🟢 綠色虛線: 真實線性關係
- 🔴 紅色實線: 模型預測線

**互動功能**:
- Hover 顯示詳細數值
- 縮放和平移
- 圖例切換

##### 殘差分析圖

```python
def create_residual_plot(data: pd.DataFrame, residuals: np.ndarray) -> go.Figure
```

**功能**: 建立殘差分析圖

**用途**: 檢查模型假設是否成立

**理想情況**:
- 殘差隨機分佈在零線附近
- 無明顯的模式或趨勢
- 方差恆定（homoscedasticity）

**診斷問題**:
- 如果有明顯趨勢 → 可能需要非線性模型
- 如果方差不恆定 → 可能需要資料轉換
- 如果有離群值 → 需要檢查資料品質

##### 噪音分佈圖

```python
def create_noise_distribution(noise: np.ndarray) -> go.Figure
```

**功能**: 顯示噪音的分佈

**用途**: 驗證噪音是否服從常態分佈

**理想情況**:
- 呈現鐘形曲線（常態分佈）
- 對稱分佈
- 無明顯偏態

#### 5. 輔助功能模組

##### R² 評級系統

```python
def get_r2_interpretation(r2: float) -> tuple
```

**評級標準**:
- R² ≥ 0.9: 🟢 優秀（非常好的擬合）
- 0.7 ≤ R² < 0.9: 🔵 良好（良好的擬合）
- 0.5 ≤ R² < 0.7: 🟡 中等（中等的擬合）
- R² < 0.5: 🔴 較差（需要改進）

##### 參數比較表格

```python
def display_parameter_comparison(true_slope: float, true_intercept: float,
                                pred_slope: float, pred_intercept: float) -> pd.DataFrame
```

**功能**: 比較真實參數與預測參數

**顯示內容**:
- 真實值
- 預測值
- 絕對誤差
- 相對誤差 (%)

**評估標準**:
- 相對誤差 < 5%: ✅ 非常準確
- 相對誤差 < 10%: ✓ 良好
- 相對誤差 ≥ 10%: ⚠️ 需要改進

### 🎨 UI 設計

#### 側邊欄 (Sidebar)

**參數設定區域**:

1. **線性模型參數**
   - 斜率 (a): -5.0 到 5.0
   - 截距 (b): -10.0 到 10.0

2. **資料參數**
   - 噪音標準差: 0.0 到 3.0
   - 資料點數量: 50 到 500

3. **資料範圍**
   - x 最小值: 數值輸入
   - x 最大值: 數值輸入

**互動方式**:
- 滑桿即時調整
- 數值輸入框精確設定
- 自動驗證輸入合法性

#### 主要內容區域

##### Tab 1: 📈 迴歸結果

**內容**:
- 迴歸線圖（散點 + 真實線 + 預測線）
- 使用說明
- CSV 下載按鈕

**功能**:
- 互動式 Plotly 圖表
- 一鍵下載資料
- Hover 顯示詳細資訊

##### Tab 2: 🔍 殘差分析

**內容**:
- 殘差分析圖
- 殘差統計指標
- 診斷說明

**用途**:
- 檢查模型假設
- 識別資料問題
- 評估預測品質

##### Tab 3: 📊 噪音分佈

**內容**:
- 噪音分佈直方圖
- 噪音標準差顯示
- 常態分佈說明

**用途**:
- 驗證噪音特性
- 理解資料品質
- 學習統計概念

##### Tab 4: 📋 參數比較

**內容**:
- 參數比較表格
- 參數估計品質評估
- 改進建議

**用途**:
- 評估模型準確度
- 比較真實 vs 預測
- 理解估計誤差

#### 擴展式區塊

##### 📚 關於 CRISP-DM 方法論

**內容**:
- 6 個階段說明
- 每個階段的目標
- 本專案的實踐方式

##### ❓ 常見問題 FAQ

**問題包含**:
1. 什麼是 R² 決定係數？
2. MSE、RMSE、MAE 有什麼差別？
3. 為什麼增加噪音會降低 R²？
4. 殘差分析的意義是什麼？

### 🔧 技術特點

#### 程式碼品質

1. **Type Hints**
```python
def generate_data(slope: float, intercept: float, noise_std: float, 
                  n_points: int, x_range: tuple) -> pd.DataFrame:
```

2. **完整 Docstrings**
```python
"""
生成合成線性迴歸資料

參數:
    slope: 真實斜率
    intercept: 真實截距
    noise_std: 噪音標準差
    n_points: 資料點數量
    x_range: x 值範圍 (min, max)

回傳:
    包含 x, y, y_true, noise 的資料框
"""
```

3. **錯誤處理**
```python
try:
    # 主要邏輯
except Exception as e:
    st.error(f"❌ 發生錯誤: {str(e)}")
    st.write("請調整參數後重試，或檢查參數設定是否合理")
```

4. **程式碼組織**
- 清晰的區塊分隔
- 功能模組化
- 單一職責原則

#### 效能優化

- 避免不必要的重複計算
- 使用 NumPy 向量化運算
- Plotly 圖表快取

---

## 簡化版: streamlit_app_simple.py

### 🎯 設計目的

適合快速演示和基礎教學的簡化版本。

### 📊 核心功能

1. **基本參數調整**
   - 斜率、截距、噪音、資料點數
   
2. **迴歸線視覺化**
   - 散點圖 + 真實線 + 預測線
   
3. **基本評估指標**
   - R², MSE, RMSE
   
4. **參數比較**
   - 真實參數 vs 預測參數

### 🎨 UI 設計

- **單頁式布局**: 所有內容在一頁
- **指標卡片**: 3 個並排的 metric 卡片
- **主要圖表**: 迴歸線圖
- **參數比較**: 兩欄對比顯示

### 💡 適用場景

- 快速演示線性迴歸概念
- 課堂教學（時間有限）
- 初學者入門

---

## 標準版: streamlit_app.py

### 🎯 設計目的

平衡功能完整性和簡潔性的標準版本。

### 📊 核心功能

1. **完整參數調整**
2. **雙圖表顯示**
   - 迴歸線圖
   - 殘差圖（基本版）
3. **標準評估指標**
   - R², MSE, RMSE
4. **參數比較和誤差分析**

### 🎨 UI 設計

- **單頁式布局**
- **指標卡片** + **主要圖表**
- **次要圖表**（殘差）
- **參數分析區塊**

### 💡 適用場景

- 日常學習和練習
- 個人專案
- 功能展示

---

## 🚀 使用指南

### 啟動應用

```bash
# 優化增強版（推薦）
streamlit run streamlit_app_optimized.py

# 簡化版
streamlit run streamlit_app_simple.py

# 標準版
streamlit run streamlit_app.py
```

### 基本操作流程

#### 1. 調整參數

**步驟**:
1. 在左側邊欄找到參數滑桿
2. 拖動滑桿調整數值
3. 觀察右側圖表即時更新

**建議實驗**:
- 增加斜率 → 觀察線條變陡
- 增加噪音 → 觀察 R² 下降
- 增加資料點 → 觀察擬合更穩定

#### 2. 分析結果

**查看指標**:
- R²: 數值越接近 1 越好
- MSE/RMSE: 數值越小越好
- MAE: 平均誤差大小

**查看圖表**:
- 藍點離紅線越近 → 預測越準確
- 殘差圖無明顯模式 → 模型假設成立
- 噪音分佈呈鐘形 → 符合常態分佈

#### 3. 比較參數

**參數比較表**:
- 檢查相對誤差
- 評估估計準確度
- 理解噪音的影響

#### 4. 下載資料（優化版）

**步驟**:
1. 切換到「迴歸結果」分頁
2. 點擊「📥 下載資料 (CSV)」按鈕
3. 選擇儲存位置

**用途**:
- 匯出資料進行進一步分析
- 保存實驗結果
- 用於其他工具

### 進階使用

#### 實驗 1: 噪音影響分析

**目標**: 理解噪音對模型效能的影響

**步驟**:
1. 設定固定斜率和截距
2. 噪音標準差從 0.0 逐漸增加到 3.0
3. 觀察 R² 變化趨勢
4. 記錄關鍵數值

**預期結果**:
- 噪音增加 → R² 下降
- 殘差標準差增加
- 參數估計誤差增大

#### 實驗 2: 樣本大小影響

**目標**: 理解樣本大小對估計準確度的影響

**步驟**:
1. 設定固定斜率、截距和噪音
2. 資料點數從 50 逐漸增加到 500
3. 觀察參數估計誤差變化

**預期結果**:
- 樣本增加 → 估計更準確
- 相對誤差減小
- R² 更穩定

#### 實驗 3: 極端情況測試

**測試案例**:

1. **零噪音情況**
   - 噪音標準差 = 0.0
   - 預期: R² = 1.0（完美擬合）

2. **高噪音情況**
   - 噪音標準差 = 3.0
   - 預期: R² 顯著下降

3. **小樣本情況**
   - 資料點數 = 50
   - 預期: 估計不穩定

4. **大樣本情況**
   - 資料點數 = 500
   - 預期: 估計穩定

---

## ❓ 常見問題

### Q1: 為什麼我的 R² 是負數？

**A**: R² 理論上不會是負數，但在極端情況下可能出現：
- 原因: 模型表現比簡單平均值還差
- 解決: 檢查程式碼邏輯，確認使用正確的評估方式

### Q2: 圖表無法顯示怎麼辦？

**A**: 
1. 檢查 Plotly 是否正確安裝: `pip install plotly`
2. 檢查瀏覽器是否支援 JavaScript
3. 嘗試重新整理頁面
4. 查看 Terminal 錯誤訊息

### Q3: 如何修改主題配色？

**A**:
```bash
# 使用主題切換工具
python theme_switcher.py light  # 淺色
python theme_switcher.py dark   # 深色
python theme_switcher.py blue   # 藍色

# 或手動編輯 .streamlit/config.toml
```

### Q4: 可以新增其他迴歸演算法嗎？

**A**: 可以！修改建議：
1. 匯入其他模型: `from sklearn.linear_model import Ridge, Lasso`
2. 新增模型選擇: `model_type = st.sidebar.selectbox(...)`
3. 修改 `train_model()` 函數支援不同模型

### Q5: 如何匯出高解析度圖表？

**A**: Plotly 圖表支援多種匯出格式：
1. 點擊圖表右上角的相機圖示
2. 或使用程式碼: `fig.write_image("figure.png", width=1920, height=1080)`

---

## 🔧 技術說明

### 依賴套件

```python
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
```

### 資料流程

```
1. 使用者調整參數（Sidebar）
   ↓
2. generate_data() 生成資料
   ↓
3. train_model() 訓練模型
   ↓
4. calculate_metrics() 計算指標
   ↓
5. create_plots() 建立圖表
   ↓
6. Streamlit 渲染 UI
```

### 效能考量

- **快取策略**: 考慮使用 `@st.cache_data` 快取資料生成
- **資料大小**: 建議資料點數 ≤ 1000（效能考量）
- **圖表渲染**: Plotly 圖表效能良好，支援大量資料點

---

## 📚 延伸學習

### 推薦閱讀

1. **Streamlit 文件**: https://docs.streamlit.io/
2. **Plotly Python**: https://plotly.com/python/
3. **scikit-learn 線性模型**: https://scikit-learn.org/stable/modules/linear_model.html

### 改進建議

1. **新增功能**:
   - [ ] 多項式迴歸
   - [ ] 模型比較
   - [ ] 批次預測
   - [ ] 資料上傳功能

2. **UI 改進**:
   - [ ] 響應式設計
   - [ ] 更多主題選項
   - [ ] 動畫效果

3. **效能優化**:
   - [ ] 實作快取策略
   - [ ] 優化大資料集處理
   - [ ] 改進圖表載入速度

---

## 📞 技術支援

如有問題，請：

1. 查閱 [DEVELOPMENT.md](DEVELOPMENT.md)
2. 開啟 [GitHub Issue](https://github.com/Alice-LTY/aiot-lr-crisp-dm/issues)
3. 參考 [常見問題](#常見問題)

---

<div align="center">

**📅 最後更新: 2025年1月4日**

**🎉 享受探索線性迴歸的樂趣！**

</div>
