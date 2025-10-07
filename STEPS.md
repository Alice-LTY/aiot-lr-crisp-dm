# 🔄 Vibe Coding 開發步驟 (Development Steps)

## 簡單線性迴歸 CRISP-DM 專案開發流程

本文件詳細記錄使用 Vibe Coding（AI 輔助開發）方法完成此專案的完整步驟。

---

## 📋 目錄

- [開發準備](#開發準備)
- [階段 1: 專案初始化](#階段-1-專案初始化)
- [階段 2: 環境設定](#階段-2-環境設定)
- [階段 3: 核心功能開發](#階段-3-核心功能開發)
- [階段 4: UI/UX 設計](#階段-4-uiux-設計)
- [階段 5: 版本控制](#階段-5-版本控制)
- [階段 6: 問題修正](#階段-6-問題修正)
- [階段 7: 程式碼優化](#階段-7-程式碼優化)
- [階段 8: 文件完善](#階段-8-文件完善)
- [Vibe Coding 技巧](#vibe-coding-技巧)
- [常見問題與解決](#常見問題與解決)

---
## 🛠️ 開發準備

### 工具清單

| 工具 | 用途 | 必要性 |
|------|------|--------|
| AI 助手 (GitHub Copilot/ChatGPT) | 程式碼生成 | ✅ 必要 |
| VS Code / IDE | 程式編輯 | ✅ 必要 |
| Python 3.8+ | 執行環境 | ✅ 必要 |
| Git | 版本控制 | ✅ 必要 |
| Terminal | 命令列操作 | ✅ 必要 |
| GitHub | 程式碼託管 | ⭐ 推薦 |

### 心態準備

1. **信任 AI**: 但要驗證輸出
2. **清晰表達**: 精確描述需求
3. **漸進開發**: 小步前進，逐步完善
4. **保持好奇**: 理解 AI 生成的程式碼
5. **及時反饋**: 發現問題立即回報

---

## 階段 1: 專案初始化

### 🎯 目標
建立專案基礎結構，包含檔案架構、核心程式、配置檔案。

### 💬 對話範例

**👤 人類提示 (Prompt)**:
```
練習 1: 簡單線性迴歸 (Simple Linear Regression) with CRISP-DM

請建立一個完整的專案，包含：
1. 使用 CRISP-DM 方法論
2. 生成合成資料 (y = ax + b + noise)
3. 訓練線性迴歸模型
4. 評估模型效能（R², MSE）
5. 視覺化結果
6. Streamlit 互動式應用
```

**🤖 AI 回應**:
- 建立專案資料夾結構
- 生成 `linear_regression.py`（核心實作）
- 生成 `streamlit_app.py`（網頁應用）
- 生成 `requirements.txt`（依賴套件）
- 生成 `README.md`（專案說明）
- 生成 `.gitignore`（Git 忽略規則）

### ✅ 完成檢查清單

- [ ] 專案資料夾已建立
- [ ] 核心 Python 檔案存在
- [ ] requirements.txt 包含所有依賴
- [ ] README.md 說明清楚
- [ ] .gitignore 配置正確

### 📊 時間消耗
- 傳統方法: ~2 小時
- Vibe Coding: ~5 分鐘
- **節省: 95%** ⬇️

---

## 階段 2: 環境設定

### 🎯 目標
建立 Python 虛擬環境，安裝所有依賴套件。

### 💬 對話範例

**👤 人類提示**:
```
透過 terminal 建立一個虛擬環境，名稱為 aiot_env
```

**🤖 AI 回應**:
```bash
# AI 執行以下命令
python3 -m venv aiot_env
source aiot_env/bin/activate  # macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
```

### 🔍 驗證步驟

```bash
# 1. 確認虛擬環境啟動
which python
# 應該顯示: /path/to/aiot_env/bin/python

# 2. 確認套件已安裝
pip list
# 應該看到: numpy, pandas, scikit-learn, streamlit, plotly 等

# 3. 測試匯入
python -c "import numpy, pandas, sklearn, streamlit, plotly; print('All packages imported successfully!')"
```

### ✅ 完成檢查清單

- [ ] 虛擬環境已建立
- [ ] 虛擬環境已啟動
- [ ] 所有套件已安裝
- [ ] 匯入測試通過

### 📊 時間消耗
- 傳統方法: ~30 分鐘（查文件、除錯）
- Vibe Coding: ~3 分鐘
- **節省: 90%** ⬇️

---

## 階段 3: 核心功能開發

### 🎯 目標
實作線性迴歸核心功能，包含資料生成、模型訓練、評估。

### 💬 對話範例

**👤 人類提示**:
```
請確認 linear_regression.py 包含以下功能：
1. generate_data() - 生成合成資料
2. train_model() - 訓練線性迴歸模型
3. calculate_metrics() - 計算 R², MSE, RMSE
4. visualize_results() - 視覺化結果
5. main() - 主程式
```

**🤖 AI 回應**:
- 檢查現有程式碼
- 補充缺少的功能
- 優化函數設計
- 新增類型提示
- 撰寫 Docstrings

### 🧪 測試步驟

```bash
# 測試核心功能
python linear_regression.py

# 預期輸出:
# - 資料生成成功訊息
# - 模型訓練完成訊息
# - R², MSE, RMSE 數值
# - 視覺化圖表顯示
```

### ✅ 完成檢查清單

- [ ] 資料生成功能正常
- [ ] 模型訓練成功
- [ ] 評估指標正確
- [ ] 視覺化圖表顯示
- [ ] 無錯誤訊息

### 📊 時間消耗
- 傳統方法: ~8 小時
- Vibe Coding: ~30 分鐘
- **節省: 94%** ⬇️

---

## 階段 4: UI/UX 設計

### 🎯 目標
建立 Streamlit 互動式網頁應用，配置專業主題。

### 💬 對話範例 - 主題配置

**👤 人類提示**:
```
為我的 Streamlit 專案建立 .streamlit/config.toml，設定專業主題：
- primaryColor: 藍色
- backgroundColor: 白色
- textColor: 深灰色
讓應用看起來更專業
```

**🤖 AI 回應**:
- 建立 `.streamlit/` 目錄
- 生成 `config.toml` 主題配置
- 提供 3 種主題選項（淺色、深色、藍色）
- 建立 `theme_switcher.py` 工具

### 💬 對話範例 - 功能擴充

**👤 人類提示**:
```
在 Streamlit 應用中新增：
1. 側邊欄參數調整（斜率、截距、噪音）
2. 即時圖表更新
3. 指標卡片顯示 R², MSE, RMSE
4. 參數誤差比較
```

**🤖 AI 回應**:
- 修改 `streamlit_app.py`
- 新增側邊欄滑桿
- 實作即時更新邏輯
- 設計指標卡片布局
- 新增參數比較區塊

### 🧪 測試步驟

```bash
# 啟動 Streamlit 應用
streamlit run streamlit_app.py

# 測試項目:
# 1. 調整斜率滑桿 → 圖表即時更新 ✓
# 2. 調整噪音標準差 → R² 數值改變 ✓
# 3. 增加資料點數量 → 圖表更新 ✓
# 4. 切換主題 → 色彩方案改變 ✓
```

### ✅ 完成檢查清單

- [ ] Streamlit 應用啟動成功
- [ ] 所有參數滑桿正常
- [ ] 圖表即時更新
- [ ] 指標正確顯示
- [ ] 主題切換正常

### 📊 時間消耗
- 傳統方法: ~4 小時
- Vibe Coding: ~15 分鐘
- **節省: 94%** ⬇️

---

## 階段 5: 版本控制

### 🎯 目標
初始化 Git 儲存庫，配置 .gitignore，連接 GitHub。

### 💬 對話範例 - Git 設定

**👤 人類提示**:
```
我想要初始化 Git 並推送到 GitHub，
但不確定 .gitignore 應該忽略哪些檔案
```

**🤖 AI 回應**:
```bash
# AI 建議並執行:

# 1. 初始化 Git
git init

# 2. 配置 .gitignore（AI 生成完整規則）
# 包含: __pycache__, *.pyc, aiot_env/, .DS_Store, etc.

# 3. 首次提交
git add .
git commit -m "Initial commit: Simple Linear Regression CRISP-DM project"

# 4. 連接 GitHub
git remote add origin https://github.com/Alice-LTY/aiot-lr-crisp-dm.git
git branch -M master
git push -u origin master
```

### 💬 對話範例 - .gitignore 優化

**👤 人類提示**:
```
我覺得一些測試的 py 也可以 ignore
deployment_check.py 需要 ignore 吧
```

**🤖 AI 回應**:
- 更新 .gitignore
- 新增 `test_*.py`
- 新增 `deployment_check.py`
- 新增 `scratch_*.py`

### ✅ 完成檢查清單

- [ ] Git 儲存庫已初始化
- [ ] .gitignore 配置完整
- [ ] 已連接 GitHub 遠端
- [ ] 首次推送成功
- [ ] 提交訊息清晰

### 📊 時間消耗
- 傳統方法: ~1 小時
- Vibe Coding: ~10 分鐘
- **節省: 83%** ⬇️

---

## 階段 6: 問題修正

### 🎯 目標
修正部署錯誤、套件衝突、程式碼 bug。

### 💬 對話範例 - 套件版本問題

**👤 人類提示**:
```
執行時出現錯誤:
ERROR: Could not find a version that satisfies the requirement numpy==1.24.3
(from versions: ..., 1.26.0, 1.26.1)
```

**🤖 AI 分析與修正**:
```
問題原因: numpy==1.24.3 不支援 Python 3.12

解決方案:
1. 移除 requirements.txt 中的版本限制
2. 改用彈性版本: numpy>=1.24 或直接 numpy
3. 重新安裝套件
```

### 💬 對話範例 - 匯入錯誤

**👤 人類提示**:
```
Oh no. Error running app.
ModuleNotFoundError: No module named 'plotly.express'
```

**🤖 AI 修正**:
- 檢查 `streamlit_app.py`
- 新增缺少的匯入: `import plotly.express as px`
- 驗證其他必要的匯入
- 重新測試應用

### 🔍 除錯流程

```bash
# 1. 檢查語法錯誤
python -m py_compile streamlit_app.py

# 2. 測試匯入
python -c "import streamlit, plotly, numpy, pandas, sklearn"

# 3. 執行應用
streamlit run streamlit_app.py

# 4. 查看錯誤日誌
# 複製錯誤訊息給 AI 分析
```

### ✅ 完成檢查清單

- [ ] 所有錯誤已修正
- [ ] 應用正常運行
- [ ] 無警告訊息
- [ ] 功能測試通過

### 📊 時間消耗
- 傳統方法: ~3 小時（查文件、Stack Overflow）
- Vibe Coding: ~20 分鐘
- **節省: 89%** ⬇️

---

## 階段 7: 程式碼優化

### 🎯 目標
在不抄襲範例的前提下，優化程式碼品質和功能。

### 💬 對話範例

**👤 人類提示**:
```
read sample/ 這是專業教授透過 vibe coding 完成這項作業的樣子
與 sample 比較後，你認為現在我們的專案有哪些可以改進？

注意: 優化程式碼但不可以抄襲 sample
```

**🤖 AI 分析與建議**:
```
分析結果：
sample/ 有以下優點：
1. 殘差分析圖表
2. 更多評估指標（MAE）
3. 參數比較表格
4. 教育導向的說明文字

我們的改進方向（原創設計）：
1. 新增殘差分析（不同於 sample 的實作方式）
2. 新增噪音分佈直方圖（sample 沒有的功能）
3. 新增相對誤差計算（原創功能）
4. 新增分頁式 UI（不同於 sample 的布局）
5. 新增 Type Hints 和完整 Docstrings
```

### 🎨 優化實作

**👤 人類提示**:
```
直接優化程式碼但不可以抄襲 sample
```

**🤖 AI 執行**:
- 建立新檔案 `streamlit_app_optimized.py`
- 新增原創功能:
  - `create_residual_plot()` - 殘差分析圖
  - `create_noise_distribution()` - 噪音分佈直方圖
  - `display_parameter_comparison()` - 參數比較表格
  - `get_r2_interpretation()` - R² 評級系統
- 優化現有程式碼:
  - 新增 Type Hints
  - 完整 Docstrings
  - 錯誤處理
  - 程式碼註解
- 設計分頁式 UI（4 個 tabs）
- 新增教育內容（FAQ, CRISP-DM 說明）

### ✅ 完成檢查清單

- [ ] 新增功能都是原創設計
- [ ] Type Hints 完整
- [ ] Docstrings 詳細
- [ ] 程式碼可讀性高
- [ ] 無重複程式碼
- [ ] 測試通過

### 📊 時間消耗
- 傳統方法: ~6 小時
- Vibe Coding: ~40 分鐘
- **節省: 89%** ⬇️

---

## 階段 8: 文件完善

### 🎯 目標
建立完整的專案文件系統。

### 💬 對話範例

**👤 人類提示**:
```
包括文件 Readme 等各種 md 更新

需要:
1. 更新 README.md（更專業、更完整）
2. 建立 CHANGELOG.md（版本歷史）
3. 建立 CONTRIBUTING.md（貢獻指南）
4. 建立 DEVELOPMENT.md（開發指南）
5. 建立 LICENSE（MIT 授權）
6. 更新 .gitignore（更完整）
```

**🤖 AI 執行**:
- 全面重寫 README.md（~300 行）
- 建立 CHANGELOG.md（~250 行）
- 建立 CONTRIBUTING.md（~450 行）
- 建立 DEVELOPMENT.md（~550 行）
- 建立 LICENSE（MIT）
- 更新 .gitignore（~150 行）

### 📝 文件檢查清單

#### README.md ✅
- [ ] 專案簡介清楚
- [ ] 主要特色列表
- [ ] 安裝步驟詳細
- [ ] 使用方式完整
- [ ] CRISP-DM 流程說明
- [ ] 效能指標解釋
- [ ] 技術棧清單
- [ ] 貢獻指南連結

#### CHANGELOG.md ✅
- [ ] 版本號遵循 SemVer
- [ ] 變更類型分類（Added, Fixed, Changed）
- [ ] 時間軸清晰
- [ ] 每個版本有說明

#### CONTRIBUTING.md ✅
- [ ] 行為準則
- [ ] 貢獻類型說明
- [ ] 開發流程指引
- [ ] Commit 規範
- [ ] 程式碼風格指南
- [ ] PR 模板

#### DEVELOPMENT.md ✅
- [ ] 環境需求
- [ ] 環境設定步驟
- [ ] 專案結構說明
- [ ] 開發工作流程
- [ ] 測試指南
- [ ] 常見問題解答

### ✅ 完成檢查清單

- [ ] 所有文件已建立
- [ ] 文件內容完整
- [ ] 排版美觀
- [ ] 連結正確
- [ ] 無錯字

### 📊 時間消耗
- 傳統方法: ~4 小時
- Vibe Coding: ~15 分鐘
- **節省: 94%** ⬇️

---

## 🎓 Vibe Coding 技巧

### 1. 有效的 Prompt 設計

#### ❌ 不好的 Prompt
```
幫我寫程式
做一個網站
修正錯誤
```

#### ✅ 好的 Prompt
```
建立一個 Python 函數 generate_data()，
功能是生成線性迴歸的合成資料，
參數包括 slope, intercept, noise_std, n_points，
回傳 pandas DataFrame，
包含 x, y, y_true, noise 四個欄位
```

### 2. Prompt 模板

#### 功能開發模板
```
請實作 [功能名稱]，需求如下：
1. [具體需求 1]
2. [具體需求 2]
3. [具體需求 3]

輸入: [輸入格式]
輸出: [輸出格式]
限制: [特殊限制]
```

#### 問題修正模板
```
遇到以下錯誤:
[錯誤訊息複製貼上]

環境資訊:
- Python 版本: [版本]
- 相關套件: [套件名稱和版本]
- 作業系統: [OS]

請分析原因並提供解決方案
```

#### 程式碼審查模板
```
請審查以下程式碼:
[程式碼]

檢查項目:
1. 是否有 bug
2. 效能是否可優化
3. 程式碼可讀性
4. 是否遵循最佳實踐
5. 是否需要新增註解或文件
```

### 3. 漸進式開發策略

```
階段 1: 最小可行產品 (MVP)
├── 核心功能 ✓
└── 基本測試 ✓

階段 2: 功能擴充
├── 新增視覺化 ✓
├── 新增互動式 UI ✓
└── 主題配置 ✓

階段 3: 品質提升
├── 程式碼優化 ✓
├── 新增 Type Hints ✓
├── 完善文件 ✓
└── 錯誤處理 ✓

階段 4: 文件完善
├── README.md ✓
├── CHANGELOG.md ✓
├── CONTRIBUTING.md ✓
└── 其他文件 ✓
```

### 4. 驗證與測試流程

```bash
# 1. 語法檢查
python -m py_compile your_file.py

# 2. 匯入測試
python -c "import your_module"

# 3. 功能測試
python your_file.py

# 4. 互動測試（Streamlit）
streamlit run streamlit_app.py

# 5. 回報結果給 AI
```

### 5. 有效的反饋技巧

#### ❌ 無效反饋
```
不行
有問題
不是我要的
```

#### ✅ 有效反饋
```
功能正常運作，但有以下建議:
1. 圖表標題可以改為中文
2. 希望新增下載按鈕
3. 參數範圍可以擴大到 -10 到 10

錯誤訊息:
[詳細錯誤訊息]

預期行為:
[描述你期望的結果]

實際行為:
[描述實際發生的事]
```

---

## ❓ 常見問題與解決

### Q1: AI 生成的程式碼有錯怎麼辦？

**A**: 
1. 複製完整錯誤訊息給 AI
2. 描述預期行為 vs 實際行為
3. 提供環境資訊（Python 版本、套件版本）
4. AI 會分析並提供修正方案

### Q2: 如何確保不過度依賴 AI？

**A**:
1. 閱讀並理解 AI 生成的程式碼
2. 嘗試自己先思考解決方案
3. 將 Vibe Coding 視為學習工具
4. 定期複習和整理學到的知識
5. 逐漸減少 AI 協助，增加自主開發

### Q3: Prompt 不夠清楚，AI 理解錯誤怎麼辦？

**A**:
1. 重新描述需求，更具體明確
2. 提供範例或參考
3. 分解大需求為多個小需求
4. 使用 Prompt 模板
5. 及時糾正 AI 的誤解

### Q4: 如何判斷 AI 生成的程式碼品質？

**A**:
檢查項目:
- [ ] 程式碼能正常執行
- [ ] 邏輯正確，符合需求
- [ ] 程式碼可讀性高
- [ ] 有適當的註解
- [ ] 遵循 Python 最佳實踐
- [ ] 沒有明顯的效能問題

### Q5: 遇到複雜問題，AI 無法解決？

**A**:
1. 將問題拆解為更小的子問題
2. 先解決簡單部分，再處理複雜部分
3. 搜尋相關文件或範例
4. 在社群論壇尋求幫助
5. 考慮調整需求或尋找替代方案
---

## 💡 最佳實踐總結

### Do's ✅

- ✅ 清晰描述需求
- ✅ 提供完整的錯誤訊息
- ✅ 閱讀並理解 AI 生成的程式碼
- ✅ 漸進式開發，小步前進
- ✅ 及時測試驗證
- ✅ 保持學習態度
- ✅ 給予明確的反饋

### Don'ts ❌

- ❌ 模糊的需求描述
- ❌ 不閱讀就直接使用程式碼
- ❌ 一次要求太多功能
- ❌ 不測試就推送到生產環境
- ❌ 完全依賴 AI
- ❌ 忽略程式碼品質
- ❌ 不提供反饋

---

## 📚 參考資源

### Vibe Coding 相關
- [GitHub Copilot 文件](https://docs.github.com/en/copilot)
- [ChatGPT 最佳實踐](https://platform.openai.com/docs/guides/prompt-engineering)
- [AI 輔助程式設計](https://www.oreilly.com/radar/how-to-use-ai-assisted-coding/)

### Python 開發
- [Python 官方文件](https://docs.python.org/3/)
- [PEP 8 風格指南](https://www.python.org/dev/peps/pep-0008/)
- [Real Python 教學](https://realpython.com/)

### Streamlit 開發
- [Streamlit 官方文件](https://docs.streamlit.io/)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Streamlit Community](https://discuss.streamlit.io/)

---

<div align="center">

**📅 最後更新: 2025年1月4日**

**🎉 Happy Vibe Coding! 🎉**

</div>
