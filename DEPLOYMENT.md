# 🚀 部署指南

## 本地部署（固定網址）

### 方式 1: 本機訪問（預設）

```bash
streamlit run streamlit_app_optimized.py
```

**固定網址**: `http://localhost:8501`

### 方式 2: 區域網路訪問

修改 `.streamlit/config.toml`:

```toml
[server]
address = "0.0.0.0"  # 改為 0.0.0.0
port = 8501
```

然後執行：

```bash
streamlit run streamlit_app_optimized.py
```

**固定網址**: 
- 本機: `http://localhost:8501`
- 區域網路: `http://你的IP:8501` (例如: `http://192.168.1.100:8501`)

查看你的 IP:
```bash
# macOS/Linux
ifconfig | grep "inet "

# 或
ipconfig getifaddr en0
```

### 方式 3: 使用特定端口

```bash
streamlit run streamlit_app_optimized.py --server.port 8502
```

**固定網址**: `http://localhost:8502`

## 線上部署（公開固定網址）

### Streamlit Cloud 部署（推薦，免費）

1. **推送到 GitHub**
   ```bash
   git add .
   git commit -m "準備部署"
   git push origin master
   ```

2. **登入 Streamlit Cloud**
   - 前往 [share.streamlit.io](https://share.streamlit.io)
   - 使用 GitHub 帳號登入

3. **部署應用**
   - 點擊 "New app"
   - 選擇倉庫: `Alice-LTY/aiot-lr-crisp-dm`
   - 分支: `master`
   - 主檔案: `streamlit_app_optimized.py`
   - 點擊 "Deploy"

4. **獲得固定網址**
   ```
   https://aiot-lr-crisp-dm.streamlit.app
   ```
   或自訂子域名:
   ```
   https://alice-aiot-lr.streamlit.app
   ```

### 其他部署選項

#### Heroku
```bash
# 需要 Procfile 和 setup.sh
echo "web: sh setup.sh && streamlit run streamlit_app_optimized.py" > Procfile
```

#### Docker
```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app_optimized.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Nginx 反向代理（自有伺服器）
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

## 環境變數配置

創建 `.streamlit/secrets.toml` (不要提交到 Git):

```toml
# API keys 或其他敏感資訊
[secrets]
api_key = "your-api-key"
```

在程式中使用:
```python
import streamlit as st
api_key = st.secrets["secrets"]["api_key"]
```

## 效能優化

### 快取配置

在程式中添加:
```python
@st.cache_data
def load_data():
    # 資料載入邏輯
    pass

@st.cache_resource
def load_model():
    # 模型載入邏輯
    pass
```

### 記憶體限制

在 `.streamlit/config.toml` 添加:
```toml
[server]
maxUploadSize = 200  # MB
maxMessageSize = 200  # MB
```

## 監控與日誌

### Streamlit Cloud 內建監控
- CPU 使用率
- 記憶體使用率
- 請求數量
- 錯誤日誌

### 自訂日誌
```python
import logging

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info("應用啟動")
```

## 安全性建議

1. **不要在程式碼中硬編碼敏感資訊**
   - 使用 `secrets.toml`
   - 使用環境變數

2. **啟用 XSRF 保護**
   ```toml
   [server]
   enableXsrfProtection = true
   ```

3. **限制上傳大小**
   ```toml
   [server]
   maxUploadSize = 200
   ```

4. **使用 HTTPS**
   - Streamlit Cloud 自動提供
   - 自有伺服器需配置 SSL 證書

## 故障排除

### 端口被佔用
```bash
# 查看佔用 8501 的進程
lsof -i :8501

# 或使用其他端口
streamlit run streamlit_app_optimized.py --server.port 8502
```

### 模組找不到
```bash
# 確認虛擬環境已啟動
source aiot_env/bin/activate

# 重新安裝依賴
pip install -r requirements.txt
```

### Streamlit Cloud 部署失敗
- 檢查 `requirements.txt` 是否完整
- 確認 Python 版本相容性
- 查看部署日誌找出錯誤

## 網址管理

### 本地開發
| 環境 | 網址 | 用途 |
|------|------|------|
| 預設 | `http://localhost:8501` | 本機開發 |
| 網路 | `http://你的IP:8501` | 區域網路測試 |
| 自訂端口 | `http://localhost:8502` | 避免端口衝突 |

### 線上部署
| 平台 | 網址格式 | 特點 |
|------|---------|------|
| Streamlit Cloud | `https://app名稱.streamlit.app` | 免費、易用 |
| Heroku | `https://app名稱.herokuapp.com` | 免費層有限制 |
| 自有伺服器 | `https://your-domain.com` | 完全控制 |

## 持續整合/持續部署 (CI/CD)

### GitHub Actions 自動部署

創建 `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Streamlit Cloud

on:
  push:
    branches: [ master ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.12'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest tests/
```

---

**更新日期**: 2025-01-04  
**維護者**: Alice LTY
