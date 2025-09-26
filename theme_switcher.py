#!/usr/bin/env python3
"""
Streamlit 主題切換工具
Theme Switcher for Streamlit
"""

import os
import shutil
import argparse

def switch_theme(theme_name):
    """切換 Streamlit 主題"""
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(current_dir, '.streamlit')
    config_file = os.path.join(config_dir, 'config.toml')
    
    if theme_name == 'light':
        # 使用淺色主題
        light_config = '''[theme]
# 主要顏色 - 用於按鈕、連結、選中狀態等
primaryColor = "#1f77b4"

# 背景顏色 - 主要背景
backgroundColor = "#ffffff"

# 次要背景顏色 - 側邊欄背景
secondaryBackgroundColor = "#f0f2f6"

# 文字顏色
textColor = "#262730"

# 字體 - 可選：sans serif, serif, monospace
font = "sans serif"

[server]
# 設定埠號
port = 8501

# 設定主機
headless = false

# 自動重載
runOnSave = true

# 檔案監控器類型
fileWatcherType = "auto"

[browser]
# 自動開啟瀏覽器
gatherUsageStats = false

[client]
# 顯示錯誤詳細資訊
showErrorDetails = true

# 工具列模式
toolbarMode = "auto"'''
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(light_config)
        print("✅ 已切換到淺色主題")
        
    elif theme_name == 'dark':
        # 使用深色主題
        dark_config = '''[theme]
# 主要顏色 - 科技藍
primaryColor = "#00d4ff"

# 背景顏色 - 深色背景
backgroundColor = "#0e1117"

# 次要背景顏色 - 側邊欄深色背景
secondaryBackgroundColor = "#262730"

# 文字顏色 - 淺色文字
textColor = "#fafafa"

# 字體
font = "sans serif"

[server]
port = 8501
headless = false
runOnSave = true
fileWatcherType = "auto"

[browser]
gatherUsageStats = false

[client]
showErrorDetails = true
toolbarMode = "auto"'''
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(dark_config)
        print("🌙 已切換到深色主題")
        
    elif theme_name == 'blue':
        # 藍色專業主題
        blue_config = '''[theme]
# 主要顏色 - 專業藍
primaryColor = "#2E86AB"

# 背景顏色 - 淺藍背景
backgroundColor = "#F8FBFF"

# 次要背景顏色 - 側邊欄背景
secondaryBackgroundColor = "#E3F2FD"

# 文字顏色
textColor = "#1A365D"

# 字體
font = "sans serif"

[server]
port = 8501
headless = false
runOnSave = true
fileWatcherType = "auto"

[browser]
gatherUsageStats = false

[client]
showErrorDetails = true
toolbarMode = "auto"'''
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(blue_config)
        print("💙 已切換到藍色專業主題")
        
    else:
        print("❌ 不支援的主題。可用主題: light, dark, blue")
        return
    
    print(f"🔄 請重新啟動 Streamlit 應用以看到變更: streamlit run streamlit_app.py")

def main():
    parser = argparse.ArgumentParser(description='切換 Streamlit 主題')
    parser.add_argument('theme', choices=['light', 'dark', 'blue'], 
                       help='選擇主題: light (淺色), dark (深色), blue (藍色專業)')
    
    args = parser.parse_args()
    switch_theme(args.theme)

if __name__ == "__main__":
    main()
