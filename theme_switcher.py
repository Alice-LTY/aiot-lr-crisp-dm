#!/usr/bin/env python3
"""
Streamlit 主題切換工具 (Clean Code 版本)
Theme Switcher for Streamlit (Clean Code Version)

遵循 Clean Code 原則：
- argparse 參數化
- logging 取代 print
- 安全寫檔（備份機制）
- 完整錯誤處理
- PEP8 規範
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional
import shutil
from datetime import datetime


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    設定 logging 配置
    
    Args:
        verbose: 是否顯示詳細日誌
        
    Returns:
        配置好的 logger
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


# 主題配置模板
THEME_CONFIGS: Dict[str, str] = {
    'light': '''[theme]
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
toolbarMode = "auto"''',
    
    'dark': '''[theme]
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
toolbarMode = "auto"''',
    
    'blue': '''[theme]
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
}


def create_backup(file_path: Path, logger: logging.Logger) -> Optional[Path]:
    """
    建立配置檔案的備份
    
    Args:
        file_path: 要備份的檔案路徑
        logger: logger 實例
        
    Returns:
        備份檔案路徑，如果原檔案不存在則回傳 None
    """
    if not file_path.exists():
        logger.debug(f"檔案不存在，跳過備份: {file_path}")
        return None
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.with_suffix(f'.toml.backup_{timestamp}')
        shutil.copy2(file_path, backup_path)
        logger.info(f"已建立備份: {backup_path}")
        return backup_path
    except Exception as e:
        logger.warning(f"備份建立失敗: {e}")
        return None


def write_config_safely(
    config_content: str,
    file_path: Path,
    overwrite: bool,
    logger: logging.Logger
) -> bool:
    """
    安全地寫入配置檔案
    
    Args:
        config_content: 配置內容
        file_path: 目標檔案路徑
        overwrite: 是否覆蓋現有檔案
        logger: logger 實例
        
    Returns:
        是否成功寫入
    """
    try:
        # 檢查檔案是否存在
        if file_path.exists() and not overwrite:
            logger.error(f"檔案已存在: {file_path}")
            logger.error("使用 --overwrite 參數強制覆蓋")
            return False
        
        # 建立備份
        if file_path.exists():
            create_backup(file_path, logger)
        
        # 確保目錄存在
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 寫入檔案
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        logger.info(f"配置檔案已寫入: {file_path}")
        return True
        
    except PermissionError:
        logger.error(f"沒有寫入權限: {file_path}")
        return False
    except Exception as e:
        logger.error(f"寫入檔案失敗: {e}")
        return False


def switch_theme(
    theme_name: str,
    config_dir: Path,
    overwrite: bool,
    logger: logging.Logger
) -> bool:
    """
    切換 Streamlit 主題
    
    Args:
        theme_name: 主題名稱 (light, dark, blue)
        config_dir: 配置目錄路徑
        overwrite: 是否覆蓋現有檔案
        logger: logger 實例
        
    Returns:
        是否成功切換
    """
    if theme_name not in THEME_CONFIGS:
        logger.error(f"不支援的主題: {theme_name}")
        logger.error(f"可用主題: {', '.join(THEME_CONFIGS.keys())}")
        return False
    
    config_file = config_dir / 'config.toml'
    config_content = THEME_CONFIGS[theme_name]
    
    logger.info(f"準備切換到 {theme_name} 主題...")
    
    success = write_config_safely(config_content, config_file, overwrite, logger)
    
    if success:
        # 顯示對應的 emoji
        emoji_map = {'light': '☀️', 'dark': '🌙', 'blue': '💙'}
        logger.info(f"{emoji_map.get(theme_name, '✅')} 已切換到 {theme_name} 主題")
        logger.info(f"🔄 請重新啟動 Streamlit 應用以看到變更")
        logger.info(f"   streamlit run streamlit_app_optimized.py")
        return True
    else:
        logger.error(f"❌ 主題切換失敗")
        return False


def parse_arguments() -> argparse.Namespace:
    """
    解析命令列參數
    
    Returns:
        解析後的參數命名空間
    """
    parser = argparse.ArgumentParser(
        description='Streamlit 主題切換工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
可用主題:
  light  - 淺色主題 (經典藍色主題，適合日間使用)
  dark   - 深色主題 (科技藍色主題，適合夜間使用)
  blue   - 藍色專業主題 (淺藍背景，適合商業報告)

範例:
  # 切換到淺色主題
  %(prog)s light
  
  # 切換到深色主題並覆蓋現有配置
  %(prog)s dark --overwrite
  
  # 切換到藍色主題並指定配置目錄
  %(prog)s blue --config-dir /path/to/.streamlit --overwrite
  
  # 顯示詳細日誌
  %(prog)s light --verbose --overwrite
        """
    )
    
    parser.add_argument(
        'theme',
        choices=['light', 'dark', 'blue'],
        help='選擇主題'
    )
    parser.add_argument(
        '--config-dir',
        type=Path,
        default=None,
        help='Streamlit 配置目錄路徑 (預設: ./.streamlit)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='覆蓋現有配置檔案（需要此選項才能覆蓋）'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='顯示詳細日誌訊息'
    )
    
    return parser.parse_args()


def main() -> int:
    """
    主函數
    
    Returns:
        退出碼：0 表示成功，非 0 表示失敗
    """
    try:
        # 解析參數
        args = parse_arguments()
        
        # 設定 logging
        logger = setup_logging(args.verbose)
        
        # 確定配置目錄
        if args.config_dir:
            config_dir = args.config_dir
        else:
            # 預設使用當前目錄的 .streamlit
            current_dir = Path.cwd()
            config_dir = current_dir / '.streamlit'
        
        logger.debug(f"配置目錄: {config_dir}")
        
        # 切換主題
        success = switch_theme(
            theme_name=args.theme,
            config_dir=config_dir,
            overwrite=args.overwrite,
            logger=logger
        )
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n操作已取消")
        return 130
    except Exception as e:
        print(f"\n錯誤: {e}", file=sys.stderr)
        if args.verbose if 'args' in locals() else False:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
