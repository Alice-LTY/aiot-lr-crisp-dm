#!/usr/bin/env python3
"""
演示腳本：簡單線性迴歸 CRISP-DM (Clean Code 版本)
Demo Script: Simple Linear Regression with CRISP-DM (Clean Code Version)

遵循 Clean Code 原則：
- argparse 參數化
- logging 取代 print
- 完整錯誤處理
- 清楚的 CLI 介面
- PEP8 規範
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

try:
    from linear_regression_clean import SimpleLinearRegressionCRISPDM
except ImportError:
    # 如果 clean 版本不存在，嘗試使用原版
    from linear_regression import SimpleLinearRegressionCRISPDM


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


def parse_arguments() -> argparse.Namespace:
    """
    解析命令列參數
    
    Returns:
        解析後的參數命名空間
    """
    parser = argparse.ArgumentParser(
        description='簡單線性迴歸 CRISP-DM 演示腳本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 執行單一情境
  %(prog)s --slope 2.5 --intercept 1.0 --noise 0.5 --points 100
  
  # 執行預設的多情境比較
  %(prog)s --compare
  
  # 執行並儲存視覺化結果
  %(prog)s --visualize --save --output-dir ./results
  
  # 顯示詳細日誌
  %(prog)s --verbose --compare
        """
    )
    
    # 互斥模式：單一情境 vs 多情境比較
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--compare',
        action='store_true',
        help='執行多情境比較模式（低/中/高噪音）'
    )
    
    # 單一情境參數
    scenario_group = parser.add_argument_group('單一情境參數')
    scenario_group.add_argument(
        '--slope',
        type=float,
        default=2.5,
        help='真實斜率 (預設: 2.5)'
    )
    scenario_group.add_argument(
        '--intercept',
        type=float,
        default=1.0,
        help='真實截距 (預設: 1.0)'
    )
    scenario_group.add_argument(
        '--noise',
        type=float,
        default=0.5,
        help='噪音標準差 (預設: 0.5)'
    )
    scenario_group.add_argument(
        '--points',
        type=int,
        default=100,
        help='資料點數量 (預設: 100)'
    )
    
    # 視覺化選項
    viz_group = parser.add_argument_group('視覺化選項')
    viz_group.add_argument(
        '--visualize',
        action='store_true',
        help='顯示視覺化圖表'
    )
    viz_group.add_argument(
        '--save',
        action='store_true',
        help='儲存視覺化圖表到檔案'
    )
    viz_group.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./output'),
        help='圖表輸出目錄 (預設: ./output)'
    )
    
    # 其他選項
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='顯示詳細日誌訊息'
    )
    parser.add_argument(
        '--no-pause',
        action='store_true',
        help='情境間不暫停等待（比較模式）'
    )
    
    return parser.parse_args()


def run_single_scenario(
    analyzer: SimpleLinearRegressionCRISPDM,
    slope: float,
    intercept: float,
    noise_std: float,
    n_points: int,
    visualize: bool,
    save_plots: bool,
    output_dir: Path,
    logger: logging.Logger
) -> Tuple[Any, Dict[str, float]]:
    """
    執行單一情境分析
    
    Args:
        analyzer: 分析器實例
        slope: 斜率
        intercept: 截距
        noise_std: 噪音標準差
        n_points: 資料點數量
        visualize: 是否視覺化
        save_plots: 是否儲存圖表
        output_dir: 輸出目錄
        logger: logger 實例
        
    Returns:
        (model, metrics): 模型和評估指標
    """
    logger.info(f"執行情境：斜率={slope}, 截距={intercept}, 噪音={noise_std}, 點數={n_points}")
    
    try:
        # 準備輸出目錄
        if save_plots:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"圖表將儲存至: {output_dir}")
        
        # 執行分析
        model, metrics = analyzer.run_complete_analysis(
            slope=slope,
            intercept=intercept,
            noise_std=noise_std,
            n_points=n_points,
            visualize=visualize,
            save_plots=save_plots,
            output_dir=output_dir if save_plots else None
        )
        
        logger.info(f"✅ 情境完成！R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
        return model, metrics
        
    except Exception as e:
        logger.error(f"❌ 情境執行失敗: {e}")
        raise


def run_comparison_scenarios(
    analyzer: SimpleLinearRegressionCRISPDM,
    visualize: bool,
    save_plots: bool,
    output_dir: Path,
    no_pause: bool,
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    執行多情境比較
    
    Args:
        analyzer: 分析器實例
        visualize: 是否視覺化
        save_plots: 是否儲存圖表
        output_dir: 輸出目錄
        no_pause: 是否跳過暫停
        logger: logger 實例
        
    Returns:
        結果列表
    """
    # 定義測試情境
    scenarios: List[Dict[str, Any]] = [
        {
            "name": "低噪音情境",
            "params": {"slope": 2.0, "intercept": 1.0, "noise_std": 0.2, "n_points": 100}
        },
        {
            "name": "中等噪音情境",
            "params": {"slope": -1.5, "intercept": 2.0, "noise_std": 0.8, "n_points": 150}
        },
        {
            "name": "高噪音情境",
            "params": {"slope": 3.0, "intercept": -0.5, "noise_std": 1.5, "n_points": 200}
        }
    ]
    
    results: List[Dict[str, Any]] = []
    
    logger.info("🚀 簡單線性迴歸 CRISP-DM 演示")
    logger.info("=" * 50)
    
    for i, scenario in enumerate(scenarios, 1):
        logger.info(f"\n{'='*20} 情境 {i}: {scenario['name']} {'='*20}")
        
        try:
            # 為每個情境建立子目錄
            scenario_output_dir = output_dir / scenario['name'] if save_plots else None
            if scenario_output_dir:
                scenario_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 執行分析
            model, metrics = analyzer.run_complete_analysis(
                **scenario['params'],
                visualize=visualize,
                save_plots=save_plots,
                output_dir=scenario_output_dir
            )
            
            # 儲存結果
            result = {
                "scenario": scenario['name'],
                "params": scenario['params'],
                "metrics": metrics
            }
            results.append(result)
            
            logger.info(f"\n📊 {scenario['name']} 結果摘要:")
            logger.info(f"R² = {metrics['r2']:.4f}")
            logger.info(f"RMSE = {metrics['rmse']:.4f}")
            
            # 等待用戶確認（可選）
            if i < len(scenarios) and not no_pause:
                try:
                    input("\n按 Enter 繼續下一個情境...")
                except EOFError:
                    # 處理非互動環境
                    pass
                    
        except Exception as e:
            logger.error(f"情境 {i} 執行失敗: {e}")
            continue
    
    # 比較所有情境
    if results:
        logger.info(f"\n{'='*20} 情境比較 {'='*20}")
        logger.info(f"{'情境':<15} {'R²':<8} {'RMSE':<8} {'噪音':<8}")
        logger.info("-" * 45)
        
        for result in results:
            logger.info(
                f"{result['scenario']:<15} "
                f"{result['metrics']['r2']:<8.3f} "
                f"{result['metrics']['rmse']:<8.3f} "
                f"{result['params']['noise_std']:<8.1f}"
            )
    
    return results


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
        
        # 建立分析器實例
        logger.info("初始化分析器...")
        analyzer = SimpleLinearRegressionCRISPDM(verbose=args.verbose)
        
        # 執行對應模式
        if args.compare:
            # 多情境比較模式
            results = run_comparison_scenarios(
                analyzer=analyzer,
                visualize=args.visualize,
                save_plots=args.save,
                output_dir=args.output_dir,
                no_pause=args.no_pause,
                logger=logger
            )
            
            if not results:
                logger.error("所有情境都失敗了")
                return 1
                
        else:
            # 單一情境模式
            model, metrics = run_single_scenario(
                analyzer=analyzer,
                slope=args.slope,
                intercept=args.intercept,
                noise_std=args.noise,
                n_points=args.points,
                visualize=args.visualize,
                save_plots=args.save,
                output_dir=args.output_dir,
                logger=logger
            )
        
        logger.info(f"\n✅ 演示完成！")
        logger.info(f"\n💡 接下來您可以：")
        logger.info(f"1. 執行 Streamlit 應用: streamlit run streamlit_app_optimized.py")
        logger.info(f"2. 修改參數重新執行: python {sys.argv[0]} --help")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n👋 演示已中斷")
        return 130
    except Exception as e:
        print(f"\n❌ 錯誤: {e}", file=sys.stderr)
        if args.verbose if 'args' in locals() else False:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
