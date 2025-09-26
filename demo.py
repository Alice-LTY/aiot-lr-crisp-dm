#!/usr/bin/env python3
"""
演示腳本：簡單線性迴歸 CRISP-DM
Demo Script: Simple Linear Regression with CRISP-DM
"""

import sys
import os

# 添加當前目錄到路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from linear_regression import SimpleLinearRegressionCRISPDM

def run_demo():
    """執行演示"""
    print("🚀 簡單線性迴歸 CRISP-DM 演示")
    print("=" * 50)
    
    # 建立分析器實例
    analyzer = SimpleLinearRegressionCRISPDM()
    
    # 不同情境的測試
    scenarios = [
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
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*20} 情境 {i}: {scenario['name']} {'='*20}")
        
        # 執行分析
        model, metrics = analyzer.run_complete_analysis(**scenario['params'])
        
        # 儲存結果
        result = {
            "scenario": scenario['name'],
            "params": scenario['params'],
            "metrics": metrics
        }
        results.append(result)
        
        print(f"\n📊 {scenario['name']} 結果摘要:")
        print(f"R² = {metrics['r2']:.4f}")
        print(f"RMSE = {metrics['rmse']:.4f}")
        
        # 等待用戶確認（可選）
        if i < len(scenarios):
            input("\n按 Enter 繼續下一個情境...")
    
    # 比較所有情境
    print(f"\n{'='*20} 情境比較 {'='*20}")
    print(f"{'情境':<15} {'R²':<8} {'RMSE':<8} {'噪音':<8}")
    print("-" * 45)
    
    for result in results:
        print(f"{result['scenario']:<15} "
              f"{result['metrics']['r2']:<8.3f} "
              f"{result['metrics']['rmse']:<8.3f} "
              f"{result['params']['noise_std']:<8.1f}")
    
    print(f"\n✅ 演示完成！")
    print(f"\n💡 接下來您可以：")
    print(f"1. 執行 Streamlit 應用: streamlit run streamlit_app.py")
    print(f"2. 開啟 Jupyter 筆記本: jupyter notebook notebooks/linear_regression_analysis.ipynb")
    print(f"3. 修改參數重新執行: python demo.py")

if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\n👋 演示已中斷")
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        print("請檢查是否已安裝所有必要套件")
