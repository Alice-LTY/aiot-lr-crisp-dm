#!/usr/bin/env python3
"""
簡單線性迴歸 CRISP-DM 實作 (Clean Code 版本)
Simple Linear Regression with CRISP-DM Methodology (Clean Code Version)

遵循 Clean Code 原則：
- 模組化設計
- 完整 type hints
- Logging 取代 print
- 錯誤處理
- PEP8 規範
"""

import logging
import sys
from typing import Tuple, Dict, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


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
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


class SimpleLinearRegressionCRISPDM:
    """
    簡單線性迴歸 CRISP-DM 實作類別
    
    Attributes:
        data: 存儲資料的 DataFrame
        model: 訓練好的線性迴歸模型
        true_slope: 真實斜率
        true_intercept: 真實截距
        predicted_slope: 預測斜率
        predicted_intercept: 預測截距
        logger: logging 實例
    """
    
    def __init__(self, verbose: bool = False) -> None:
        """
        初始化
        
        Args:
            verbose: 是否顯示詳細日誌
        """
        self.data: Optional[pd.DataFrame] = None
        self.model: Optional[LinearRegression] = None
        self.true_slope: Optional[float] = None
        self.true_intercept: Optional[float] = None
        self.predicted_slope: Optional[float] = None
        self.predicted_intercept: Optional[float] = None
        self.logger: logging.Logger = setup_logging(verbose)
        
    # 1. Business Understanding (業務理解)
    def business_understanding(self) -> None:
        """
        業務理解階段
        定義問題和目標
        """
        self.logger.info("=== 1. Business Understanding (業務理解) ===")
        self.logger.info("問題定義：建立簡單線性迴歸模型來理解兩個變數之間的線性關係")
        self.logger.info("目標：")
        self.logger.info("- 生成合成資料 y = ax + b + noise")
        self.logger.info("- 訓練線性迴歸模型")
        self.logger.info("- 評估模型效能")
        self.logger.info("- 比較真實參數與預測參數")
        self.logger.info("- 視覺化結果")
        self.logger.debug("預期輸入：斜率(a)、截距(b)、噪音大小、資料點數量")
        self.logger.debug("預期輸出：迴歸模型、效能指標、視覺化圖表")
        self.logger.info("-" * 50)
    
    # 2. Data Understanding (資料理解)
    def data_understanding(
        self, 
        slope: float = 2.5, 
        intercept: float = 1.0, 
        noise_std: float = 0.5, 
        n_points: int = 100, 
        x_range: Tuple[float, float] = (-5.0, 5.0),
        random_seed: int = 42
    ) -> None:
        """
        資料理解階段
        生成和探索資料
        
        Args:
            slope: 真實斜率
            intercept: 真實截距
            noise_std: 噪音標準差
            n_points: 資料點數量
            x_range: x 值範圍 (最小值, 最大值)
            random_seed: 隨機種子，確保可重現性
            
        Raises:
            ValueError: 參數不合法時拋出
        """
        # 參數驗證
        if n_points <= 0:
            raise ValueError(f"資料點數量必須大於 0，目前為 {n_points}")
        if noise_std < 0:
            raise ValueError(f"噪音標準差不能為負數，目前為 {noise_std}")
        if x_range[0] >= x_range[1]:
            raise ValueError(f"x_range 最小值必須小於最大值，目前為 {x_range}")
        
        self.logger.info("\n=== 2. Data Understanding (資料理解) ===")
        
        # 儲存真實參數
        self.true_slope = slope
        self.true_intercept = intercept
        
        try:
            # 生成資料
            np.random.seed(random_seed)
            x: np.ndarray = np.linspace(x_range[0], x_range[1], n_points)
            noise: np.ndarray = np.random.normal(0, noise_std, n_points)
            y_true: np.ndarray = slope * x + intercept
            y: np.ndarray = y_true + noise
            
            # 建立 DataFrame
            self.data = pd.DataFrame({
                'x': x,
                'y': y,
                'y_true': y_true,
                'noise': noise
            })
            
            self.logger.info(f"資料來源：人工生成 (synthetic data)")
            self.logger.info(f"真實模型：y = {slope}x + {intercept} + noise")
            self.logger.info(f"資料點數量：{n_points}")
            self.logger.info(f"噪音標準差：{noise_std}")
            self.logger.info(f"x 範圍：{x_range}")
            
            # 基本統計描述
            self.logger.debug("\n資料統計描述：")
            self.logger.debug(f"\n{self.data.describe()}")
            
            self.logger.info("-" * 50)
            
        except Exception as e:
            self.logger.error(f"資料生成失敗: {e}")
            raise
        
    def visualize_data(self, save_path: Optional[Path] = None) -> None:
        """
        視覺化原始資料
        
        Args:
            save_path: 如果提供，將圖表儲存到此路徑
            
        Raises:
            ValueError: 如果資料尚未生成
        """
        if self.data is None:
            raise ValueError("資料尚未生成，請先執行 data_understanding()")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 散點圖
            axes[0, 0].scatter(self.data['x'], self.data['y'], alpha=0.6, color='blue')
            axes[0, 0].plot(self.data['x'], self.data['y_true'], 'r-', linewidth=2, label='真實關係')
            axes[0, 0].set_xlabel('x')
            axes[0, 0].set_ylabel('y')
            axes[0, 0].set_title('原始資料散點圖')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # x 的分布
            axes[0, 1].hist(self.data['x'], bins=20, alpha=0.7, color='skyblue')
            axes[0, 1].set_xlabel('x')
            axes[0, 1].set_ylabel('頻率')
            axes[0, 1].set_title('x 變數分布')
            axes[0, 1].grid(True, alpha=0.3)
            
            # y 的分布
            axes[1, 0].hist(self.data['y'], bins=20, alpha=0.7, color='lightgreen')
            axes[1, 0].set_xlabel('y')
            axes[1, 0].set_ylabel('頻率')
            axes[1, 0].set_title('y 變數分布')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 噪音分布
            axes[1, 1].hist(self.data['noise'], bins=20, alpha=0.7, color='orange')
            axes[1, 1].set_xlabel('噪音')
            axes[1, 1].set_ylabel('頻率')
            axes[1, 1].set_title('噪音分布')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"圖表已儲存至: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"視覺化失敗: {e}")
            raise
    
    # 3. Data Preparation (資料準備)
    def data_preparation(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        資料準備階段
        準備建模所需的資料格式
        
        Returns:
            (X, y): 特徵矩陣和目標變數
            
        Raises:
            ValueError: 如果資料尚未生成
        """
        if self.data is None:
            raise ValueError("資料尚未生成，請先執行 data_understanding()")
        
        self.logger.info("\n=== 3. Data Preparation (資料準備) ===")
        
        try:
            # 檢查缺失值
            missing_values = self.data.isnull().sum()
            self.logger.info("缺失值檢查：")
            self.logger.info(f"\n{missing_values}")
            
            # 準備特徵和目標變數
            X: pd.DataFrame = self.data[['x']]
            y: pd.Series = self.data['y']
            
            self.logger.info(f"\n特徵矩陣 X 形狀：{X.shape}")
            self.logger.info(f"目標變數 y 形狀：{y.shape}")
            
            # 資料範圍
            self.logger.debug(f"\nx 範圍：[{X['x'].min():.2f}, {X['x'].max():.2f}]")
            self.logger.debug(f"y 範圍：[{y.min():.2f}, {y.max():.2f}]")
            
            self.logger.info("資料準備完成，準備進行建模...")
            self.logger.info("-" * 50)
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"資料準備失敗: {e}")
            raise
    
    # 4. Modeling (建模)
    def modeling(self, X: pd.DataFrame, y: pd.Series) -> LinearRegression:
        """
        建模階段
        訓練線性迴歸模型
        
        Args:
            X: 特徵矩陣
            y: 目標變數
            
        Returns:
            訓練好的線性迴歸模型
            
        Raises:
            ValueError: 如果真實參數尚未設定
        """
        if self.true_slope is None or self.true_intercept is None:
            raise ValueError("真實參數尚未設定，請先執行 data_understanding()")
        
        self.logger.info("\n=== 4. Modeling (建模) ===")
        
        try:
            # 建立和訓練模型
            self.model = LinearRegression()
            self.model.fit(X, y)
            
            # 取得模型參數
            self.predicted_slope = float(self.model.coef_[0])
            self.predicted_intercept = float(self.model.intercept_)
            
            self.logger.info("模型選擇：線性迴歸 (Linear Regression)")
            self.logger.info(f"訓練資料大小：{X.shape[0]} 個樣本")
            
            self.logger.info(f"\n模型參數：")
            self.logger.info(f"預測斜率：{self.predicted_slope:.4f}")
            self.logger.info(f"預測截距：{self.predicted_intercept:.4f}")
            
            self.logger.info(f"\n真實參數比較：")
            self.logger.info(f"真實斜率：{self.true_slope:.4f}")
            self.logger.info(f"真實截距：{self.true_intercept:.4f}")
            
            slope_error = abs(self.predicted_slope - self.true_slope)
            intercept_error = abs(self.predicted_intercept - self.true_intercept)
            
            self.logger.info(f"\n參數誤差：")
            self.logger.info(f"斜率誤差：{slope_error:.4f}")
            self.logger.info(f"截距誤差：{intercept_error:.4f}")
            
            self.logger.info("模型訓練完成！")
            self.logger.info("-" * 50)
            
            return self.model
            
        except Exception as e:
            self.logger.error(f"模型訓練失敗: {e}")
            raise
    
    # 5. Evaluation (評估)
    def evaluation(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        評估階段
        評估模型效能
        
        Args:
            X: 特徵矩陣
            y: 目標變數
            
        Returns:
            包含評估指標的字典 {'r2': float, 'mse': float, 'rmse': float}
            
        Raises:
            ValueError: 如果模型尚未訓練
        """
        if self.model is None:
            raise ValueError("模型尚未訓練，請先執行 modeling()")
        
        self.logger.info("\n=== 5. Evaluation (評估) ===")
        
        try:
            # 預測
            y_pred: np.ndarray = self.model.predict(X)
            
            # 計算評估指標
            r2: float = r2_score(y, y_pred)
            mse: float = mean_squared_error(y, y_pred)
            rmse: float = np.sqrt(mse)
            
            self.logger.info(f"評估指標：")
            self.logger.info(f"R² (決定係數)：{r2:.4f}")
            self.logger.info(f"MSE (均方誤差)：{mse:.4f}")
            self.logger.info(f"RMSE (均方根誤差)：{rmse:.4f}")
            
            # 解釋 R²
            self.logger.info(f"\nR² 解釋：")
            r2_percentage = r2 * 100
            if r2 >= 0.9:
                self.logger.info(f"- 模型解釋了 {r2_percentage:.1f}% 的變異，擬合效果非常好")
            elif r2 >= 0.7:
                self.logger.info(f"- 模型解釋了 {r2_percentage:.1f}% 的變異，擬合效果良好")
            elif r2 >= 0.5:
                self.logger.info(f"- 模型解釋了 {r2_percentage:.1f}% 的變異，擬合效果中等")
            else:
                self.logger.info(f"- 模型解釋了 {r2_percentage:.1f}% 的變異，擬合效果較差")
            
            self.logger.info("-" * 50)
            
            return {'r2': r2, 'mse': mse, 'rmse': rmse}
            
        except Exception as e:
            self.logger.error(f"評估失敗: {e}")
            raise
    
    def visualize_results(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        y_pred: np.ndarray,
        save_path: Optional[Path] = None
    ) -> None:
        """
        視覺化模型結果
        
        Args:
            X: 特徵矩陣
            y: 目標變數
            y_pred: 預測值
            save_path: 如果提供，將圖表儲存到此路徑
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            x_values = X['x'].values
            
            # 1. 擬合結果
            axes[0, 0].scatter(x_values, y, alpha=0.6, color='blue', label='實際資料')
            axes[0, 0].plot(x_values, y_pred, 'r-', linewidth=2, label='預測線')
            axes[0, 0].plot(x_values, self.data['y_true'], 'g--', linewidth=2, label='真實關係')
            axes[0, 0].set_xlabel('x')
            axes[0, 0].set_ylabel('y')
            axes[0, 0].set_title('迴歸結果比較')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 殘差圖
            residuals = y - y_pred
            axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='purple')
            axes[0, 1].axhline(y=0, color='red', linestyle='--')
            axes[0, 1].set_xlabel('預測值')
            axes[0, 1].set_ylabel('殘差')
            axes[0, 1].set_title('殘差圖')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 預測 vs 實際
            axes[1, 0].scatter(y, y_pred, alpha=0.6, color='green')
            min_val = min(y.min(), y_pred.min())
            max_val = max(y.max(), y_pred.max())
            axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            axes[1, 0].set_xlabel('實際值')
            axes[1, 0].set_ylabel('預測值')
            axes[1, 0].set_title('預測 vs 實際')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 殘差分布
            axes[1, 1].hist(residuals, bins=20, alpha=0.7, color='orange')
            axes[1, 1].set_xlabel('殘差')
            axes[1, 1].set_ylabel('頻率')
            axes[1, 1].set_title('殘差分布')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"圖表已儲存至: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"結果視覺化失敗: {e}")
            raise
    
    # 6. Deployment (部署)
    def deployment_info(self) -> None:
        """部署階段資訊"""
        self.logger.info("\n=== 6. Deployment (部署) ===")
        self.logger.info("部署選項：")
        self.logger.info("1. Streamlit 網頁應用：")
        self.logger.info("   - 執行：streamlit run streamlit_app_optimized.py")
        self.logger.info("   - 功能：互動式參數調整和即時視覺化")
        self.logger.info("")
        self.logger.info("2. Python 模組：")
        self.logger.info("   - 檔案：linear_regression_clean.py")
        self.logger.info("   - 功能：可重複使用的類別和函數")
        self.logger.info("")
        self.logger.info("3. 部署建議：")
        self.logger.info("   - 使用虛擬環境進行環境隔離")
        self.logger.info("   - 使用 Git 進行版本控制")
        self.logger.info("-" * 50)
    
    def run_complete_analysis(
        self, 
        slope: float = 2.5, 
        intercept: float = 1.0, 
        noise_std: float = 0.5, 
        n_points: int = 100,
        visualize: bool = False,
        save_plots: bool = False,
        output_dir: Optional[Path] = None
    ) -> Tuple[LinearRegression, Dict[str, float]]:
        """
        執行完整的 CRISP-DM 分析流程
        
        Args:
            slope: 真實斜率
            intercept: 真實截距
            noise_std: 噪音標準差
            n_points: 資料點數量
            visualize: 是否顯示視覺化
            save_plots: 是否儲存圖表
            output_dir: 圖表輸出目錄
            
        Returns:
            (model, metrics): 訓練好的模型和評估指標
            
        Raises:
            ValueError: 參數不合法時拋出
        """
        self.logger.info("簡單線性迴歸 CRISP-DM 完整分析")
        self.logger.info("=" * 50)
        
        try:
            # 1. 業務理解
            self.business_understanding()
            
            # 2. 資料理解
            self.data_understanding(slope, intercept, noise_std, n_points)
            
            if visualize or save_plots:
                save_path = output_dir / 'data_visualization.png' if save_plots and output_dir else None
                self.visualize_data(save_path)
            
            # 3. 資料準備
            X, y = self.data_preparation()
            
            # 4. 建模
            model = self.modeling(X, y)
            
            # 5. 評估
            metrics = self.evaluation(X, y)
            
            if visualize or save_plots:
                y_pred = model.predict(X)
                save_path = output_dir / 'results_visualization.png' if save_plots and output_dir else None
                self.visualize_results(X, y, y_pred, save_path)
            
            # 6. 部署
            self.deployment_info()
            
            self.logger.info("\n分析完成！")
            return model, metrics
            
        except Exception as e:
            self.logger.error(f"分析過程發生錯誤: {e}")
            raise


def main() -> int:
    """
    主函數
    
    Returns:
        退出碼：0 表示成功，1 表示失敗
    """
    try:
        # 建立分析實例
        analyzer = SimpleLinearRegressionCRISPDM(verbose=False)
        
        # 執行完整分析
        model, metrics = analyzer.run_complete_analysis(
            slope=2.5,
            intercept=1.0,
            noise_std=0.8,
            n_points=150,
            visualize=True
        )
        
        print(f"\n最終結果摘要：")
        print(f"R² = {metrics['r2']:.4f}")
        print(f"RMSE = {metrics['rmse']:.4f}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n程式已被使用者中斷")
        return 130
    except Exception as e:
        print(f"\n錯誤: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
