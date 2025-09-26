"""
簡單線性迴歸 CRISP-DM 實作
Simple Linear Regression with CRISP-DM Methodology
"""

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

class SimpleLinearRegressionCRISPDM:
    """
    簡單線性迴歸 CRISP-DM 實作類別
    """
    
    def __init__(self):
        """初始化"""
        self.data = None
        self.model = None
        self.true_slope = None
        self.true_intercept = None
        self.predicted_slope = None
        self.predicted_intercept = None
        
    # 1. Business Understanding (業務理解)
    def business_understanding(self):
        """
        業務理解階段
        定義問題和目標
        """
        print("=== 1. Business Understanding (業務理解) ===")
        print("問題定義：建立簡單線性迴歸模型來理解兩個變數之間的線性關係")
        print("目標：")
        print("- 生成合成資料 y = ax + b + noise")
        print("- 訓練線性迴歸模型")
        print("- 評估模型效能")
        print("- 比較真實參數與預測參數")
        print("- 視覺化結果")
        print("\n預期輸入：斜率(a)、截距(b)、噪音大小、資料點數量")
        print("預期輸出：迴歸模型、效能指標、視覺化圖表")
        print("-" * 50)
    
    # 2. Data Understanding (資料理解)
    def data_understanding(self, slope=2.5, intercept=1.0, noise_std=0.5, n_points=100, x_range=(-5, 5)):
        """
        資料理解階段
        生成和探索資料
        """
        print("\n=== 2. Data Understanding (資料理解) ===")
        
        # 儲存真實參數
        self.true_slope = slope
        self.true_intercept = intercept
        
        # 生成資料
        np.random.seed(42)  # 確保結果可重現
        x = np.linspace(x_range[0], x_range[1], n_points)
        noise = np.random.normal(0, noise_std, n_points)
        y_true = slope * x + intercept
        y = y_true + noise
        
        # 建立 DataFrame
        self.data = pd.DataFrame({
            'x': x,
            'y': y,
            'y_true': y_true,
            'noise': noise
        })
        
        print(f"資料來源：人工生成 (synthetic data)")
        print(f"真實模型：y = {slope}x + {intercept} + noise")
        print(f"資料點數量：{n_points}")
        print(f"噪音標準差：{noise_std}")
        print(f"x 範圍：{x_range}")
        
        # 基本統計描述
        print("\n資料統計描述：")
        print(self.data.describe())
        
        # 視覺化資料
        self.visualize_data()
        print("-" * 50)
        
    def visualize_data(self):
        """視覺化原始資料"""
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
        plt.show()
    
    # 3. Data Preparation (資料準備)
    def data_preparation(self):
        """
        資料準備階段
        準備建模所需的資料格式
        """
        print("\n=== 3. Data Preparation (資料準備) ===")
        
        # 檢查缺失值
        missing_values = self.data.isnull().sum()
        print("缺失值檢查：")
        print(missing_values)
        
        # 準備特徵和目標變數
        X = self.data[['x']]  # 特徵矩陣 (需要是二維)
        y = self.data['y']    # 目標變數
        
        print(f"\n特徵矩陣 X 形狀：{X.shape}")
        print(f"目標變數 y 形狀：{y.shape}")
        
        # 資料範圍
        print(f"\nx 範圍：[{X['x'].min():.2f}, {X['x'].max():.2f}]")
        print(f"y 範圍：[{y.min():.2f}, {y.max():.2f}]")
        
        print("資料準備完成，準備進行建模...")
        print("-" * 50)
        
        return X, y
    
    # 4. Modeling (建模)
    def modeling(self, X, y):
        """
        建模階段
        訓練線性迴歸模型
        """
        print("\n=== 4. Modeling (建模) ===")
        
        # 建立和訓練模型
        self.model = LinearRegression()
        self.model.fit(X, y)
        
        # 取得模型參數
        self.predicted_slope = self.model.coef_[0]
        self.predicted_intercept = self.model.intercept_
        
        print("模型選擇：線性迴歸 (Linear Regression)")
        print(f"訓練資料大小：{X.shape[0]} 個樣本")
        
        print(f"\n模型參數：")
        print(f"預測斜率：{self.predicted_slope:.4f}")
        print(f"預測截距：{self.predicted_intercept:.4f}")
        
        print(f"\n真實參數比較：")
        print(f"真實斜率：{self.true_slope:.4f}")
        print(f"真實截距：{self.true_intercept:.4f}")
        
        print(f"\n參數誤差：")
        print(f"斜率誤差：{abs(self.predicted_slope - self.true_slope):.4f}")
        print(f"截距誤差：{abs(self.predicted_intercept - self.true_intercept):.4f}")
        
        print("模型訓練完成！")
        print("-" * 50)
        
        return self.model
    
    # 5. Evaluation (評估)
    def evaluation(self, X, y):
        """
        評估階段
        評估模型效能
        """
        print("\n=== 5. Evaluation (評估) ===")
        
        # 預測
        y_pred = self.model.predict(X)
        
        # 計算評估指標
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"評估指標：")
        print(f"R² (決定係數)：{r2:.4f}")
        print(f"MSE (均方誤差)：{mse:.4f}")
        print(f"RMSE (均方根誤差)：{rmse:.4f}")
        
        # 解釋 R²
        print(f"\nR² 解釋：")
        if r2 >= 0.9:
            print("- 模型解釋了 {:.1f}% 的變異，擬合效果非常好".format(r2 * 100))
        elif r2 >= 0.7:
            print("- 模型解釋了 {:.1f}% 的變異，擬合效果良好".format(r2 * 100))
        elif r2 >= 0.5:
            print("- 模型解釋了 {:.1f}% 的變異，擬合效果中等".format(r2 * 100))
        else:
            print("- 模型解釋了 {:.1f}% 的變異，擬合效果較差".format(r2 * 100))
        
        # 視覺化評估結果
        self.visualize_results(X, y, y_pred)
        
        print("-" * 50)
        
        return {'r2': r2, 'mse': mse, 'rmse': rmse}
    
    def visualize_results(self, X, y, y_pred):
        """視覺化模型結果"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 擬合結果
        x_values = X['x'].values
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
        plt.show()
    
    # 6. Deployment (部署)
    def deployment_info(self):
        """
        部署階段資訊
        """
        print("\n=== 6. Deployment (部署) ===")
        print("部署選項：")
        print("1. Streamlit 網頁應用：")
        print("   - 執行：streamlit run streamlit_app.py")
        print("   - 功能：互動式參數調整和即時視覺化")
        print()
        print("2. Jupyter 筆記本：")
        print("   - 檔案：notebooks/linear_regression_analysis.ipynb")
        print("   - 功能：詳細分析和實驗")
        print()
        print("3. Python 模組：")
        print("   - 檔案：linear_regression.py")
        print("   - 功能：可重複使用的類別和函數")
        print()
        print("4. 部署建議：")
        print("   - 使用 Streamlit Cloud 進行線上部署")
        print("   - 建立 Docker 容器進行環境隔離")
        print("   - 使用 GitHub Actions 進行 CI/CD")
        print("-" * 50)
    
    def run_complete_analysis(self, slope=2.5, intercept=1.0, noise_std=0.5, n_points=100):
        """
        執行完整的 CRISP-DM 分析流程
        """
        print("簡單線性迴歸 CRISP-DM 完整分析")
        print("=" * 50)
        
        # 1. 業務理解
        self.business_understanding()
        
        # 2. 資料理解
        self.data_understanding(slope, intercept, noise_std, n_points)
        
        # 3. 資料準備
        X, y = self.data_preparation()
        
        # 4. 建模
        model = self.modeling(X, y)
        
        # 5. 評估
        metrics = self.evaluation(X, y)
        
        # 6. 部署
        self.deployment_info()
        
        print("\n分析完成！")
        return model, metrics

def main():
    """主函數"""
    # 建立分析實例
    analyzer = SimpleLinearRegressionCRISPDM()
    
    # 執行完整分析
    model, metrics = analyzer.run_complete_analysis(
        slope=2.5,
        intercept=1.0,
        noise_std=0.8,
        n_points=150
    )
    
    print(f"\n最終結果摘要：")
    print(f"R² = {metrics['r2']:.4f}")
    print(f"RMSE = {metrics['rmse']:.4f}")

if __name__ == "__main__":
    main()
