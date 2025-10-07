#!/usr/bin/env python3
"""
簡單線性迴歸 CRISP-DM 演示 - 重構版 (Clean Code)
Simple Linear Regression CRISP-DM Demo - Refactored Version (Clean Code)

本應用展示 CRISP-DM 方法論的完整流程，並遵循 Clean Code 原則：
- 模組化：重用 linear_regression.py 模組
- 參數化：使用外部類別而非重複邏輯
- 程式碼風格：完整 type hints 和 docstrings
- 穩健性：錯誤處理和日誌記錄

CRISP-DM 流程：
1. Business Understanding - 理解線性關係建模需求
2. Data Understanding - 探索合成資料特性
3. Data Preparation - 生成並準備訓練資料
4. Modeling - 訓練線性迴歸模型
5. Evaluation - 評估模型效能（含殘差分析）
6. Deployment - 部署為互動式應用
"""

import logging
import sys
from typing import Dict, Optional, Tuple

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 導入自定義模組
try:
    from linear_regression import SimpleLinearRegressionCRISPDM
except ImportError as e:
    st.error(f"❌ 無法載入 linear_regression 模組: {e}")
    st.info("請確保 linear_regression.py 在同一目錄下")
    sys.exit(1)

# ==================== 配置 ====================

# 設定 logging（僅顯示錯誤）
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 頁面配置
st.set_page_config(
    page_title="簡單線性迴歸 CRISP-DM - 重構版",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 視覺化函數 ====================

def create_scatter_plot(
    data: pd.DataFrame, 
    y_pred: np.ndarray, 
    model: LinearRegression, 
    true_slope: float, 
    true_intercept: float
) -> go.Figure:
    """
    建立散點圖和迴歸線
    
    Args:
        data: 資料框
        y_pred: 預測值
        model: 訓練好的模型
        true_slope: 真實斜率
        true_intercept: 真實截距
    
    Returns:
        Plotly 圖表物件
    """
    fig = go.Figure()
    
    # 實際資料點
    fig.add_trace(go.Scatter(
        x=data['x'],
        y=data['y'],
        mode='markers',
        name='實際資料',
        marker=dict(color='#1f77b4', size=8, opacity=0.6),
        hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
    ))
    
    # 真實關係線
    fig.add_trace(go.Scatter(
        x=data['x'],
        y=data['y_true'],
        mode='lines',
        name=f'真實關係 (y = {true_slope}x + {true_intercept})',
        line=dict(color='#2ca02c', width=3, dash='dash'),
        hovertemplate='真實 y: %{y:.2f}<extra></extra>'
    ))
    
    # 預測線
    fig.add_trace(go.Scatter(
        x=data['x'],
        y=y_pred,
        mode='lines',
        name=f'預測線 (y = {model.coef_[0]:.3f}x + {model.intercept_:.3f})',
        line=dict(color='#d62728', width=3),
        hovertemplate='預測 y: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={'text': '線性迴歸結果', 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='自變數 (x)',
        yaxis_title='目標變數 (y)',
        height=500,
        hovermode='closest',
        template='plotly_white'
    )
    
    return fig


def create_residual_plot(data: pd.DataFrame, residuals: np.ndarray) -> go.Figure:
    """
    建立殘差圖
    
    Args:
        data: 資料框
        residuals: 殘差陣列
    
    Returns:
        Plotly 圖表物件
    """
    fig = go.Figure()
    
    # 殘差散點圖
    fig.add_trace(go.Scatter(
        x=data['x'],
        y=residuals,
        mode='markers',
        name='殘差',
        marker=dict(color='#ff7f0e', size=8, opacity=0.6),
        hovertemplate='x: %{x:.2f}<br>殘差: %{y:.2f}<extra></extra>'
    ))
    
    # 零線
    fig.add_hline(
        y=0, 
        line_dash="dash", 
        line_color="red", 
        annotation_text="理想殘差 = 0"
    )
    
    fig.update_layout(
        title={'text': '殘差分析圖', 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='自變數 (x)',
        yaxis_title='殘差 (實際 - 預測)',
        height=400,
        template='plotly_white'
    )
    
    return fig


def create_noise_distribution(noise: np.ndarray) -> go.Figure:
    """
    建立噪音分佈直方圖
    
    Args:
        noise: 噪音陣列
    
    Returns:
        Plotly 圖表物件
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=noise,
        nbinsx=30,
        name='噪音分佈',
        marker_color='#9467bd',
        opacity=0.7,
        hovertemplate='區間: %{x}<br>數量: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title={'text': '噪音分佈', 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='噪音值',
        yaxis_title='頻率',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


# ==================== 輔助函數 ====================

def get_r2_interpretation(r2: float) -> Tuple[str, str]:
    """
    根據 R² 值提供解釋
    
    Args:
        r2: R² 決定係數
    
    Returns:
        (狀態類型, 解釋文字)
    """
    percentage = r2 * 100
    
    if r2 >= 0.9:
        return ("success", f"模型解釋了 {percentage:.1f}% 的變異，擬合效果**非常好**！✨")
    elif r2 >= 0.7:
        return ("info", f"模型解釋了 {percentage:.1f}% 的變異，擬合效果**良好** ✓")
    elif r2 >= 0.5:
        return ("warning", f"模型解釋了 {percentage:.1f}% 的變異，擬合效果**中等** ⚠️")
    else:
        return ("error", f"模型解釋了 {percentage:.1f}% 的變異，擬合效果**較差** ⚠")


def display_parameter_comparison(
    true_slope: float, 
    true_intercept: float,
    pred_slope: float, 
    pred_intercept: float
) -> pd.DataFrame:
    """
    顯示參數比較表格
    
    Args:
        true_slope: 真實斜率
        true_intercept: 真實截距
        pred_slope: 預測斜率
        pred_intercept: 預測截距
    
    Returns:
        比較表格資料框
    """
    slope_error = abs(pred_slope - true_slope)
    intercept_error = abs(pred_intercept - true_intercept)
    
    slope_rel_error = (slope_error / abs(true_slope) * 100) if true_slope != 0 else 0
    intercept_rel_error = (intercept_error / abs(true_intercept) * 100) if true_intercept != 0 else 0
    
    comparison_df = pd.DataFrame({
        '參數': ['斜率 (Slope)', '截距 (Intercept)'],
        '真實值': [f"{true_slope:.4f}", f"{true_intercept:.4f}"],
        '預測值': [f"{pred_slope:.4f}", f"{pred_intercept:.4f}"],
        '絕對誤差': [f"{slope_error:.4f}", f"{intercept_error:.4f}"],
        '相對誤差 (%)': [f"{slope_rel_error:.2f}%", f"{intercept_rel_error:.2f}%"]
    })
    
    return comparison_df


def calculate_additional_metrics(
    y_with_noise: np.ndarray, 
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    計算額外的評估指標
    
    Args:
        y_with_noise: 實際目標變數 (含噪音)
        y_pred: 模型預測值
    
    Returns:
        包含額外指標的字典
    """
    mae = mean_absolute_error(y_with_noise, y_pred)
    residuals = y_with_noise - y_pred
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    
    return {
        'mae': mae,
        'residual_mean': residual_mean,
        'residual_std': residual_std,
        'residuals': residuals
    }


# ==================== 主程式 ====================

def main() -> None:
    """主應用程式"""
    
    # 標題與說明
    st.title("📈 簡單線性迴歸 CRISP-DM 演示 - 重構版")
    st.markdown("""
    本應用展示**簡單線性迴歸**模型的完整 **CRISP-DM** 流程，採用 **Clean Code** 設計：
    - 🎯 **模組化設計**: 重用 `linear_regression.py` 核心邏輯
    - 📊 **視覺化分析**: 迴歸線、殘差圖、噪音分佈
    - 📈 **效能評估**: R²、MSE、RMSE、MAE 等指標
    - 🔍 **參數比較**: 真實參數 vs 預測參數
    """)
    
    # st.info("💡 **重構優勢**: 此版本使用外部 `linear_regression.py` 模組，避免程式碼重複，提高可維護性！")
    st.markdown("---")
    
    # 側邊欄參數設定
    st.sidebar.header("🎛️ 參數設定")
    st.sidebar.markdown("### 線性模型參數")
    
    slope = st.sidebar.slider(
        "斜率 (a)", -5.0, 5.0, 2.5, 0.1,
        help="控制線性關係的傾斜程度"
    )
    intercept = st.sidebar.slider(
        "截距 (b)", -10.0, 10.0, 1.0, 0.1,
        help="當 x=0 時的 y 值"
    )
    
    st.sidebar.markdown("### 資料參數")
    noise_std = st.sidebar.slider(
        "噪音標準差", 0.0, 3.0, 0.5, 0.1,
        help="控制資料的隨機性程度"
    )
    n_points = st.sidebar.slider(
        "資料點數量", 50, 500, 100, 10,
        help="生成資料的樣本數"
    )
    
    st.sidebar.markdown("### 資料範圍")
    x_min = st.sidebar.number_input("x 最小值", value=-5.0)
    x_max = st.sidebar.number_input("x 最大值", value=5.0)
    
    # 驗證輸入
    if x_min >= x_max:
        st.error("❌ x 最小值必須小於 x 最大值")
        return
    
    try:
        # ==================== 使用外部模組執行 CRISP-DM ====================
        
        # 初始化分析器（禁用 verbose 以避免 console 輸出）
        analyzer = SimpleLinearRegressionCRISPDM(verbose=False)
        
        # 執行 CRISP-DM 流程
        # 1. Data Understanding（資料理解與生成）
        analyzer.data_understanding(
            slope=slope,
            intercept=intercept,
            noise_std=noise_std,
            n_points=n_points,
            x_range=(x_min, x_max),
            random_seed=42
        )
        
        # 2. Data Preparation（資料準備）
        X, y = analyzer.data_preparation()
        
        # 3. Modeling（建模）
        model = analyzer.modeling(X, y)
        
        # 4. Evaluation（評估）
        metrics = analyzer.evaluation(X, y)
        
        # 獲取資料和預測值
        data = analyzer.data
        y_pred = model.predict(X)
        
        # 計算額外指標
        additional_metrics = calculate_additional_metrics(data['y'].values, y_pred)
        
        # ==================== 指標卡片顯示 ====================
        st.subheader("📊 模型效能指標")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "R² (決定係數)", 
                f"{metrics['r2']:.4f}",
                help="模型解釋變異的比例 (0-1，越接近1越好)"
            )
        with col2:
            st.metric(
                "MSE (均方誤差)", 
                f"{metrics['mse']:.4f}",
                help="預測誤差的平方平均值 (越小越好)"
            )
        with col3:
            st.metric(
                "RMSE (均方根誤差)", 
                f"{metrics['rmse']:.4f}",
                help="MSE 的平方根，與目標變數同單位 (越小越好)"
            )
        with col4:
            st.metric(
                "MAE (平均絕對誤差)",
                f"{additional_metrics['mae']:.4f}",
                help="預測誤差絕對值的平均 (越小越好)"
            )
        
        # R² 解釋
        status_type, interpretation = get_r2_interpretation(metrics['r2'])
        if status_type == "success":
            st.success(interpretation)
        elif status_type == "info":
            st.info(interpretation)
        elif status_type == "warning":
            st.warning(interpretation)
        else:
            st.error(interpretation)
        
        st.markdown("---")
        
        # ==================== 分頁展示 ====================
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 迴歸結果", 
            "🔍 殘差分析", 
            "📊 噪音分佈",
            "📋 參數比較"
        ])
        
        with tab1:
            st.markdown("### 線性迴歸視覺化")
            st.markdown("""
            此圖展示：
            - 🔵 **藍點**: 實際資料點（含噪音）
            - 🟢 **綠虛線**: 真實線性關係
            - 🔴 **紅實線**: 模型預測線
            """)
            
            fig_main = create_scatter_plot(
                data, y_pred, model, 
                analyzer.true_slope, analyzer.true_intercept
            )
            st.plotly_chart(fig_main, use_container_width=True)
            
            # 下載資料選項
            csv_data = data[['x', 'y']].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 下載資料 (CSV)",
                data=csv_data,
                file_name="linear_regression_data.csv",
                mime="text/csv"
            )
        
        with tab2:
            st.markdown("### 殘差分析")
            st.markdown("""
            **殘差 = 實際值 - 預測值**
            
            理想情況下：
            - 殘差應該隨機分佈在零線附近
            - 不應該有明顯的模式或趨勢
            - 標準差越小表示預測越準確
            """)
            
            fig_residual = create_residual_plot(data, additional_metrics['residuals'])
            st.plotly_chart(fig_residual, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "殘差平均值", 
                    f"{additional_metrics['residual_mean']:.4f}",
                    help="理想值應接近 0"
                )
            with col2:
                st.metric(
                    "殘差標準差", 
                    f"{additional_metrics['residual_std']:.4f}",
                    help="數值越小表示預測越穩定"
                )
        
        with tab3:
            st.markdown("### 噪音分佈")
            st.markdown("""
            此直方圖顯示資料中噪音的分佈情況。
            
            在理想的線性迴歸假設下，噪音應該服從**常態分佈**（鐘形曲線）。
            """)
            
            fig_noise = create_noise_distribution(data['noise'])
            st.plotly_chart(fig_noise, use_container_width=True)
            
            st.info(f"📊 設定的噪音標準差: **{noise_std}**")
        
        with tab4:
            st.markdown("### 參數比較表")
            st.markdown("""
            比較真實參數與模型預測出的參數，評估模型的參數估計準確度。
            """)
            
            comparison_df = display_parameter_comparison(
                analyzer.true_slope, 
                analyzer.true_intercept,
                analyzer.predicted_slope, 
                analyzer.predicted_intercept
            )
            
            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )
            
            # 參數估計品質評估
            slope_error_pct = float(comparison_df.iloc[0]['相對誤差 (%)'].rstrip('%'))
            intercept_error_pct = float(comparison_df.iloc[1]['相對誤差 (%)'].rstrip('%'))
            
            if slope_error_pct < 5 and intercept_error_pct < 5:
                st.success("✅ 參數估計非常準確！相對誤差 < 5%")
            elif slope_error_pct < 10 and intercept_error_pct < 10:
                st.info("✓ 參數估計良好，相對誤差 < 10%")
            else:
                st.warning("⚠️ 參數估計誤差較大，可能需要更多資料或降低噪音")
        
        # ==================== CRISP-DM 說明 ====================
        st.markdown("---")
        with st.expander("📚 關於 CRISP-DM 方法論"):
            st.markdown("""
            ### CRISP-DM (Cross-Industry Standard Process for Data Mining)
            
            這是資料科學專案的標準流程，包含 6 個階段：
            
            1. **業務理解 (Business Understanding)**  
               確定專案目標和需求
            
            2. **資料理解 (Data Understanding)**  
               收集資料並探索其特性
            
            3. **資料準備 (Data Preparation)**  
               清理和轉換資料以供建模
            
            4. **建模 (Modeling)**  
               選擇並應用適當的機器學習算法
            
            5. **評估 (Evaluation)**  
               評估模型效能是否滿足業務目標
            
            6. **部署 (Deployment)**  
               將模型整合到實際應用中
            
            本應用展示了完整的 CRISP-DM 循環！
            """)
        
        with st.expander("🎨 Clean Code 重構說明"):
            st.markdown("""
            ### 本版本的 Clean Code 優化
            
            #### ✅ 模組化設計
            - 重用 `linear_regression.py` 的 `SimpleLinearRegressionCRISPDM` 類別
            - 避免程式碼重複（DRY 原則）
            - 分離關注點：核心邏輯 vs UI 邏輯
            
            #### ✅ Type Hints
            - 所有函式都有完整的 type hints
            - 使用 `typing` 模組的進階類型
            - 提高程式碼可讀性和 IDE 支援
            
            #### ✅ 錯誤處理
            - 適當的 try-except 區塊
            - 友善的錯誤訊息
            - 參數驗證
            
            #### ✅ Docstrings
            - 所有函式都有 Google 風格的 docstrings
            - 說明參數、返回值、異常
            - 便於生成 API 文件
            
            #### ✅ 程式碼組織
            - 清晰的區段劃分
            - 邏輯分組
            - 一致的命名規範
            """)
        
        with st.expander("❓ 常見問題 FAQ"):
            st.markdown("""
            **Q: 什麼是 R² 決定係數？**  
            A: R² 表示模型解釋了目標變數變異的比例，範圍 0-1。R²=1 表示完美擬合。
            
            **Q: MSE、RMSE、MAE 有什麼差別？**  
            A: 
            - MSE: 誤差平方的平均，對大誤差敏感
            - RMSE: MSE 的平方根，與目標變數同單位
            - MAE: 誤差絕對值的平均，對離群值較不敏感
            
            **Q: 為什麼增加噪音會降低 R²？**  
            A: 噪音增加了資料的隨機性，使得線性模型更難完美擬合資料。
            
            **Q: 殘差分析的意義是什麼？**  
            A: 檢查殘差是否隨機分佈可以幫助驗證線性迴歸的假設是否成立。
            
            **Q: 重構版和原版有什麼差異？**  
            A: 重構版使用外部模組，避免程式碼重複，更易於維護和測試。
            """)
            
    except ValueError as e:
        st.error(f"❌ 參數錯誤: {str(e)}")
        st.info("請調整參數後重試")
    except Exception as e:
        st.error(f"❌ 發生錯誤: {str(e)}")
        st.info("請檢查參數設定是否合理，或查看控制台日誌")
        logger.exception("應用程式執行失敗")
    
    # ==================== 頁腳資訊 ====================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p><strong>💡 使用提示：</strong></p>
    <p>調整左側參數來探索不同情境 | 觀察噪音對模型效能的影響 | 比較真實參數與預測參數</p>
    <p style='font-size: 0.9em; margin-top: 10px;'>
    Built with Streamlit | 展示 CRISP-DM 方法論 | Clean Code 重構版
    </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
