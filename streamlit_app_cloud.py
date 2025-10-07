"""
簡單線性迴歸 CRISP-DM 演示 - Cloud 部署版
Simple Linear Regression CRISP-DM Demo - Cloud Deployment Version

專為 Streamlit Cloud 部署優化的版本：
- 移除可能導致部署失敗的功能
- 簡化錯誤處理
- 確保跨平台兼容性
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings

# 忽略警告，避免 Cloud 部署問題
warnings.filterwarnings('ignore')

# ==================== 頁面配置 ====================
st.set_page_config(
    page_title="簡單線性迴歸 CRISP-DM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 核心函數 ====================
@st.cache_data
def generate_data(slope: float, intercept: float, noise_std: float, 
                  n_points: int, x_range: tuple) -> pd.DataFrame:
    """生成合成線性迴歸資料"""
    try:
        np.random.seed(42)
        
        x = np.linspace(x_range[0], x_range[1], n_points)
        noise = np.random.normal(0, noise_std, n_points)
        y_true = slope * x + intercept
        y = y_true + noise
        
        return pd.DataFrame({
            'x': x,
            'y': y,
            'y_true': y_true,
            'noise': noise
        })
    except Exception as e:
        st.error(f"資料生成錯誤: {e}")
        return pd.DataFrame()

def train_model(data: pd.DataFrame) -> tuple:
    """訓練線性迴歸模型"""
    try:
        X = data[['x']].values
        y = data['y'].values
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        return model, y_pred
    except Exception as e:
        st.error(f"模型訓練錯誤: {e}")
        return None, None

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_with_noise: np.ndarray) -> dict:
    """計算評估指標"""
    try:
        r2 = r2_score(y_with_noise, y_pred)
        mse = mean_squared_error(y_with_noise, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_with_noise, y_pred)
        
        residuals = y_with_noise - y_pred
        
        return {
            'r2': r2, 
            'mse': mse, 
            'rmse': rmse,
            'mae': mae,
            'residuals': residuals
        }
    except Exception as e:
        st.error(f"指標計算錯誤: {e}")
        return {}

def create_main_plot(data: pd.DataFrame, y_pred: np.ndarray, 
                    model: LinearRegression, true_slope: float, 
                    true_intercept: float) -> go.Figure:
    """建立主要散點圖"""
    fig = go.Figure()
    
    # 實際資料點
    fig.add_trace(go.Scatter(
        x=data['x'],
        y=data['y'],
        mode='markers',
        name='實際資料',
        marker=dict(color='blue', size=8, opacity=0.6)
    ))
    
    # 真實關係線
    fig.add_trace(go.Scatter(
        x=data['x'],
        y=data['y_true'],
        mode='lines',
        name=f'真實關係 (y = {true_slope}x + {true_intercept})',
        line=dict(color='green', width=3, dash='dash')
    ))
    
    # 預測線
    fig.add_trace(go.Scatter(
        x=data['x'],
        y=y_pred,
        mode='lines',
        name=f'預測線 (y = {model.coef_[0]:.3f}x + {model.intercept_:.3f})',
        line=dict(color='red', width=3)
    ))
    
    fig.update_layout(
        title='線性迴歸結果',
        xaxis_title='自變數 (x)',
        yaxis_title='目標變數 (y)',
        height=500,
        template='plotly_white'
    )
    
    return fig

def create_residual_plot(data: pd.DataFrame, residuals: np.ndarray) -> go.Figure:
    """建立殘差圖"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['x'],
        y=residuals,
        mode='markers',
        name='殘差',
        marker=dict(color='orange', size=8, opacity=0.6)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title='殘差分析圖',
        xaxis_title='自變數 (x)',
        yaxis_title='殘差 (實際 - 預測)',
        height=400,
        template='plotly_white'
    )
    
    return fig

def get_r2_interpretation(r2: float) -> tuple:
    """R² 解釋"""
    percentage = r2 * 100
    
    if r2 >= 0.9:
        return ("success", f"模型解釋了 {percentage:.1f}% 的變異，擬合效果非常好！")
    elif r2 >= 0.7:
        return ("info", f"模型解釋了 {percentage:.1f}% 的變異，擬合效果良好")
    elif r2 >= 0.5:
        return ("warning", f"模型解釋了 {percentage:.1f}% 的變異，擬合效果中等")
    else:
        return ("error", f"模型解釋了 {percentage:.1f}% 的變異，擬合效果較差")

def display_parameter_comparison(true_slope: float, true_intercept: float,
                               pred_slope: float, pred_intercept: float) -> pd.DataFrame:
    """參數比較表格"""
    slope_error = abs(pred_slope - true_slope)
    intercept_error = abs(pred_intercept - true_intercept)
    
    slope_rel_error = (slope_error / abs(true_slope) * 100) if true_slope != 0 else 0
    intercept_rel_error = (intercept_error / abs(true_intercept) * 100) if true_intercept != 0 else 0
    
    return pd.DataFrame({
        '參數': ['斜率', '截距'],
        '真實值': [f"{true_slope:.4f}", f"{true_intercept:.4f}"],
        '預測值': [f"{pred_slope:.4f}", f"{pred_intercept:.4f}"],
        '絕對誤差': [f"{slope_error:.4f}", f"{intercept_error:.4f}"],
        '相對誤差 (%)': [f"{slope_rel_error:.2f}%", f"{intercept_rel_error:.2f}%"]
    })

# ==================== 主程式 ====================
def main():
    """主應用程式"""
    
    # 標題
    st.title("📈 簡單線性迴歸 CRISP-DM 演示")
    st.markdown("展示 **CRISP-DM 方法論**的完整流程，建立簡單線性迴歸模型。")
    st.markdown("---")
    
    # 側邊欄參數設定
    st.sidebar.header("🎛️ 參數設定")
    
    slope = st.sidebar.slider("斜率", -5.0, 5.0, 2.5, 0.1)
    intercept = st.sidebar.slider("截距", -10.0, 10.0, 1.0, 0.1)
    noise_std = st.sidebar.slider("噪音標準差", 0.0, 3.0, 0.5, 0.1)
    n_points = st.sidebar.slider("資料點數量", 50, 500, 100, 10)
    
    x_min = st.sidebar.number_input("x 最小值", value=-5.0)
    x_max = st.sidebar.number_input("x 最大值", value=5.0)
    
    # 驗證輸入
    if x_min >= x_max:
        st.error("❌ x 最小值必須小於 x 最大值")
        return
    
    # 生成資料和訓練模型
    data = generate_data(slope, intercept, noise_std, n_points, (x_min, x_max))
    
    if data.empty:
        return
    
    model, y_pred = train_model(data)
    
    if model is None:
        return
    
    metrics = calculate_metrics(data['y_true'], y_pred, data['y'])
    
    if not metrics:
        return
    
    # 顯示指標
    st.subheader("📊 模型效能指標")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² (決定係數)", f"{metrics['r2']:.4f}")
    with col2:
        st.metric("MSE (均方誤差)", f"{metrics['mse']:.4f}")
    with col3:
        st.metric("RMSE (均方根誤差)", f"{metrics['rmse']:.4f}")
    with col4:
        st.metric("MAE (平均絕對誤差)", f"{metrics['mae']:.4f}")
    
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
    
    # 分頁展示
    tab1, tab2, tab3 = st.tabs(["📈 迴歸結果", "🔍 殘差分析", "📋 參數比較"])
    
    with tab1:
        st.markdown("### 線性迴歸視覺化")
        fig_main = create_main_plot(data, y_pred, model, slope, intercept)
        st.plotly_chart(fig_main, use_container_width=True)
        
        # 下載資料
        csv_data = data[['x', 'y']].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 下載資料 (CSV)",
            data=csv_data,
            file_name="linear_regression_data.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.markdown("### 殘差分析")
        st.markdown("**殘差 = 實際值 - 預測值**")
        
        fig_residual = create_residual_plot(data, metrics['residuals'])
        st.plotly_chart(fig_residual, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("殘差平均值", f"{np.mean(metrics['residuals']):.4f}")
        with col2:
            st.metric("殘差標準差", f"{np.std(metrics['residuals']):.4f}")
    
    with tab3:
        st.markdown("### 參數比較表")
        
        comparison_df = display_parameter_comparison(
            slope, intercept, 
            model.coef_[0], model.intercept_
        )
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # CRISP-DM 說明
    with st.expander("📚 關於 CRISP-DM"):
        st.markdown("""
        ### CRISP-DM 流程
        
        1. **業務理解**: 確定專案目標
        2. **資料理解**: 探索資料特性  
        3. **資料準備**: 清理和轉換資料
        4. **建模**: 訓練機器學習模型
        5. **評估**: 評估模型效能
        6. **部署**: 整合到實際應用
        
        本應用展示了完整的 CRISP-DM 循環！
        """)
    
    # 頁腳
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p>Built with Streamlit | 展示 CRISP-DM 方法論</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()