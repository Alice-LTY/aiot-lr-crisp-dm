"""
簡單線性迴歸 CRISP-DM 演示
Simple Linear Regression CRISP-DM Demo
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 頁面配置
st.set_page_config(
    page_title="簡單線性迴歸 CRISP-DM",
    page_icon="📈",
    layout="wide"
)

def generate_data(slope, intercept, noise_std, n_points, x_range):
    """生成合成資料"""
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

def train_model(data):
    """訓練線性迴歸模型"""
    X = data[['x']]
    y = data['y']
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    
    return model, y_pred

def calculate_metrics(y_true, y_pred):
    """計算評估指標"""
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    return {'r2': r2, 'mse': mse, 'rmse': rmse}

def create_scatter_plot(data, y_pred, model, true_slope, true_intercept):
    """建立散點圖和迴歸線"""
    fig = go.Figure()
    
    # 實際資料點
    fig.add_trace(go.Scatter(
        x=data['x'],
        y=data['y'],
        mode='markers',
        name='實際資料',
        marker=dict(color='blue', opacity=0.6)
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
        name=f'預測線 (y = {model.coef_[0]:.2f}x + {model.intercept_:.2f})',
        line=dict(color='red', width=3)
    ))
    
    fig.update_layout(
        title='線性迴歸結果',
        xaxis_title='x',
        yaxis_title='y',
        hovermode='closest'
    )
    
    return fig

def create_residual_plot(y_true, y_pred):
    """建立殘差圖"""
    residuals = y_true - y_pred
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        name='殘差',
        marker=dict(color='purple', opacity=0.6)
    ))
    
    # 零線
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title='殘差圖',
        xaxis_title='預測值',
        yaxis_title='殘差'
    )
    
    return fig

def create_comparison_plot(y_true, y_pred):
    """建立預測 vs 實際比較圖"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='預測 vs 實際',
        marker=dict(color='green', opacity=0.6)
    ))
    
    # 完美預測線
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='完美預測線',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='預測 vs 實際值',
        xaxis_title='實際值',
        yaxis_title='預測值'
    )
    
    return fig

def main():
    """主應用程式"""
    st.title("📈 簡單線性迴歸 CRISP-DM 演示")
    st.markdown("---")
    
    # 側邊欄參數設定
    st.sidebar.header("🎛️ 參數設定")
    
    # CRISP-DM 階段選擇
    st.sidebar.subheader("CRISP-DM 階段")
    show_business = st.sidebar.checkbox("1. Business Understanding", True)
    show_data = st.sidebar.checkbox("2. Data Understanding", True)
    show_prep = st.sidebar.checkbox("3. Data Preparation", True)
    show_model = st.sidebar.checkbox("4. Modeling", True)
    show_eval = st.sidebar.checkbox("5. Evaluation", True)
    show_deploy = st.sidebar.checkbox("6. Deployment", True)
    
    st.sidebar.subheader("模型參數")
    slope = st.sidebar.slider("斜率 (a)", -5.0, 5.0, 2.5, 0.1)
    intercept = st.sidebar.slider("截距 (b)", -10.0, 10.0, 1.0, 0.1)
    noise_std = st.sidebar.slider("噪音標準差", 0.0, 2.0, 0.5, 0.1)
    n_points = st.sidebar.slider("資料點數量", 50, 500, 100, 10)
    
    st.sidebar.subheader("資料範圍")
    x_min = st.sidebar.number_input("x 最小值", value=-5.0)
    x_max = st.sidebar.number_input("x 最大值", value=5.0)
    
    # 生成資料和訓練模型
    data = generate_data(slope, intercept, noise_std, n_points, (x_min, x_max))
    model, y_pred = train_model(data)
    metrics = calculate_metrics(data['y'], y_pred)
    
    # 1. Business Understanding
    if show_business:
        st.header("1. 🎯 Business Understanding (業務理解)")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("問題定義")
            st.write("""
            - **目標**：建立簡單線性迴歸模型來理解兩個變數之間的線性關係
            - **應用**：預測、趨勢分析、關係探索
            - **成功指標**：高 R²、低 MSE、參數準確性
            """)
        
        with col2:
            st.subheader("使用案例")
            st.write("""
            - 銷售額與廣告支出的關係
            - 房價與面積的關係
            - 溫度與能源消耗的關係
            - 學習時間與考試成績的關係
            """)
    
    # 2. Data Understanding
    if show_data:
        st.header("2. 📊 Data Understanding (資料理解)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("資料點數量", n_points)
        with col2:
            st.metric("真實斜率", f"{slope:.2f}")
        with col3:
            st.metric("真實截距", f"{intercept:.2f}")
        with col4:
            st.metric("噪音標準差", f"{noise_std:.2f}")
        
        # 資料統計
        st.subheader("資料統計描述")
        st.dataframe(data.describe())
        
        # 資料視覺化
        col1, col2 = st.columns(2)
        
        with col1:
            fig_x = px.histogram(data, x='x', title='x 變數分布')
            st.plotly_chart(fig_x, use_container_width=True)
        
        with col2:
            fig_y = px.histogram(data, x='y', title='y 變數分布')
            st.plotly_chart(fig_y, use_container_width=True)
    
    # 3. Data Preparation
    if show_prep:
        st.header("3. 🔧 Data Preparation (資料準備)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("資料品質檢查")
            st.write(f"- 缺失值：{data.isnull().sum().sum()}")
            st.write(f"- 資料形狀：{data.shape}")
            st.write(f"- x 範圍：[{data['x'].min():.2f}, {data['x'].max():.2f}]")
            st.write(f"- y 範圍：[{data['y'].min():.2f}, {data['y'].max():.2f}]")
        
        with col2:
            st.subheader("資料樣本")
            st.dataframe(data.head(10))
    
    # 4. Modeling
    if show_model:
        st.header("4. 🤖 Modeling (建模)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("模型參數")
            st.write(f"**預測斜率：** {model.coef_[0]:.4f}")
            st.write(f"**預測截距：** {model.intercept_:.4f}")
            st.write(f"**真實斜率：** {slope:.4f}")
            st.write(f"**真實截距：** {intercept:.4f}")
        
        with col2:
            st.subheader("參數誤差")
            slope_error = abs(model.coef_[0] - slope)
            intercept_error = abs(model.intercept_ - intercept)
            st.write(f"**斜率誤差：** {slope_error:.4f}")
            st.write(f"**截距誤差：** {intercept_error:.4f}")
            
            if slope_error < 0.1:
                st.success("斜率預測非常準確！")
            elif slope_error < 0.5:
                st.info("斜率預測良好")
            else:
                st.warning("斜率預測需要改善")
    
    # 5. Evaluation
    if show_eval:
        st.header("5. 📈 Evaluation (評估)")
        
        # 評估指標
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("R² (決定係數)", f"{metrics['r2']:.4f}")
        with col2:
            st.metric("MSE (均方誤差)", f"{metrics['mse']:.4f}")
        with col3:
            st.metric("RMSE (均方根誤差)", f"{metrics['rmse']:.4f}")
        
        # R² 解釋
        if metrics['r2'] >= 0.9:
            st.success(f"模型解釋了 {metrics['r2']*100:.1f}% 的變異，擬合效果非常好！")
        elif metrics['r2'] >= 0.7:
            st.info(f"模型解釋了 {metrics['r2']*100:.1f}% 的變異，擬合效果良好")
        elif metrics['r2'] >= 0.5:
            st.warning(f"模型解釋了 {metrics['r2']*100:.1f}% 的變異，擬合效果中等")
        else:
            st.error(f"模型解釋了 {metrics['r2']*100:.1f}% 的變異，擬合效果較差")
        
        # 視覺化結果
        st.subheader("視覺化結果")
        
        # 主要散點圖
        fig_main = create_scatter_plot(data, y_pred, model, slope, intercept)
        st.plotly_chart(fig_main, use_container_width=True)
        
        # 殘差分析
        col1, col2 = st.columns(2)
        
        with col1:
            fig_residual = create_residual_plot(data['y'], y_pred)
            st.plotly_chart(fig_residual, use_container_width=True)
        
        with col2:
            fig_comparison = create_comparison_plot(data['y'], y_pred)
            st.plotly_chart(fig_comparison, use_container_width=True)
    
    # 6. Deployment
    if show_deploy:
        st.header("6. 🚀 Deployment (部署)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("部署選項")
            st.write("""
            - **Streamlit Cloud**：線上免費部署
            - **Heroku**：雲端平台部署
            - **Docker**：容器化部署
            - **AWS/GCP/Azure**：雲端服務商
            """)
        
        with col2:
            st.subheader("部署考量")
            st.write("""
            - **效能**：回應時間、併發處理
            - **可擴展性**：使用者增長適應
            - **維護性**：更新、監控、除錯
            - **安全性**：資料保護、存取控制
            """)
        
        st.subheader("模型輸出格式")
        
        # 模型摘要
        model_summary = {
            "模型類型": "簡單線性迴歸",
            "斜率": f"{model.coef_[0]:.4f}",
            "截距": f"{model.intercept_:.4f}",
            "R²": f"{metrics['r2']:.4f}",
            "RMSE": f"{metrics['rmse']:.4f}",
            "訓練資料點": n_points
        }
        
        st.json(model_summary)
    
    # 底部資訊
    st.markdown("---")
    st.markdown("""
    **💡 使用提示：**
    - 調整左側參數來探索不同情境
    - 觀察噪音對模型效能的影響
    - 比較真實參數與預測參數
    - 透過視覺化理解模型表現
    """)

if __name__ == "__main__":
    main()
