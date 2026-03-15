import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ─── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background-color: #0a0f1e; color: #e2e8f0; }

    section[data-testid="stSidebar"] {
        background: #0d1526;
        border-right: 1px solid #1e2d4a;
    }

    .metric-card {
        background: linear-gradient(135deg, #0d1526, #111d35);
        border: 1px solid #1e3a5f;
        border-radius: 14px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-card .val {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #22d3ee, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card .lbl {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    .prediction-box {
        background: linear-gradient(135deg, #0d1526, #111d35);
        border: 2px solid #22d3ee;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    .prediction-box .price {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #22d3ee, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .prediction-box .label {
        font-size: 1rem;
        color: #94a3b8;
        margin-top: 8px;
    }

    .section-title {
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #22d3ee, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }

    div[data-testid="stTabs"] button {
        color: #64748b;
        font-weight: 600;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #22d3ee;
        border-bottom: 2px solid #22d3ee;
    }

    .stSlider > div > div { background: #1e3a5f; }

    div[data-testid="metric-container"] {
        background: #0d1526;
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load & Cache Data ───────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        housing = fetch_california_housing()
        df = pd.DataFrame(housing.data, columns=housing.feature_names)
        df['Price'] = housing.target
        return df
    except Exception as exc:
        raise RuntimeError(
            "Unable to load the California Housing dataset. "
            "Please check your network connection or local sklearn cache."
        ) from exc

@st.cache_resource
def train_models(df):
    X = df.drop('Price', axis=1)
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        'Linear Regression': (LinearRegression(), True),
        'Ridge':             (Ridge(alpha=1.0), True),
        'Lasso':             (Lasso(alpha=0.1), True),
        'Random Forest':     (RandomForestRegressor(n_estimators=100, random_state=42), False),
        'Gradient Boosting': (GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42), False),
    }

    results = {}
    for name, (model, scaled) in models.items():
        Xtr = X_train_s if scaled else X_train
        Xte = X_test_s  if scaled else X_test
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        results[name] = {'model': model, 'rmse': rmse, 'r2': r2, 'scaled': scaled,
                         'y_test': y_test, 'y_pred': y_pred}

    return results, scaler, X_test, y_test

try:
    df = load_data()
except RuntimeError as exc:
    st.error("Dataset unavailable. The app cannot continue right now.")
    st.caption(str(exc))
    st.stop()

results, scaler, X_test, y_test = train_models(df)
best_model_name = min(results, key=lambda x: results[x]['rmse'])

# ─── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 House Price Predictor")
    st.markdown("*California Housing Dataset*")
    st.divider()

    st.markdown("### 📊 Dataset Info")
    st.markdown(f"- **Samples:** {len(df):,}")
    st.markdown(f"- **Features:** {len(df.columns)-1}")
    st.markdown(f"- **Best Model:** {best_model_name}")
    st.divider()

    st.markdown("### 🧭 Navigation")
    page = st.radio("", ["🏠 Overview", "📊 EDA", "🤖 Model Comparison", "🔮 Predict Price"], label_visibility="collapsed")

# ─── OVERVIEW ───────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown('<div class="section-title">California House Price Predictor</div>', unsafe_allow_html=True)
    st.markdown("End-to-end ML project — EDA, model training, and live predictions on the California Housing dataset.")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="val">{len(df):,}</div><div class="lbl">Total Samples</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="val">{len(df.columns)-1}</div><div class="lbl">Features</div></div>', unsafe_allow_html=True)
    with c3:
        best_r2 = results[best_model_name]['r2']
        st.markdown(f'<div class="metric-card"><div class="val">{best_r2*100:.1f}%</div><div class="lbl">Best R² Score</div></div>', unsafe_allow_html=True)
    with c4:
        best_rmse = results[best_model_name]['rmse']
        st.markdown(f'<div class="metric-card"><div class="val">${best_rmse*100000:,.0f}</div><div class="lbl">Best RMSE</div></div>', unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 💡 What This App Does")
        st.markdown("""
        - 📊 **Exploratory Data Analysis** — distributions, correlations, geo map
        - 🤖 **5 ML Models** compared — Linear, Ridge, Lasso, Random Forest, Gradient Boosting
        - 🔮 **Live Predictor** — input house features, get instant price estimate
        - 📈 **Visual insights** — Plotly interactive charts throughout
        """)

    with col2:
        st.markdown("#### 🏆 Model Leaderboard")
        leaderboard = pd.DataFrame([
            {"Model": k, "R²": f"{v['r2']*100:.1f}%", "RMSE": f"${v['rmse']*100000:,.0f}"}
            for k, v in sorted(results.items(), key=lambda x: -x[1]['r2'])
        ])
        st.dataframe(leaderboard, hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("#### 📋 Dataset Sample")
    st.dataframe(df.head(10).style.format({'Price': '${:.2f}'}), use_container_width=True)

# ─── EDA ────────────────────────────────────────────────────────
elif page == "📊 EDA":
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔗 Correlations", "🌍 Geo Map", "📋 Stats"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x='Price', nbins=50, title='House Price Distribution',
                               color_discrete_sequence=['#22d3ee'], template='plotly_dark')
            fig.update_layout(paper_bgcolor='#0d1526', plot_bgcolor='#0d1526')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(df, x='MedInc', y='Price', title='Median Income vs Price',
                             color='Price', color_continuous_scale='viridis',
                             opacity=0.4, template='plotly_dark')
            fig.update_layout(paper_bgcolor='#0d1526', plot_bgcolor='#0d1526')
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig = px.scatter(df, x='AveRooms', y='Price', title='Average Rooms vs Price',
                             color='Price', color_continuous_scale='plasma',
                             opacity=0.4, template='plotly_dark')
            fig.update_layout(paper_bgcolor='#0d1526', plot_bgcolor='#0d1526')
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            fig = px.scatter(df, x='HouseAge', y='Price', title='House Age vs Price',
                             color='MedInc', color_continuous_scale='turbo',
                             opacity=0.4, template='plotly_dark')
            fig.update_layout(paper_bgcolor='#0d1526', plot_bgcolor='#0d1526')
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        corr = df.corr()
        fig = px.imshow(corr, text_auto='.2f', title='Feature Correlation Heatmap',
                        color_continuous_scale='RdBu_r', template='plotly_dark')
        fig.update_layout(paper_bgcolor='#0d1526', plot_bgcolor='#0d1526', height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Correlation with Price")
        price_corr = corr['Price'].drop('Price').sort_values()
        fig2 = px.bar(x=price_corr.values, y=price_corr.index, orientation='h',
                      color=price_corr.values, color_continuous_scale='RdBu_r',
                      template='plotly_dark')
        fig2.update_layout(paper_bgcolor='#0d1526', plot_bgcolor='#0d1526')
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig = px.scatter_mapbox(df, lat='Latitude', lon='Longitude', color='Price',
                                size='Price', color_continuous_scale='viridis',
                                size_max=8, zoom=5, opacity=0.6,
                                title='California House Prices by Location',
                                mapbox_style='carto-darkmatter')
        fig.update_layout(paper_bgcolor='#0d1526', height=550)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("#### Dataset Statistics")
        st.dataframe(df.describe().style.format('{:.3f}'), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Missing Values**")
            missing = df.isnull().sum().reset_index()
            missing.columns = ['Feature', 'Missing']
            st.dataframe(missing, hide_index=True, use_container_width=True)
        with col2:
            st.markdown("**Data Types**")
            dtypes = df.dtypes.reset_index()
            dtypes.columns = ['Feature', 'Type']
            st.dataframe(dtypes, hide_index=True, use_container_width=True)

# ─── MODEL COMPARISON ───────────────────────────────────────────
elif page == "🤖 Model Comparison":
    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
    st.divider()

    # Metrics bar charts
    model_names = list(results.keys())
    r2_scores  = [results[m]['r2'] * 100 for m in model_names]
    rmse_scores = [results[m]['rmse'] * 100000 for m in model_names]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(x=model_names, y=r2_scores, title='R² Score by Model (%)',
                     color=r2_scores, color_continuous_scale='viridis',
                     template='plotly_dark', text=[f"{v:.1f}%" for v in r2_scores])
        fig.update_layout(paper_bgcolor='#0d1526', plot_bgcolor='#0d1526', showlegend=False)
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(x=model_names, y=rmse_scores, title='RMSE by Model ($)',
                     color=rmse_scores, color_continuous_scale='reds_r',
                     template='plotly_dark', text=[f"${v:,.0f}" for v in rmse_scores])
        fig.update_layout(paper_bgcolor='#0d1526', plot_bgcolor='#0d1526', showlegend=False)
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Actual vs Predicted
    st.markdown("#### Actual vs Predicted — Select Model")
    selected = st.selectbox("", model_names)
    y_pred = results[selected]['y_pred']
    y_true = results[selected]['y_test']

    fig = px.scatter(x=y_true, y=y_pred, opacity=0.4,
                     labels={'x': 'Actual Price ($100k)', 'y': 'Predicted Price ($100k)'},
                     title=f'{selected} — Actual vs Predicted',
                     color_discrete_sequence=['#22d3ee'], template='plotly_dark')
    fig.add_shape(type='line', x0=y_true.min(), y0=y_true.min(),
                  x1=y_true.max(), y1=y_true.max(),
                  line=dict(color='#a78bfa', width=2, dash='dash'))
    fig.update_layout(paper_bgcolor='#0d1526', plot_bgcolor='#0d1526')
    st.plotly_chart(fig, use_container_width=True)

    # Residuals
    residuals = y_pred - y_true
    fig2 = px.histogram(x=residuals, nbins=60, title='Residual Distribution',
                        color_discrete_sequence=['#a78bfa'], template='plotly_dark')
    fig2.update_layout(paper_bgcolor='#0d1526', plot_bgcolor='#0d1526')
    st.plotly_chart(fig2, use_container_width=True)

# ─── PREDICTOR ──────────────────────────────────────────────────
elif page == "🔮 Predict Price":
    st.markdown('<div class="section-title">Live House Price Predictor</div>', unsafe_allow_html=True)
    st.markdown("Adjust the sliders to match your house details and get an instant price estimate.")
    st.divider()

    col_model, _ = st.columns([1, 2])
    with col_model:
        model_choice = st.selectbox("🤖 Choose Model", list(results.keys()),
                                    index=list(results.keys()).index(best_model_name))

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📍 Location")
        latitude  = st.slider("Latitude",  float(df.Latitude.min()),  float(df.Latitude.max()),  35.0)
        longitude = st.slider("Longitude", float(df.Longitude.min()), float(df.Longitude.max()), -119.0)

        st.markdown("#### 🏡 Property")
        house_age  = st.slider("House Age (years)", 1, 52, 20)
        ave_rooms  = st.slider("Avg Rooms per Household", 1.0, 20.0, 5.5)
        ave_bedrms = st.slider("Avg Bedrooms per Household", 0.5, 5.0, 1.1)

    with col2:
        st.markdown("#### 👥 Demographics")
        med_inc    = st.slider("Median Income ($10k units)", 0.5, 15.0, 4.0)
        population = st.slider("Block Population", 3, 35682, 1200)
        ave_occup  = st.slider("Avg Occupancy per Household", 1.0, 20.0, 3.0)

    # Predict
    input_data = pd.DataFrame([[med_inc, house_age, ave_rooms, ave_bedrms,
                                 population, ave_occup, latitude, longitude]],
                               columns=df.drop('Price', axis=1).columns)

    model_info = results[model_choice]
    if model_info['scaled']:
        input_scaled = scaler.transform(input_data)
        prediction = model_info['model'].predict(input_scaled)[0]
    else:
        prediction = model_info['model'].predict(input_data)[0]

    prediction = max(0, prediction)

    st.divider()
    st.markdown(f"""
    <div class="prediction-box">
        <div class="label">Estimated House Price</div>
        <div class="price">${prediction * 100000:,.0f}</div>
        <div class="label">using {model_choice} &nbsp;|&nbsp; R² = {model_info['r2']*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Feature importance for tree models
    if model_choice in ['Random Forest', 'Gradient Boosting']:
        st.markdown("#### 🔍 Feature Importance")
        feat_imp = pd.DataFrame({
            'Feature': df.drop('Price', axis=1).columns,
            'Importance': model_info['model'].feature_importances_
        }).sort_values('Importance', ascending=True)

        fig = px.bar(feat_imp, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='viridis',
                     template='plotly_dark')
        fig.update_layout(paper_bgcolor='#0d1526', plot_bgcolor='#0d1526')
        st.plotly_chart(fig, use_container_width=True)