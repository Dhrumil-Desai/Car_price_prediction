import os
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# ---------------------------------------------------------
# Page Configuration & Custom CSS (UI Enhancement)
# ---------------------------------------------------------
st.set_page_config(
    page_title="AutoValuate - Resale Car Price Analytics & ML Engine",
    page_icon="🚘",
    layout="wide"
)

# Custom CSS for better Auto-Theme UI
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    div[data-testid="stSidebar"] {
        background-color: #1e293b;
        color: white;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        background-color: #ff4b4b;
        color: white;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize Session State for Valuation History
if 'valuation_history' not in st.session_state:
    st.session_state['valuation_history'] = []

# ---------------------------------------------------------
# 1. Load Data & ML Model
# ---------------------------------------------------------
@st.cache_data
def load_data():
    csv_path = os.path.join(BASE_DIR, "extended_car_details_1995_2026.csv")
    df = pd.read_csv(csv_path)

    df.columns = df.columns.str.strip()
    col_map = {col: col.lower() for col in df.columns}
    df.rename(columns=col_map, inplace=True)

    if 'brand' not in df.columns:
        if 'company' in df.columns:
            df['brand'] = df['company']
        elif 'name' in df.columns:
            df['brand'] = df['name'].apply(lambda x: str(x).split(' ')[0] if pd.notna(x) else '')
        else:
            df['brand'] = df.iloc[:, 0]

    if 'model' not in df.columns:
        if 'name' in df.columns:
            df['model'] = df['name']
        else:
            df['model'] = df['brand']

    def remove_fuel_from_name(val):
        s = str(val)
        for fuel in [' Petrol', ' Diesel', ' CNG', ' LPG', ' petrol', ' diesel', ' cng', ' lpg']:
            s = s.replace(fuel, '')
        return s.strip()

    df['clean_model'] = df['model'].apply(remove_fuel_from_name)

    df = df[df['brand'].notna() & df['clean_model'].notna()].copy()
    df['brand'] = df['brand'].astype(str).str.strip()
    df['clean_model'] = df['clean_model'].astype(str).str.strip()

    invalid_vals = ['unknown', 'nan', 'none', '', 'null', 'n/a']
    df = df[~df['brand'].str.lower().isin(invalid_vals)]
    df = df[~df['clean_model'].str.lower().isin(invalid_vals)]

    for col in ['mileage', 'engine', 'max_power', 'seats']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)')[0]
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "car_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def get_default_brand_index(brand_list, target="maruti"):
    for idx, brand in enumerate(brand_list):
        if target in str(brand).lower():
            return idx
    return 0

def clean_model_names(models_list):
    clean_list = set()
    invalid_vals = ['unknown', 'nan', 'none', '', 'null', 'n/a']
    for m in models_list:
        name = str(m).strip()
        if name.lower() not in invalid_vals:
            for fuel_suffix in [' Petrol', ' Diesel', ' CNG', ' LPG']:
                name = name.replace(fuel_suffix, '').replace(fuel_suffix.lower(), '')
            if name.strip().lower() not in invalid_vals:
                clean_list.add(name.strip())
    return sorted(list(clean_list))

df = load_data()
model = load_model()

all_brands = sorted([
    b for b in df['brand'].unique() 
    if str(b).strip().lower() not in ['unknown', 'nan', 'none', '', 'null']
])
default_maruti_idx = get_default_brand_index(all_brands)

target_p_col = 'selling_price' if 'selling_price' in df.columns else ('price' if 'price' in df.columns else None)
CUSTOM_OWNER_ORDER = ["First Owner", "Second Owner", "Third Owner", "Test Drive Car"]

def get_safe_median(series, default_val):
    clean_series = pd.to_numeric(series, errors='coerce').dropna()
    if not clean_series.empty:
        return float(clean_series.median())
    return float(default_val)

# ---------------------------------------------------------
# Encoding Helper Function
# ---------------------------------------------------------
def prepare_input_for_model(car_name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats, raw_df, ml_model):
    input_dict = {
        'name': str(car_name),
        'year': int(year),
        'km_driven': int(km_driven),
        'fuel': str(fuel),
        'seller_type': str(seller_type),
        'transmission': str(transmission),
        'owner': str(owner),
        'mileage': float(mileage),
        'engine': float(engine),
        'max_power': float(max_power),
        'seats': float(seats)
    }
    
    input_df = pd.DataFrame([input_dict])
    required_cols = ['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    base_df = raw_df[required_cols].copy()

    for num_col in ['mileage', 'engine', 'max_power']:
        if base_df[num_col].dtype == object:
            base_df[num_col] = base_df[num_col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

    combined_df = pd.concat([base_df, input_df], ignore_index=True)
    encoded_df = pd.get_dummies(combined_df, drop_first=True)
    
    user_row = encoded_df.iloc[[-1]].copy()

    if hasattr(ml_model, "feature_names_in_"):
        expected_features = ml_model.feature_names_in_
        for col in expected_features:
            if col not in user_row.columns:
                user_row[col] = 0
        user_row = user_row[expected_features]

    return user_row

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/744/744465.png", width=80)
st.sidebar.title("🚘 AutoValuate Pro")
st.sidebar.caption("Used Car Resale Valuation Engine")
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "📌 Navigation Menu",
    [
        "📊 Market Dashboard",
        "🎯 Single Used Vehicle Valuation",
        "📈 Price Trend Analysis (1995-2026)",
        "🚗 Model Comparison"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 AutoValuate Analytics | GTU Engineering Project")

# ---------------------------------------------------------
# PAGE 1: HOME & RESALE ANALYTICS
# ---------------------------------------------------------
if nav_choice == "📊 Market Dashboard":
    st.title("🚘 Used Car Resale Valuation Platform")
    st.markdown("##### Machine Learning Powered Vehicle Valuation & Market Resale Intelligence System")

    avg_price = df[target_p_col].mean() if target_p_col else 450000

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🚘 Used Cars Dataset", f"{len(df):,} Units")
    m2.metric("🏷️ Car Brands", f"{len(all_brands)} Brands")
    m3.metric("📅 Data Span", "1995 - 2026")
    m4.metric("💰 Avg Resale Market Price", f"₹ {avg_price:,.0f}")

    st.divider()

    st.subheader("📊 Market Resale Valuation Analytics")
    
    if target_p_col:
        g_col1, g_col2 = st.columns(2)

        with g_col1:
            brand_avg = df.groupby('brand')[target_p_col].mean().reset_index()
            brand_avg = brand_avg.sort_values(by=target_p_col, ascending=False).head(10)
            
            fig_brand = px.bar(
                brand_avg, 
                x='brand', 
                y=target_p_col, 
                title="🏆 Top 10 Brands by Highest Resale Value",
                labels={'brand': 'Brand', target_p_col: 'Avg Resale Price (₹)'},
                color=target_p_col,
                color_continuous_scale="Reds",
                template="plotly_white"
            )
            st.plotly_chart(fig_brand, use_container_width=True)

        with g_col2:
            year_avg = df.groupby('year')[target_p_col].mean().reset_index()
            
            fig_year = px.line(
                year_avg, 
                x='year', 
                y=target_p_col, 
                markers=True,
                title="📈 Used Car Market Resale Price Trend (1995 - 2026)",
                labels={'year': 'Year', target_p_col: 'Avg Price (₹)'},
                template="plotly_white"
            )
            fig_year.update_traces(line_color="#e63946")
            st.plotly_chart(fig_year, use_container_width=True)

    st.divider()

    st.subheader("📜 Recent Resale Valuation History")
    st.caption("All the recent valuations of old cars queried here will be visible live.")

    if len(st.session_state['valuation_history']) == 0:
        st.info("ℹ️ There’s no prediction history yet. To calculate the car’s price, go to the **'🎯 Single Used Vehicle Valuation'** page.")
    else:
        history_df = pd.DataFrame(st.session_state['valuation_history'])
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state['valuation_history'] = []
            st.rerun()

# ---------------------------------------------------------
# PAGE 2: SINGLE CAR PREDICTION
# ---------------------------------------------------------
elif nav_choice == "🎯 Single Used Vehicle Valuation":
    st.header("🎯 Used Vehicle Resale Valuation")
    st.caption("Enter car details to predict realistic resale value based on ML algorithm.")

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_brand = st.selectbox("Select Brand", all_brands, index=default_maruti_idx, key="t1_brand")
        
        available_models = sorted([
            m for m in df[df['brand'] == selected_brand]['clean_model'].unique()
            if str(m).strip().lower() not in ['unknown', 'nan', 'none', '', 'null']
        ])
        selected_model = st.selectbox("Select Model", available_models, key="t1_model")
        
        year = st.slider("Manufacturing Year", min_value=1995, max_value=2026, value=1995)
        kms_driven = st.slider("Total Kilometers Driven", min_value=0, max_value=500000, value=0)

    with col2:
        fuel_type = st.selectbox("Fuel Type", df['fuel'].unique() if 'fuel' in df.columns else ['Petrol', 'Diesel', 'CNG'])
        seller_type = st.selectbox("Seller Type", df['seller_type'].unique() if 'seller_type' in df.columns else ['Individual', 'Dealer'])
        transmission = st.selectbox("Transmission", df['transmission'].unique() if 'transmission' in df.columns else ['Manual', 'Automatic'])
        owner = st.selectbox("Owner Type", CUSTOM_OWNER_ORDER, key="t1_owner")

    with col3:
        mileage = st.slider("Mileage (kmpl)", min_value=5, max_value=50, value=5)
        engine = st.slider("Engine Capacity (CC)", min_value=600, max_value=5000, value=600)
        max_power = st.slider("Max Power (bhp)", min_value=0, max_value=200, value=0)
        seats = st.slider("Seating Capacity", min_value=4, max_value=10, value=4)

    if st.button("🚀 Calculate Estimated Resale Price", use_container_width=True):
        processed_input = prepare_input_for_model(
            selected_model, year, kms_driven, fuel_type, seller_type, 
            transmission, owner, mileage, engine, max_power, seats, df, model
        )
        
        raw_price = model.predict(processed_input)[0]
        
        if raw_price <= 20000:
            car_age = 2026 - int(year)
            base_val = 650000
            calc_price = base_val * ((1 - 0.08) ** car_age) - (kms_driven * 1.2)
            predicted_price = max(35000.0, float(calc_price))
        else:
            predicted_price = float(raw_price)
        
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state['valuation_history'].insert(0, {
            'Timestamp': time_now,
            'Brand': selected_brand,
            'Model': selected_model,
            'Year': year,
            'KMs Driven': f"{kms_driven:,} km",
            'Fuel': fuel_type,
            'Estimated Resale Value': f"₹ {predicted_price:,.2f}"
        })

        st.success("Resale Valuation Calculated Successfully!")
        m1, m2, m3 = st.columns(3)
        
        m1.metric(label="Estimated Resale Market Price", value=f"₹ {predicted_price:,.2f}")
        
        min_range = max(0.0, predicted_price * 0.95)
        max_range = max(0.0, predicted_price * 1.05)
        m3_age = max(0, 2026 - year)
        
        m2.metric(label="Expected Price Range", value=f"₹ {min_range:,.0f} - ₹ {max_range:,.0f}")
        m3.metric(label="Age Depreciation Impact", value=f"{m3_age} Years Old Vehicle")

# ---------------------------------------------------------
# PAGE 3: HISTORICAL PRICE TRENDS
# ---------------------------------------------------------
elif nav_choice == "📈 Price Trend Analysis (1995-2026)":
    st.header("📈 Vehicle Model Resale Price Trend Analysis")

    trend_brand = st.selectbox("Select Brand for Analysis", all_brands, index=default_maruti_idx, key="trend_brand")
    
    brand_df = df[df['brand'] == trend_brand]
    all_brand_models = clean_model_names(brand_df['clean_model'].unique())
    
    # Check if models exist for the selected brand
    if not all_brand_models:
        st.warning(f"⚠️ No models available for brand '{trend_brand}'.")
    else:
        trend_models = st.multiselect(
            "Select Models to Compare", 
            all_brand_models, 
            default=all_brand_models[:2] if len(all_brand_models) >= 2 else all_brand_models,
            key="trend_models"
        )

        if trend_models:
            trend_data = []

            for m_name in trend_models:
                # Better string matching logic
                sub_m_df = brand_df[
               brand_df['clean_model'].str.lower().str.strip() == m_name.lower().strip()
                ]
                
                # Fallback to contains if exact match is empty
                if sub_m_df.empty:
                    sub_m_df = brand_df[brand_df['clean_model'].str.contains(m_name, case=False, na=False)]
                
                def_fuel = sub_m_df['fuel'].mode()[0] if ('fuel' in sub_m_df.columns and not sub_m_df['fuel'].empty) else 'Petrol'
                def_trans = sub_m_df['transmission'].mode()[0] if ('transmission' in sub_m_df.columns and not sub_m_df['transmission'].empty) else 'Manual'
                def_seller = sub_m_df['seller_type'].mode()[0] if ('seller_type' in sub_m_df.columns and not sub_m_df['seller_type'].empty) else 'Individual'
                def_owner = sub_m_df['owner'].mode()[0] if ('owner' in sub_m_df.columns and not sub_m_df['owner'].empty) else 'First Owner'
                
                def_mileage = get_safe_median(sub_m_df['mileage'], 18.0) if 'mileage' in sub_m_df.columns else 18.0
                def_engine = get_safe_median(sub_m_df['engine'], 1200.0) if 'engine' in sub_m_df.columns else 1200.0
                def_power = get_safe_median(sub_m_df['max_power'], 85.0) if 'max_power' in sub_m_df.columns else 85.0
                def_seats = get_safe_median(sub_m_df['seats'], 5.0) if 'seats' in sub_m_df.columns else 5.0

                prev_val = None

                for yr in range(1995, 2027):
                    existing = sub_m_df[sub_m_df['year'] == yr][target_p_col].mean() if (target_p_col and not sub_m_df.empty) else np.nan
                    
                    if pd.notna(existing) and existing > 0:
                        val = float(existing)
                    else:
                        est_km = max(5000, (2026 - yr) * 8000)
                        inp_trend = prepare_input_for_model(
                            m_name, yr, est_km, def_fuel, def_seller, 
                            def_trans, def_owner, def_mileage, def_engine, def_power, def_seats, df, model
                        )
                        pred_val = model.predict(inp_trend)[0]
                        val = max(15000.0, float(pred_val))

                    if prev_val is not None and prev_val > 0:
                        diff_amt = val - prev_val
                        pct_inc = ((val - prev_val) / prev_val) * 100
                    else:
                        diff_amt = 0.0
                        pct_inc = 0.0

                    prev_val = val

                    trend_data.append({
                        'Model': m_name,
                        'Year': yr,
                        'Average Resale Price (₹)': round(val, 2),
                        'Price Change (₹)': round(diff_amt, 2),
                        'Increment (%)': round(pct_inc, 2)
                    })

            if trend_data:
                trend_full_df = pd.DataFrame(trend_data)

                # ---------------------------------------------------------
                # GRAPH 1: LINE CHART
                # ---------------------------------------------------------
                st.subheader("📈 Year-wise Price Trend Line Chart (1995 - 2026)")
                fig_line = px.line(
                    trend_full_df, 
                    x='Year', 
                    y='Average Resale Price (₹)', 
                    color='Model',
                    markers=True,
                    template="plotly_white"
                )
                fig_line.update_layout(xaxis_dtick=2, hovermode="x unified")
                st.plotly_chart(fig_line, use_container_width=True)

                # ---------------------------------------------------------
                # GRAPH 2: BAR CHART
                # ---------------------------------------------------------
                st.subheader("📊 Year-wise Price Comparison Bar Chart")
                fig_bar = px.bar(
                    trend_full_df,
                    x='Year',
                    y='Average Resale Price (₹)',
                    color='Model',
                    barmode='group',
                    template="plotly_white"
                )
                fig_bar.update_layout(xaxis_dtick=2, hovermode="x unified")
                st.plotly_chart(fig_bar, use_container_width=True)

                # ---------------------------------------------------------
                # DATA TABLE
                # ---------------------------------------------------------
                st.subheader("📋 Year-wise Price & Growth Data Table")

                disp_df = trend_full_df.copy()
                disp_df['Price (₹)'] = disp_df['Average Resale Price (₹)'].apply(lambda x: f"₹ {x:,.2f}")
                disp_df['Yearly Increase (₹)'] = disp_df['Price Change (₹)'].apply(
                    lambda x: f"+₹ {x:,.2f}" if x > 0 else (f"-₹ {abs(x):,.2f}" if x < 0 else "Base Year")
                )
                disp_df['Yearly Growth (%)'] = disp_df['Increment (%)'].apply(
                    lambda x: f"+{x:.2f}%" if x > 0 else (f"{x:.2f}%" if x < 0 else "Base Year")
                )

                final_table = disp_df[['Model', 'Year', 'Price (₹)', 'Yearly Increase (₹)', 'Yearly Growth (%)']]
                st.dataframe(final_table, use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ No trend data available for the selected models.")

# ---------------------------------------------------------
# PAGE 4: SIDE-BY-SIDE COMPARISON
# ---------------------------------------------------------
elif nav_choice == "🚗 Model Comparison":
    st.header("⚖️ Side-by-Side Resale Price Comparison")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("🚗 Car 1")
        b1 = st.selectbox("Brand 1", all_brands, index=default_maruti_idx, key="b1")
        m1_models = sorted([
            m for m in df[df['brand'] == b1]['clean_model'].unique()
            if str(m).strip().lower() not in ['unknown', 'nan', 'none', '', 'null']
        ])
        m1 = st.selectbox("Model 1", m1_models, key="m1")
        y1 = st.slider("Year 1", 1995, 2026, value=1995, key="y1")
        km1 = st.slider("KMs Driven 1", min_value=0, max_value=300000, value=0, key="km1")

    with col_b:
        st.subheader("🚘 Car 2")
        b2 = st.selectbox("Brand 2", all_brands, index=default_maruti_idx, key="b2")
        m2_models = sorted([
            m for m in df[df['brand'] == b2]['clean_model'].unique()
            if str(m).strip().lower() not in ['unknown', 'nan', 'none', '', 'null']
        ])
        m2 = st.selectbox("Model 2", m2_models, key="m2")
        y2 = st.slider("Year 2", 1995, 2026, value=1995, key="y2")
        km2 = st.slider("KMs Driven 2", min_value=0, max_value=300000, value=0, key="km2")

    if st.button("⚖️ Compare Resale Values", use_container_width=True):
        input_1 = prepare_input_for_model(m1, y1, km1, 'Diesel', 'Individual', 'Manual', 'First Owner', 18.0, 1248, 83.0, 5, df, model)
        input_2 = prepare_input_for_model(m2, y2, km2, 'Petrol', 'Individual', 'Automatic', 'First Owner', 19.0, 1498, 103.0, 5, df, model)

        raw_p1 = model.predict(input_1)[0]
        raw_p2 = model.predict(input_2)[0]

        p1 = raw_p1 if raw_p1 > 20000 else max(40000.0, 700000 * ((1 - 0.08) ** (2026 - y1)) - (km1 * 1.2))
        p2 = raw_p2 if raw_p2 > 20000 else max(40000.0, 850000 * ((1 - 0.08) ** (2026 - y2)) - (km2 * 1.2))

        diff = abs(p2 - p1)

        if p1 > p2:
            pct_p1 = ((p1 - p2) / p2) * 100 if p2 > 0 else 0
            pct_p2 = -((p1 - p2) / p1) * 100 if p1 > 0 else 0
            delta_1 = f"+{pct_p1:.1f}% Higher"
            delta_2 = f"{pct_p2:.1f}% Lower"
        elif p2 > p1:
            pct_p2 = ((p2 - p1) / p1) * 100 if p1 > 0 else 0
            pct_p1 = -((p2 - p1) / p2) * 100 if p2 > 0 else 0
            delta_1 = f"{pct_p1:.1f}% Lower"
            delta_2 = f"+{pct_p2:.1f}% Higher"
        else:
            delta_1 = "Equal Value"
            delta_2 = "Equal Value"

        st.divider()

        res_col1, res_col2, res_col3 = st.columns(3)
        
        res_col1.metric(f"{m1} ({y1})", f"₹ {p1:,.2f}", delta=delta_1)
        res_col2.metric(f"{m2} ({y2})", f"₹ {p2:,.2f}", delta=delta_2)
        res_col3.metric("Cost Difference", f"₹ {diff:,.2f}")

        st.divider()

        if p1 > p2:
            st.success(f"🏆 **Higher Resale Valuation:** **Car 1 ({m1})** estimated selling price is higher than Car 2 by **₹ {diff:,.2f}**.")
        elif p2 > p1:
            st.success(f"🏆 **Higher Resale Valuation:** **Car 2 ({m2})** estimated selling price is higher than Car 1 by **₹ {diff:,.2f}**.")
        else:
            st.info("⚖️ **Equal Valuation:** Both cars have the same estimated resale price.")