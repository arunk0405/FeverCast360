
# app.py
# FeverCast360 - ML-Powered Fever Outbreak Prediction Dashboard
# 
# Key Features:
# - Dynamic geocoding: All cities from ML pipeline data are automatically geocoded using OpenStreetMap
# - No static data: All components (map, pharma, government views) use live data from Firebase
# - Smart caching: Already geocoded cities are reused to avoid unnecessary API calls
# - Real-time visualization: Interactive Leaflet maps with actual city boundaries
#
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import time
from datetime import datetime, timezone
import os

import prediction   # your prediction.py (must expose run_pipeline_and_return)
from db_utils import (
    init_db, save_predictions, fetch_all_predictions, fetch_city_prediction,
    upsert_region_metadata, upsert_pharma_stock, get_region_metadata
)

st.set_page_config(page_title="FeverCast360", layout="wide", initial_sidebar_state="collapsed")

# Professional styling with white background
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #FFFFFF;
        color: #1F2937;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA;
    }
    
    /* All text elements */
    .stMarkdown, .stText, p, span, div, label {
        color: #1F2937 !important;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 600;
        color: #1F2937 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 14px;
        font-weight: 500;
        color: #6B7280 !important;
    }
    
    /* Headers */
    h1 {
        color: #111827 !important;
        font-weight: 700;
        padding-bottom: 1rem;
        border-bottom: 3px solid #3B82F6;
    }
    
    h2 {
        color: #1F2937 !important;
        font-weight: 600;
        margin-top: 2rem;
    }
    
    h3 {
        color: #374151 !important;
        font-weight: 600;
    }
    
    h4, h5, h6 {
        color: #374151 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F3F4F6;
        padding: 8px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #FFFFFF;
        border-radius: 6px;
        color: #4B5563 !important;
        font-weight: 500;
        padding: 0 24px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6;
        color: #FFFFFF !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #3B82F6;
        color: white !important;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #2563EB;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
    }
    
    /* Text inputs and labels */
    .stTextInput > label, .stNumberInput > label, .stSelectbox > label, 
    .stSlider > label, .stCheckbox > label, .stRadio > label {
        color: #374151 !important;
        font-weight: 500;
    }
    
    .stTextInput > div > div > input {
        border-radius: 6px;
        border: 1px solid #D1D5DB;
        padding: 0.5rem;
        color: #1F2937 !important;
        background-color: #FFFFFF;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #9CA3AF !important;
    }
    
    /* Slider labels and values */
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"],
    .stSlider [data-testid="stThumbValue"] {
        color: #374151 !important;
    }
    
    /* Checkbox text */
    .stCheckbox > label > div > p {
        color: #374151 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        color: #374151 !important;
        font-weight: 500;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #F9FAFB;
        border: 2px dashed #D1D5DB;
        border-radius: 8px;
        padding: 2rem;
    }
    
    [data-testid="stFileUploader"] label {
        color: #374151 !important;
    }
    
    /* Info/Warning/Error/Success boxes */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
    }
    
    .stAlert p, .stAlert div {
        color: #1F2937 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #3B82F6;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #3B82F6 !important;
    }
    
    /* Cards/Containers */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Dataframe */
    .dataframe {
        border: 1px solid #E5E7EB !important;
        border-radius: 8px;
        color: #1F2937 !important;
    }
    
    .dataframe th {
        background-color: #F3F4F6 !important;
        color: #111827 !important;
        font-weight: 600;
    }
    
    .dataframe td {
        color: #374151 !important;
    }
    
    /* Column headers in dataframes */
    .stDataFrame [data-testid="stDataFrameResizable"] {
        color: #1F2937 !important;
    }
    
    /* Help text */
    .stTooltipIcon {
        color: #6B7280 !important;
    }
    
    /* Caption text */
    .stCaption {
        color: #6B7280 !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background-color: #F3F4F6 !important;
    }
    
    code {
        color: #DC2626 !important;
        background-color: #F3F4F6 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("FeverCast360 — Unified Dashboard")

# initialize DB
init_db()

# Helper function to geocode any city name
def geocode_city(city_name):
    """
    Fetch latitude, longitude, and state for any city using OpenStreetMap Nominatim API.
    Returns (lat, lon, state, population_estimate) or None if not found.
    """
    try:
        import requests
        url = "https://nominatim.openstreetmap.org/search"
        
        # Try multiple search strategies, being more specific to India
        search_queries = [
            f"{city_name}, India",  # Most general, let OSM find the best match
            f"{city_name} city, India",  # Specify as a city
            f"{city_name} district, India",  # Try as district
        ]
        
        headers = {'User-Agent': 'FeverCast360/1.0'}
        
        for query in search_queries:
            params = {
                'q': query,
                'format': 'json',
                'limit': 3,  # Get top 3 results to pick the best one
                'addressdetails': 1,
                'countrycodes': 'in'  # Restrict to India only
            }
            
            try:
                response = requests.get(url, params=params, headers=headers, timeout=10)
                
                # Handle rate limiting
                if response.status_code == 429:
                    time.sleep(2)
                    response = requests.get(url, params=params, headers=headers, timeout=10)
                
                data = response.json()
                
                if data and len(data) > 0:
                    # Pick the best result - prefer cities over other types
                    best_result = None
                    for result in data:
                        place_type = result.get('type', '')
                        osm_type = result.get('osm_type', '')
                        address = result.get('address', {})
                        
                        # Check if the result actually contains our city name
                        display_name = result.get('display_name', '').lower()
                        if city_name.lower() not in display_name:
                            continue
                        
                        # Prefer administrative boundaries and cities
                        if place_type in ['city', 'town', 'municipality', 'administrative'] or osm_type == 'relation':
                            best_result = result
                            break
                        elif not best_result:
                            best_result = result
                    
                    if best_result:
                        lat = float(best_result['lat'])
                        lon = float(best_result['lon'])
                        
                        # Extract state from address details
                        address = best_result.get('address', {})
                        state = (address.get('state') or 
                                address.get('state_district') or 
                                'India')
                        
                        # Estimate population based on place type and importance
                        place_type = best_result.get('type', '')
                        importance = float(best_result.get('importance', 0.5))
                        
                        # More accurate population estimates based on city type
                        if place_type in ['city', 'town', 'municipality']:
                            population_estimate = int(importance * 15_000_000)
                        elif place_type in ['village', 'suburb', 'locality']:
                            population_estimate = int(importance * 5_000_000)
                        else:
                            population_estimate = int(importance * 10_000_000)
                        
                        # Ensure minimum population
                        population_estimate = max(50_000, population_estimate)
                        
                        print(f"✓ Geocoded {city_name} → {state} ({lat}, {lon}) [{place_type}]")
                        return (lat, lon, state, population_estimate)
                    
            except Exception as e:
                print(f"Attempt failed for query '{query}': {e}")
                continue
            
            # Small delay between attempts
            time.sleep(0.5)
        
        print(f"✗ Could not geocode {city_name} with any search strategy")
        return None
        
    except Exception as e:
        print(f"Error geocoding {city_name}: {e}")
        return None

# ML Pipeline button in sidebar
with st.sidebar:
    st.markdown("### ⚙️ ML Pipeline")
    if st.button("🚀 Open ML Pipeline", type="primary", use_container_width=True):
        st.session_state['show_ml_pipeline'] = not st.session_state.get('show_ml_pipeline', False)
    
    if st.session_state.get('show_ml_pipeline', False):
        st.markdown("---")
        st.info("👈 ML Pipeline panel is open. Scroll down to use it.")

# Initialize session state
if 'show_ml_pipeline' not in st.session_state:
    st.session_state['show_ml_pipeline'] = False

# ----------------------
# ML PIPELINE OVERLAY (Appears above dashboard)
# ----------------------
if st.session_state.get('show_ml_pipeline', False):
    # Add custom CSS for white-colored components
    st.markdown("""
    <style>
    /* Override Streamlit's default button styling for all buttons */
    button[kind="secondary"] {
        background-color: white !important;
        color: #43e97b !important;
        border: 2px solid white !important;
    }
    button[kind="secondary"]:hover {
        background-color: #f0f0f0 !important;
        color: #38f9d7 !important;
    }
    
    /* Style file uploader */
    [data-testid="stFileUploader"] {
        background: transparent !important;
    }
    [data-testid="stFileUploader"] section {
        border: 2px dashed rgba(255, 255, 255, 0.8) !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: white !important;
        color: #667eea !important;
        border: none !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: #f0f0f0 !important;
    }
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] p {
        color: rgba(255, 255, 255, 0.95) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Use a container to separate ML Pipeline from the rest of the page
    st.markdown("---")
    
    # Close button at the top
    col_close = st.columns([10, 1])
    with col_close[1]:
        if st.button("✕ Close", key="close_ml_btn", help="Close ML Pipeline", type="secondary"):
            st.session_state['show_ml_pipeline'] = False
            st.rerun()
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                padding: 2rem; border-radius: 12px 12px 0 0; margin: -1rem -1rem 1rem -1rem;'>
        <h1 style='color: white; margin: 0; font-size: 32px;'>⚙️ ML Pipeline Control Center</h1>
        <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 16px;'>
            Run predictive models and save results to Firebase cloud database
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data Requirements Info
    st.markdown("""
    <div style='background-color: #F0F9FF; padding: 1.5rem; border-radius: 10px; 
                border-left: 4px solid #3B82F6; margin-bottom: 2rem;'>
        <h4 style='margin: 0 0 8px 0; color: #1E40AF;'>📊 Data Requirements</h4>
        <p style='margin: 0; color: #1F2937;'>
            Upload a preprocessed CSV file containing: <strong>Region</strong>, <strong>outbreak_label</strong>, 
            <strong>fever_type</strong>, and relevant feature columns.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 10px; 
                    border: 2px dashed rgba(255, 255, 255, 0.5); margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 8px 0; color: white;'>📁 Upload Data File</h4>
            <p style='margin: 0 0 12px 0; color: rgba(255, 255, 255, 0.9); font-size: 14px;'>
                Select your preprocessed CSV file for ML analysis
            </p>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose CSV file", type=["csv"], 
                                         label_visibility="collapsed",
                                         help="Upload your preprocessed dataset for ML analysis")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background-color: white; padding: 1.5rem; border-radius: 10px; 
                    border: 1px solid #E5E7EB;'>
        """, unsafe_allow_html=True)
        
        st.markdown("### ⚙️ Configuration")
        threshold = st.slider("Stage 1 Threshold", 0.0, 1.0, 0.5, 0.01,
                             help="Probability threshold to trigger Stage 2 classification")
        use_xg = st.checkbox("Use XGBoost", value=False,
                            help="Enable XGBoost for advanced classification (requires installation)")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    run_btn = st.button("🚀 Run ML Pipeline & Save Results", type="primary", use_container_width=True)

    st.markdown("---")
    
    # Metadata editor with better styling
    st.markdown("### 🗺️ Region Metadata Management")
    st.markdown("""
    <div style='background-color: #FFFBEB; padding: 1rem; border-radius: 8px; 
                border-left: 4px solid #F59E0B; margin-bottom: 1rem;'>
        <p style='margin: 0; color: #92400E; font-size: 14px;'>
            ℹ️ Optionally add geographical metadata for better visualization and analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("➕ Add / Update Region Metadata", expanded=False):
        m_col1, m_col2 = st.columns(2)
        
        with m_col1:
            m_region = st.text_input("Region Name", placeholder="e.g., Chennai", key="meta_region")
            m_lat = st.text_input("Latitude", placeholder="e.g., 13.0827", key="meta_lat")
            m_lon = st.text_input("Longitude", placeholder="e.g., 80.2707", key="meta_lon")
        
        with m_col2:
            m_pop = st.text_input("Population", placeholder="e.g., 4646732", key="meta_pop")
            m_state = st.text_input("State", placeholder="e.g., Tamil Nadu", key="meta_state")
        
        if st.button("💾 Save Metadata", type="primary"):
            try:
                lat = float(m_lat) if m_lat else None
                lon = float(m_lon) if m_lon else None
                pop = int(m_pop) if m_pop else None
                state = m_state if m_state else None
                
                if m_region:
                    upsert_region_metadata(m_region, lat, lon, pop, state)
                    st.success(f"✅ Region metadata saved for {m_region}")
                else:
                    st.error("❌ Region name is required")
            except ValueError as e:
                st.error(f"❌ Invalid input: {str(e)}")
            except Exception as e:
                st.error(f"❌ Failed to save metadata: {e}")
    
    # Add Re-Geocode button for missing coordinates
    with st.expander("🔄 Re-Geocode Regions Without Coordinates", expanded=False):
        st.markdown("""
        <div style='background-color: #FEF3C7; padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #F59E0B; margin-bottom: 1rem;'>
            <p style='margin: 0; color: #92400E; font-size: 14px;'>
                ⚠️ Use this if some regions are missing from the map. This will attempt to geocode 
                all regions that don't have coordinates yet.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🌍 Geocode Missing Regions", type="secondary"):
            df_all = fetch_all_predictions()
            if not df_all.empty:
                regions_to_geocode = []
                for region in df_all['region'].unique():
                    metadata = get_region_metadata(region)
                    if not metadata or not metadata.get('lat') or not metadata.get('lon'):
                        regions_to_geocode.append(region)
                
                if regions_to_geocode:
                    st.info(f"Found {len(regions_to_geocode)} regions without coordinates: {', '.join(regions_to_geocode)}")
                    
                    progress = st.progress(0)
                    status = st.empty()
                    success_count = 0
                    
                    for idx, region in enumerate(regions_to_geocode):
                        progress.progress((idx + 1) / len(regions_to_geocode))
                        status.text(f"Geocoding {region}... ({idx + 1}/{len(regions_to_geocode)})")
                        
                        coords = geocode_city(region)
                        if coords:
                            lat, lon, state, pop = coords
                            upsert_region_metadata(region, lat, lon, pop, state)
                            success_count += 1
                            st.success(f"✓ Geocoded {region}")
                        else:
                            st.error(f"✗ Failed to geocode {region}")
                        
                        time.sleep(1)  # Respect API limits
                    
                    progress.empty()
                    status.empty()
                    st.success(f"✅ Successfully geocoded {success_count}/{len(regions_to_geocode)} regions!")
                else:
                    st.info("✓ All regions already have coordinates!")
            else:
                st.warning("No prediction data found. Run the ML pipeline first.")

    if run_btn:
        if uploaded_file is None:
            st.error("❌ Please upload the preprocessed CSV file first.")
        else:
            tmp_path = "tmp_input_for_ml.csv"
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("🔄 Running ML pipeline... This may take some time..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("⚙️ Loading data...")
                    progress_bar.progress(20)
                    
                    status_text.text("🤖 Training Stage 1 model (Outbreak Detection)...")
                    progress_bar.progress(40)
                    
                    status_text.text("🧠 Training Stage 2 model (Fever Classification)...")
                    progress_bar.progress(60)
                    
                    # call your provided pipeline wrapper
                    df_pred = prediction.run_pipeline_and_return(
                        input_csv=tmp_path,
                        models_dir="models",
                        output_dir="outputs",
                        threshold=threshold,
                        use_xgboost=use_xg
                    )
                    
                    status_text.text("💾 Saving predictions to Firebase...")
                    progress_bar.progress(80)

                    # Save raw predictions CSV for reference
                    ts = datetime.now(timezone.utc).isoformat()
                    # normalize columns for save_predictions
                    save_df = df_pred.rename(columns={
                        "Region": "Region",
                        "P_Outbreak": "P_Outbreak",
                        "Fever_Type": "Fever_Type",
                        "P_Type": "P_Type",
                        "Severity_Index": "Severity_Index"
                    })
                    # ensure types are correct then save to Firestore
                    save_predictions(save_df, ts)

                    # Geocode and save metadata for each region dynamically
                    status_text.text("🌍 Geocoding regions and saving metadata...")
                    unique_regions = save_df["Region"].unique()
                    geocode_progress = st.progress(0)
                    geocode_status = st.empty()
                    
                    geocoded_count = 0
                    skipped_count = 0
                    failed_regions = []
                    
                    for idx, region in enumerate(unique_regions):
                        geocode_progress.progress((idx + 1) / len(unique_regions))
                        geocode_status.text(f"Processing {region}... ({idx + 1}/{len(unique_regions)})")
                        
                        # Check if metadata already exists
                        existing_metadata = get_region_metadata(region)
                        
                        if existing_metadata and existing_metadata.get('lat') and existing_metadata.get('lon'):
                            # Use existing metadata
                            pop = existing_metadata.get('population', 1_000_000)
                            skipped_count += 1
                            st.info(f"♻️ {region}: Using cached coordinates")
                        else:
                            # Try to geocode the city
                            coords = geocode_city(region)
                            if coords:
                                lat, lon, state, pop_estimate = coords
                                upsert_region_metadata(region, lat, lon, pop_estimate, state)
                                pop = pop_estimate
                                geocoded_count += 1
                                st.success(f"✓ {region}: Geocoded successfully")
                            else:
                                # If geocoding fails, still save the region without coordinates
                                failed_regions.append(region)
                                st.warning(f"⚠️ {region}: Could not geocode - skipping map display")
                                upsert_region_metadata(region, None, None, 1_000_000, "Unknown")
                                pop = 1_000_000
                        
                        # Compute pharma stock suggestion heuristic
                        region_data = save_df[save_df["Region"] == region].iloc[0]
                        sev = float(region_data["Severity_Index"])
                        base = max(50, int(pop * sev * 0.001))
                        paracetamol = base
                        ors = int(base * 0.8)
                        antibiotics = int(base * 0.6)
                        iv_fluids = int(base * 0.3)
                        upsert_pharma_stock(region, paracetamol, ors, antibiotics, iv_fluids, ts)
                        
                        # Small delay to respect API rate limits (only if we geocoded)
                        if not existing_metadata or not existing_metadata.get('lat'):
                            time.sleep(0.5)
                    
                    geocode_progress.empty()
                    geocode_status.empty()
                    progress_bar.progress(100)
                    status_text.empty()
                    progress_bar.empty()

                    st.success("✅ ML pipeline completed successfully and results saved to Firebase!")
                    
                    # Show geocoding statistics
                    st.markdown("### 📍 Geocoding Summary")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Newly Geocoded", geocoded_count)
                    col2.metric("Cached (Reused)", skipped_count)
                    col3.metric("Failed", len(failed_regions))
                    
                    if failed_regions:
                        st.error(f"⚠️ Could not geocode: {', '.join(failed_regions)}")
                        st.info("💡 Tip: Use the 'Re-Geocode Regions Without Coordinates' tool above to retry failed regions.")
                    
                    # Display results in a professional table
                    st.markdown("### 📊 Prediction Results")
                    
                    # Add summary metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Regions", len(save_df))
                    m2.metric("High Risk", len(save_df[save_df["Severity_Index"] >= 0.4]))
                    m3.metric("Avg Severity", f"{save_df['Severity_Index'].mean():.3f}")
                    m4.metric("Outbreak Alert", len(save_df[save_df["P_Outbreak"] >= 0.5]))
                    
                    st.dataframe(
                        save_df.style.background_gradient(subset=['Severity_Index'], cmap='RdYlGn_r')
                                    .format({
                                        'P_Outbreak': '{:.3f}',
                                        'P_Type': '{:.3f}',
                                        'Severity_Index': '{:.3f}'
                                    }),
                        use_container_width=True,
                        height=400
                    )
                    
                except Exception as e:
                    st.error(f"❌ ML pipeline failed: {e}")
                    import traceback
                    with st.expander("🔍 View Error Details"):
                        st.code(traceback.format_exc())
    
    # Stop rendering here - don't show dashboard tabs when ML Pipeline is active
    st.stop()

# Show dashboard tabs (only when ML Pipeline is NOT active)
tabs = st.tabs(["🏛️ Government View", "💊 Pharma View", "👥 Public View"])
gov_tab, pharma_tab, public_tab = tabs

# ----------------------
# GOVERNMENT VIEW
# ----------------------
with gov_tab:
    # Professional header
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 12px; margin-bottom: 2rem;'>
        <h2 style='color: white; margin: 0; font-size: 28px;'>🏛️ Government Dashboard</h2>
        <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 16px;'>
            District-level outbreak monitoring & priority response actions
        </p>
    </div>
    """, unsafe_allow_html=True)

    df = fetch_all_predictions()

    # summary KPIs
    col1, col2, col3, col4 = st.columns(4)
    if df.empty:
        col1.metric("High-Risk Districts", "0/0")
        col2.metric("Active Alerts", "0")
        col3.metric("Average Risk Score", "0.00")
        col4.metric("Population At Risk", "0")
        st.info("No prediction data found. Click 'ML Pipeline' button at top-right to run predictions.")
    else:
        # High risk: severity >= 0.4 (configurable)
        high_risk_df = df[df["severity_index"] >= 0.4]
        active_alerts = int((df["p_outbreak"] >= 0.5).sum())
        avg_risk = float(df["severity_index"].mean())
        # population at risk: sum population for high_risk (if available) else estimate
        if "population" in df.columns and not df["population"].isnull().all():
            pop_at_risk = int(df.loc[df["severity_index"] >= 0.4, "population"].sum())
        else:
            pop_at_risk = int(len(high_risk_df) * 1_000_000)

        col1.metric("High-Risk Districts", f"{len(high_risk_df)}/{len(df)}")
        col2.metric("Active Alerts", str(active_alerts))
        col3.metric("Average Risk Score", f"{avg_risk:.2f}")
        col4.metric("Population At Risk", f"{pop_at_risk:,}")

        # Add map interaction instructions
        st.markdown("""
        <div style='background-color: #EEF2FF; padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #3B82F6; margin-bottom: 1rem;'>
            <p style='margin: 0; color: #1E40AF; font-size: 14px;'>
                <strong>🖱️ Map Controls:</strong> 
                Use your <strong>touchpad scroll</strong> or <strong>mouse wheel</strong> to zoom in/out • 
                <strong>Click and drag</strong> to pan around • 
                <strong>Hover</strong> over markers for details • 
                <strong>Double-click</strong> to reset view
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🗺️ Interactive District-Level Outbreak Map")
        
        # Interactive map instructions with better styling (inspired by React/Leaflet version)
        st.markdown("""
        <div style='background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%); 
                    padding: 1.2rem; border-radius: 10px; margin-bottom: 1.5rem;
                    border-left: 4px solid #3B82F6; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
            <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                <span style='font-size: 20px;'>🗺️</span>
                <strong style='color: #1E40AF; font-size: 15px;'>Interactive Leaflet Map Controls</strong>
            </div>
            <div style='color: #4B5563; font-size: 14px; line-height: 1.6;'>
                <span style='margin-right: 1.5rem;'>🖱️ <strong>Scroll:</strong> Zoom in/out independently</span>
                <span style='margin-right: 1.5rem;'>👆 <strong>Drag:</strong> Pan across regions</span>
                <span style='margin-right: 1.5rem;'>🔍 <strong>Hover:</strong> View district details</span>
                <span>📍 <strong>Click marker:</strong> View detailed info panel</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # map data - ensure lat/lon exist
        map_df = df.dropna(subset=["lat", "lon"])
        if not map_df.empty:
            # Create Leaflet map with professional styling
            # Center on India
            m = folium.Map(
                location=[20.5937, 78.9629],
                zoom_start=5,
                tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
                attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
                prefer_canvas=True,
                zoom_control=True,
                scrollWheelZoom=True,
                dragging=True,
                control_scale=True
            )
            
            # Add scale control
            folium.plugins.MousePosition().add_to(m)
            
            # Color mapping for risk levels
            def get_color(severity):
                if severity >= 0.6:
                    return '#EF4444'  # Critical - Red
                elif severity >= 0.4:
                    return '#F97316'  # High - Orange
                elif severity >= 0.2:
                    return '#F59E0B'  # Moderate - Amber
                else:
                    return '#10B981'  # Low - Green
            
            def get_risk_label(severity):
                if severity >= 0.6:
                    return 'Critical'
                elif severity >= 0.4:
                    return 'High'
                elif severity >= 0.2:
                    return 'Moderate'
                else:
                    return 'Low'
            
            # Add progress indicator
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            # Process each city and add to map
            total_cities = len(map_df)
            for counter, (idx, row) in enumerate(map_df.iterrows()):
                progress_bar.progress((counter + 1) / total_cities)
                progress_text.text(f"Loading regions... {counter + 1}/{total_cities}")
                
                severity = row['severity_index']
                color = get_color(severity)
                risk_label = get_risk_label(severity)
                opacity = 0.4 + (severity * 0.6)  # Dynamic opacity based on severity
                
                # Create professional popup with styled HTML
                popup_html = f"""
                <div style='font-family: Arial, sans-serif; min-width: 280px; padding: 8px;'>
                    <div style='background: linear-gradient(135deg, {color}20, {color}10); 
                                padding: 12px; border-radius: 8px 8px 0 0; 
                                border-left: 4px solid {color}; margin: -8px -8px 8px -8px;'>
                        <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 6px;'>
                            <span style='font-size: 18px;'>📍</span>
                            <h3 style='margin: 0; color: #111827; font-size: 18px; font-weight: 700;'>
                                {row['region']}
                            </h3>
                        </div>
                        <div style='display: inline-block; background-color: {color}; color: white; 
                                    padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600;'>
                            {risk_label} Risk
                        </div>
                    </div>
                    
                    <div style='padding: 4px 0;'>
                        <div style='margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #E5E7EB;'>
                            <div style='color: #6B7280; font-size: 13px; margin-bottom: 4px;'>🦠 Fever Type</div>
                            <div style='color: #111827; font-size: 15px; font-weight: 600;'>{row['fever_type']}</div>
                        </div>
                        
                        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 10px;'>
                            <div>
                                <div style='color: #6B7280; font-size: 12px; margin-bottom: 2px;'>📊 Outbreak Risk</div>
                                <div style='color: #DC2626; font-size: 16px; font-weight: 700;'>{row['p_outbreak']:.0%}</div>
                            </div>
                            <div>
                                <div style='color: #6B7280; font-size: 12px; margin-bottom: 2px;'>⚠️ Severity</div>
                                <div style='color: {color}; font-size: 16px; font-weight: 700;'>{severity:.3f}</div>
                            </div>
                        </div>
                        
                        <div style='margin-top: 10px; padding-top: 10px; border-top: 1px solid #E5E7EB;'>
                            <div style='color: #6B7280; font-size: 12px; margin-bottom: 2px;'>👥 Population</div>
                            <div style='color: #111827; font-size: 14px; font-weight: 600;'>
                                {int(row.get('population', 0)):,}
                            </div>
                        </div>
                    </div>
                    
                    <div style='margin-top: 12px; padding: 8px; background-color: #F9FAFB; 
                                border-radius: 6px; font-size: 11px; color: #6B7280;'>
                        ℹ️ <strong>Shape</strong> = Circular area (population-based radius)<br>
                        💡 Color intensity shows severity level
                    </div>
                </div>
                """
                
                # Create tooltip (shown on hover)
                tooltip_html = f"""
                <div style='font-family: Arial, sans-serif; padding: 6px;'>
                    <div style='font-weight: 700; font-size: 14px; color: #111827; margin-bottom: 4px;'>
                        📍 {row['region']}
                    </div>
                    <div style='font-size: 12px; color: #6B7280;'>
                        <span style='color: {color}; font-weight: 600;'>{risk_label}</span> • 
                        {row['fever_type']} • 
                        Risk: {row['p_outbreak']:.0%}
                    </div>
                    <div style='font-size: 10px; color: #9CA3AF; margin-top: 2px; font-style: italic;'>
                        Click for detailed view
                    </div>
                </div>
                """
                
                # Use circular area representation for all cities (efficient and consistent)
                # Calculate radius based on population to represent actual city area
                pop = row.get('population', 0)
                if pop and pop > 0:
                    # Approximate circle radius in meters based on population
                    # Assume average urban density ~5000 people/km²
                    import math
                    area_km2 = pop / 5000  # Estimated area in km²
                    radius_km = math.sqrt(area_km2 / math.pi)  # Convert to radius
                    radius_m = radius_km * 1000  # Convert to meters
                    radius_m = max(2000, min(20000, radius_m))  # Clamp between 2-20km
                else:
                    # Default to severity-based radius
                    radius_m = 3000 + (severity * 7000)  # 3-10km range
                
                # Create a circular representation of the city area
                folium.Circle(
                    location=[row['lat'], row['lon']],
                    radius=radius_m,  # Radius in meters (actual ground distance)
                    popup=folium.Popup(popup_html, max_width=320),
                    tooltip=folium.Tooltip(tooltip_html, sticky=True),
                    color=color,  # Border color matches risk level
                    weight=3,
                    fill=True,
                    fillColor=color,
                    fillOpacity=opacity * 0.5,  # Transparent fill for better map visibility
                    opacity=0.8,
                    dashArray='5, 5'  # Dashed border for visual style
                ).add_to(m)
                
                # Add a center point marker for precise location
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=8,
                    color='white',
                    weight=2,
                    fill=True,
                    fillColor=color,
                    fillOpacity=1,
                    opacity=1
                ).add_to(m)
            
            # Clear progress indicators
            progress_bar.empty()
            progress_text.empty()
            
            # Add a custom legend
            legend_html = '''
            <div style="position: fixed; 
                        top: 10px; right: 10px; 
                        background-color: rgba(255, 255, 255, 0.95); 
                        backdrop-filter: blur(10px);
                        padding: 16px; 
                        border-radius: 12px; 
                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                        border: 2px solid #E5E7EB;
                        z-index: 9999;
                        font-family: Arial, sans-serif;
                        min-width: 180px;">
                <h4 style="margin: 0 0 12px 0; color: #111827; font-size: 14px; font-weight: 700; border-bottom: 2px solid #E5E7EB; padding-bottom: 8px;">
                    🎯 Risk Levels
                </h4>
                <div style="font-size: 13px;">
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <div style="width: 20px; height: 20px; background-color: #EF4444; border-radius: 50%; margin-right: 10px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"></div>
                        <span style="color: #374151; font-weight: 600;">Critical</span>
                    </div>
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <div style="width: 20px; height: 20px; background-color: #F97316; border-radius: 50%; margin-right: 10px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"></div>
                        <span style="color: #374151; font-weight: 600;">High</span>
                    </div>
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <div style="width: 20px; height: 20px; background-color: #F59E0B; border-radius: 50%; margin-right: 10px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"></div>
                        <span style="color: #374151; font-weight: 600;">Moderate</span>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 20px; height: 20px; background-color: #10B981; border-radius: 50%; margin-right: 10px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"></div>
                        <span style="color: #374151; font-weight: 600;">Low</span>
                    </div>
                </div>
                <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #E5E7EB;">
                    <div style="font-size: 11px; color: #6B7280; margin-bottom: 8px; font-weight: 600;">
                        �️ Map Features:
                    </div>
                    <div style="font-size: 10px; color: #9CA3AF; line-height: 1.5;">
                        • <strong>Shape</strong> = Actual city boundaries<br>
                        • <strong>Color</strong> = Risk level<br>
                        • <strong>Opacity</strong> = Severity intensity<br>
                        • Hover to preview, click for details
                    </div>
                </div>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
            # Display the map with custom height
            st_folium(m, width=None, height=650, returned_objects=[])
        else:
            st.info("No geocoded regions available. Use ML Pipeline upload that contains regions or populate region metadata.")

        st.markdown("### Priority Action Plan — High Risk Districts")
        if high_risk_df.empty:
            st.info("No districts flagged as high risk currently.")
        else:
            for idx, row in enumerate(high_risk_df.sort_values("severity_index", ascending=False).iterrows()):
                _, row = row
                region = row["region"]
                
                # Risk level indicator
                if row['severity_index'] >= 0.7:
                    risk_badge = "🔴 CRITICAL"
                    badge_color = "#EF4444"
                elif row['severity_index'] >= 0.5:
                    risk_badge = "🟠 HIGH"
                    badge_color = "#F97316"
                else:
                    risk_badge = "🟡 ELEVATED"
                    badge_color = "#F59E0B"
                
                with st.container():
                    st.markdown(f"""
                    <div style='background-color: #F9FAFB; padding: 1.5rem; border-radius: 8px; 
                                border-left: 4px solid {badge_color}; margin-bottom: 1rem;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <h4 style='margin: 0; color: #111827; font-size: 18px;'>{region}</h4>
                                <p style='margin: 0.25rem 0; color: #6B7280; font-size: 14px;'>
                                    <span style='background-color: {badge_color}; color: white; padding: 2px 8px; 
                                          border-radius: 4px; font-size: 12px; font-weight: 600;'>{risk_badge}</span>
                                    <span style='margin-left: 8px;'>{row['fever_type']} Outbreak</span>
                                </p>
                            </div>
                            <div style='text-align: right;'>
                                <div style='font-size: 32px; font-weight: 700; color: {badge_color};'>
                                    {row['severity_index']:.2f}
                                </div>
                                <div style='font-size: 12px; color: #9CA3AF;'>Severity Index</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.write(f"📊 **Outbreak Probability:** {row['p_outbreak']:.1%}")
                    with col2:
                        pop = row.get('population', 'N/A')
                        if pd.notna(pop):
                            st.write(f"👥 **Population:** {int(pop):,}")
                        else:
                            st.write(f"👥 **Population:** N/A")
                    with col3:
                        st.button("🚨 Deploy Response", key=f"deploy_{region}_{idx}", type="primary")
                    
                    st.markdown("""
                    **Recommended Actions:**
                    - 🏥 Deploy rapid response medical teams
                    - 🧪 Set up community testing centers
                    - 📢 Launch public awareness campaigns
                    - 🧹 Intensify sanitation and vector control
                    """)
                    st.markdown("---")

        st.markdown("### Regional Impact Summary")
        # quick state summary using state column if available
        if "state" in df.columns and not df["state"].isnull().all():
            summary = df.groupby("state").size().reset_index(name="districts")
            cols = st.columns(len(summary))
            for i, r in summary.iterrows():
                cols[i].metric(f"{r['state']}", f"{r['districts']} districts")
        else:
            st.info("No state metadata available — region metadata can be added via the ML pipeline.")

# ----------------------
# PHARMA VIEW
# ----------------------
with pharma_tab:
    # Professional header
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                padding: 2rem; border-radius: 12px; margin-bottom: 2rem;'>
        <h2 style='color: white; margin: 0; font-size: 28px;'>💊 Pharmaceutical Dashboard</h2>
        <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 16px;'>
            Medicine stock management & demand forecasting
        </p>
    </div>
    """, unsafe_allow_html=True)

    df = fetch_all_predictions()
    
    # Enhanced KPI cards with better styling
    c1, c2, c3, c4 = st.columns(4)
    total_units = int(len(df) * 10000) if not df.empty else 0
    top_fever = df["fever_type"].mode().iloc[0] if (not df.empty and df["fever_type"].notnull().any()) else "N/A"
    top_region = df.loc[df["p_outbreak"].idxmax()]["region"] if (not df.empty) else "N/A"
    supply_pct = 80

    with c1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
            <div style='font-size: 14px; opacity: 0.9; margin-bottom: 8px;'>Total Stock Units</div>
            <div style='font-size: 32px; font-weight: 700;'>{total_units:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
            <div style='font-size: 14px; opacity: 0.9; margin-bottom: 8px;'>Top Fever Type</div>
            <div style='font-size: 28px; font-weight: 700;'>{top_fever}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c3:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
            <div style='font-size: 14px; opacity: 0.9; margin-bottom: 8px;'>Highest Risk Region</div>
            <div style='font-size: 24px; font-weight: 700;'>{top_region}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c4:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
            <div style='font-size: 14px; opacity: 0.9; margin-bottom: 8px;'>Stock Fulfillment</div>
            <div style='font-size: 32px; font-weight: 700;'>{supply_pct}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    def safe_float(val, default=0.0):
        try:
            return float(val)
        except:
            return default
            
    st.markdown("### Stock Requirements Analysis")
    if df.empty:
        st.info("No data to compute stock requirements.")
    else:
        # create a professional bar chart
        df["pop_est"] = df["population"].fillna(1_000_000)
        df["demand_index"] = df["severity_index"] * df["pop_est"]
        total_demand = df["demand_index"].sum()
        
        # Enhanced drug data with icons
        drugs_data = [
            {"drug": "💊 Paracetamol", "required": int(total_demand * 0.5 / 100000), "icon": "💊", "color": "#3B82F6"},
            {"drug": "💧 ORS Packets", "required": int(total_demand * 0.3 / 100000), "icon": "💧", "color": "#10B981"},
            {"drug": "💉 Antibiotics", "required": int(total_demand * 0.15 / 100000), "icon": "💉", "color": "#F59E0B"},
            {"drug": "🩺 IV Fluids", "required": int(total_demand * 0.05 / 100000), "icon": "🩺", "color": "#EF4444"}
        ]
        
        bar_df = pd.DataFrame(drugs_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=bar_df['drug'],
            x=bar_df['required'],
            orientation='h',
            marker=dict(
                color=bar_df['color'],
                line=dict(color='rgba(0,0,0,0.1)', width=1)
            ),
            text=bar_df['required'],
            texttemplate='%{text:,} units',
            textposition='outside',
            textfont=dict(
                size=14,
                color='#111827',
                family='Arial, sans-serif'
            ),
            hovertemplate='<b>%{y}</b><br>Required: %{x:,} units<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text="Medicine Demand Forecast by Category",
                font=dict(size=18, color="#111827", family='Arial, sans-serif')
            ),
            xaxis=dict(
                title=dict(
                    text="Units Required",
                    font=dict(size=14, color="#374151")
                ),
                showgrid=True,
                gridcolor='rgba(0,0,0,0.05)',
                tickfont=dict(size=12, color="#111827")
            ),
            yaxis=dict(
                title="",
                showgrid=False,
                tickfont=dict(size=13, color="#111827", family='Arial, sans-serif')
            ),
            height=400,
            margin=dict(l=20, r=100, t=60, b=40),
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12, color="#111827")
        )
        
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Regional Stock Recommendations")
    if df.empty:
        st.info("No regional recommendations available.")
    else:
        # Enhanced pharma cards with professional styling
        sample = df.sort_values("severity_index", ascending=False).head(6)
        
        for idx, (_, r) in enumerate(sample.iterrows()):
            region = r["region"]
            
            population_raw = r["population"] if ("population" in r and pd.notna(r["population"])) else 1_000_000
            severity_raw = r["severity_index"] if ("severity_index" in r and pd.notna(r["severity_index"])) else 0

            population = safe_float(population_raw, 1_000_000)
            severity = safe_float(severity_raw, 0)

            base = int(population * severity * 0.001)
            
            paracetamol = max(50, base)
            ors = max(30, int(base * 0.8))
            antibiotics = max(20, int(base * 0.6))
            iv_fluids = max(10, int(base * 0.3))
            
            # Risk color coding
            if severity >= 0.6:
                card_color = "#FEE2E2"
                border_color = "#EF4444"
            elif severity >= 0.4:
                card_color = "#FED7AA"
                border_color = "#F97316"
            else:
                card_color = "#FEF3C7"
                border_color = "#F59E0B"
            
            st.markdown(f"""
            <div style='background-color: {card_color}; padding: 1.5rem; border-radius: 10px; 
                        border-left: 5px solid {border_color}; margin-bottom: 1.5rem;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;'>
                    <div>
                        <h4 style='margin: 0; color: #111827; font-size: 20px;'>{region}</h4>
                        <p style='margin: 0.25rem 0; color: #6B7280;'>
                            {r['fever_type']} • Severity: <strong>{severity:.2f}</strong>
                        </p>
                    </div>
                    <div style='background-color: white; padding: 0.5rem 1rem; border-radius: 6px;'>
                        <span style='color: #6B7280; font-size: 12px;'>Population</span><br>
                        <strong style='color: #111827; font-size: 18px;'>{int(population):,}</strong>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            cols = st.columns([1,1,1,1,1.5])
            
            with cols[0]:
                st.markdown(f"""
                <div style='background: white; padding: 1rem; border-radius: 8px; text-align: center; 
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <div style='font-size: 24px;'>💊</div>
                    <div style='color: #6B7280; font-size: 12px; margin: 4px 0;'>Paracetamol</div>
                    <div style='color: #111827; font-size: 20px; font-weight: 600;'>{paracetamol}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown(f"""
                <div style='background: white; padding: 1rem; border-radius: 8px; text-align: center; 
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <div style='font-size: 24px;'>💧</div>
                    <div style='color: #6B7280; font-size: 12px; margin: 4px 0;'>ORS</div>
                    <div style='color: #111827; font-size: 20px; font-weight: 600;'>{ors}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[2]:
                st.markdown(f"""
                <div style='background: white; padding: 1rem; border-radius: 8px; text-align: center; 
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <div style='font-size: 24px;'>💉</div>
                    <div style='color: #6B7280; font-size: 12px; margin: 4px 0;'>Antibiotics</div>
                    <div style='color: #111827; font-size: 20px; font-weight: 600;'>{antibiotics}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[3]:
                st.markdown(f"""
                <div style='background: white; padding: 1rem; border-radius: 8px; text-align: center; 
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <div style='font-size: 24px;'>🩺</div>
                    <div style='color: #6B7280; font-size: 12px; margin: 4px 0;'>IV Fluids</div>
                    <div style='color: #111827; font-size: 20px; font-weight: 600;'>{iv_fluids}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[4]:
                if st.button("📦 Dispatch Stock Order", key=f"dispatch_{region}_{idx}", type="primary", width="stretch"):
                    ts = datetime.now(timezone.utc).isoformat()
                    upsert_pharma_stock(region, paracetamol, ors, antibiotics, iv_fluids, ts)
                    st.success(f"✅ Dispatch order scheduled for {region}")
            
            st.markdown("<br>", unsafe_allow_html=True)

# ----------------------
# PUBLIC VIEW
# ----------------------
with public_tab:
    # Professional header
    st.markdown("""
    <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                padding: 2rem; border-radius: 12px; margin-bottom: 2rem;'>
        <h2 style='color: white; margin: 0; font-size: 28px;'>� Public Awareness Dashboard</h2>
        <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 16px;'>
            🛡️ Recognize. Respond. Reduce. Stay informed about health risks in your area.
        </p>
    </div>
    """, unsafe_allow_html=True)

    colA, colB = st.columns([3,1])
    with colA:
        st.markdown("### 🔍 Search Your Location")
        city = st.text_input("Enter your city or district name:", placeholder="e.g., Chennai, Mumbai, Delhi")
        if city:
            rec = fetch_city_prediction(city)
            if rec:
                severity = rec["severity_index"]
                
                # Risk level styling
                if severity >= 0.4:
                    risk_level = "HIGH RISK"
                    risk_color = "#EF4444"
                    risk_bg = "#FEE2E2"
                    risk_icon = "🔴"
                    advice = "High risk detected. Take immediate precautions and seek medical help if symptomatic."
                elif severity >= 0.2:
                    risk_level = "MODERATE RISK"
                    risk_color = "#F59E0B"
                    risk_bg = "#FEF3C7"
                    risk_icon = "🟡"
                    advice = "Moderate risk level. Be vigilant and follow preventive measures."
                else:
                    risk_level = "LOW RISK"
                    risk_color = "#10B981"
                    risk_bg = "#D1FAE5"
                    risk_icon = "🟢"
                    advice = "Low risk currently. Continue following regular hygiene practices."
                
                # Display results using Streamlit components
                st.markdown(f"<h2 style='text-align: center; color: #111827;'>{city}</h2>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: center; background-color: {risk_color}; color: white; "
                          f"padding: 8px 20px; border-radius: 20px; margin: 10px auto; font-weight: 600; "
                          f"display: inline-block; width: fit-content;'>{risk_icon} {risk_level}</div>", 
                          unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Metrics in columns
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Outbreak Probability", f"{rec['p_outbreak']:.1%}")
                with col2:
                    st.metric("Severity Index", f"{severity:.2f}")
                
                st.markdown("---")
                
                # Fever Type Info
                st.markdown("**Detected Fever Type**")
                st.info(f"🦠 {rec['fever_type']} (Confidence: {rec['p_type']:.1%})")
                
                # Health Advisory
                st.markdown("**Health Advisory**")
                if severity >= 0.4:
                    st.error(f"⚠️ {advice}")
                elif severity >= 0.2:
                    st.warning(f"⚠️ {advice}")
                else:
                    st.success(f"✅ {advice}")
            else:
                st.warning(f"⚠️ No data available for '{city}'. Please check spelling or try another location.")
                
    with colB:
        st.markdown("### 📍 Quick Access")
        
        # Show location info if a city was searched
        if city and 'rec' in locals():
            st.success(f"📍 Location detected: {city}")
            st.info("Check the search results on the left")
        else:
            st.info("Enter a city name on the left to check risk levels")
        
        # Show top high-risk locations
        st.markdown("---")
        st.markdown("**🔥 High Risk Areas**")
        all_predictions = fetch_all_predictions()
        if not all_predictions.empty:
            high_risk = all_predictions[all_predictions['severity_index'] >= 0.4].head(3)
            if not high_risk.empty:
                for _, row in high_risk.iterrows():
                    st.caption(f"🔴 {row['region']}")
            else:
                st.caption("No high-risk areas currently")
        else:
            st.caption("No data available")

    st.markdown("### Understanding Risk Levels")
    r1, r2, r3 = st.columns(3)
    
    with r1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #10B981 0%, #059669 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
            <div style='font-size: 32px; margin-bottom: 8px;'>🟢</div>
            <h4 style='margin: 0 0 8px 0;'>Low Risk</h4>
            <p style='margin: 0; font-size: 14px; opacity: 0.9;'>
                No immediate concern. Stay alert and follow basic hygiene practices.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with r2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
            <div style='font-size: 32px; margin-bottom: 8px;'>🟡</div>
            <h4 style='margin: 0 0 8px 0;'>Moderate Risk</h4>
            <p style='margin: 0; font-size: 14px; opacity: 0.9;'>
                Slight rise in cases. Take preventive measures and stay vigilant.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with r3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
            <div style='font-size: 32px; margin-bottom: 8px;'>🔴</div>
            <h4 style='margin: 0 0 8px 0;'>High Risk</h4>
            <p style='margin: 0; font-size: 14px; opacity: 0.9;'>
                Local outbreak warning. Exercise caution and seek medical help if needed.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🛡️ Stay Protected - Essential Guidelines")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background-color: white; padding: 1.5rem; border-radius: 10px; 
                    border: 1px solid #E5E7EB; margin-top: 1rem;'>
            <h4 style='color: #111827; margin-top: 0;'>💧 Prevention Measures</h4>
            <ul style='color: #374151; line-height: 1.8;'>
                <li>Drink clean and boiled water</li>
                <li>Avoid mosquito breeding around home</li>
                <li>Wash hands regularly with soap</li>
                <li>Maintain proper food hygiene</li>
                <li>Use mosquito repellents and nets</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background-color: white; padding: 1.5rem; border-radius: 10px; 
                    border: 1px solid #E5E7EB; margin-top: 1rem;'>
            <h4 style='color: #111827; margin-top: 0;'>🏥 When to Seek Help</h4>
            <ul style='color: #374151; line-height: 1.8;'>
                <li>Fever lasting more than 2 days</li>
                <li>Severe headache or body pain</li>
                <li>Persistent vomiting or diarrhea</li>
                <li>Difficulty breathing</li>
                <li>Signs of dehydration</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)



