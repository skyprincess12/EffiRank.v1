# app5.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import platform
import tempfile
from supabase import create_client

# --------------------
# Cross-platform data directory setup
# --------------------
if platform.system() == "Windows":
    DATA_DIR = os.path.join(tempfile.gettempdir(), "tls_app_data")
else:
    DATA_DIR = os.path.expanduser("~/.tls_app_data")

if not os.path.exists(DATA_DIR):
    try:
        os.makedirs(DATA_DIR)
    except Exception as e:
        st.error(f"Cannot create data directory: {e}")
        DATA_DIR = tempfile.gettempdir()

# File paths
LOCATIONS_FILE = os.path.join(DATA_DIR, "locations_data.json")
HISTORY_FILE = os.path.join(DATA_DIR, "history_snapshots.json")
SETTINGS_FILE = os.path.join(DATA_DIR, "app_settings.json")

# --------------------
# Config / Credentials with proper error handling
# --------------------
def load_secrets():
    """Load secrets with proper error handling"""
    try:
        USERNAME = st.secrets["USERNAME"]
        PASSWORD = st.secrets["PASSWORD"]
        HISTORY_DELETE_PASSCODE = st.secrets["HISTORY_DELETE_PASSCODE"]
        SUPABASE_URL = st.secrets.get("SUPABASE_URL")
        SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")
        return USERNAME, PASSWORD, HISTORY_DELETE_PASSCODE, SUPABASE_URL, SUPABASE_KEY
    except KeyError as e:
        st.error(f"Missing required secret: {e}")
        st.info("Please configure your secrets in .streamlit/secrets.toml")
        st.stop()
    except Exception as e:
        st.error(f"Error loading secrets: {e}")
        st.stop()

USERNAME, PASSWORD, HISTORY_DELETE_PASSCODE, SUPABASE_URL, SUPABASE_KEY = load_secrets()

# Initialize Supabase with connection testing
def init_supabase():
    """Initialize Supabase connection with proper error handling"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.warning("Supabase credentials not configured. Using local storage only.")
        return None
    
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        # Test connection with a simple query
        test_result = supabase_client.table("history_snapshots").select("count", count="exact").limit(1).execute()
        return supabase_client
    except Exception as e:
        st.warning(f"Supabase connection failed: {e}. Using local storage only.")
        return None

supabase = init_supabase()

# --------------------
# Page config & styling (fixed Unicode characters)
# --------------------
st.set_page_config(
    page_title="TLS Cost Input & Ranking System",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

css = '''
<style>
    .main-header { background: linear-gradient(90deg,#1e40af 0%,#7c3aed 100%); padding:2rem; border-radius:10px; color:white; margin-bottom:2rem; text-align:center;}
    .main-header h1 { color:white !important; margin:0; font-size:2.5rem;}
    .main-header p { color:rgba(255,255,255,0.9)!important; margin:0.5rem 0 0 0;}
    .metric-card{background:#f8fafc; padding:1rem; border-radius:8px; border-left:4px solid #3b82f6; margin-bottom:0.5rem;}
    .calculation-box{background:#fef3c7; padding:1rem; border-radius:8px; border:1px solid #f59e0b; margin:0.5rem 0;}
    .efficiency-excellent{background:#d1fae5; color:#059669; font-weight:bold; padding:0.5rem 1rem; border-radius:8px; border:2px solid #059669; text-align:center; margin:1rem 0;}
    .efficiency-good{background:#fef3c7; color:#d97706; font-weight:bold; padding:0.5rem 1rem; border-radius:8px; border:2px solid #d97706; text-align:center; margin:1rem 0;}
    .efficiency-average{background:#e0f2fe; color:#0369a1; font-weight:bold; padding:0.5rem 1rem; border-radius:8px; border:2px solid #0369a1; text-align:center; margin:1rem 0;}
    .efficiency-poor{background:#fee2e2; color:#dc2626; font-weight:bold; padding:0.5rem 1rem; border-radius:8px; border:2px solid #dc2626; text-align:center; margin:1rem 0;}
    .calculated-values { background: #f0f9ff; padding: 1rem; border-radius: 8px; border: 1px solid #0ea5e9; margin: 1rem 0; }
    .date-header{background:linear-gradient(90deg,#10b981 0%,#059669 100%); padding:1.5rem; border-radius:10px; color:white; margin-bottom:1rem; text-align:center;}
    .login-container{max-width:400px; margin:2rem auto; padding:2rem; background:#f8fafc; border-radius:10px; box-shadow:0 4px 6px rgba(0,0,0,0.1);}
    .error-boundary{background:#fee2e2; border:1px solid #dc2626; padding:1rem; border-radius:8px; margin:1rem 0;}
</style>
'''
st.markdown(css, unsafe_allow_html=True)

# --------------------
# Input validation functions
# --------------------
def validate_location_data(data):
    """Validate location data inputs"""
    errors = []
    
    if data.get('lkgtc', 0) < 0:
        errors.append("LKGTC cannot be negative")
    
    if data.get('diesel_price', 0) < 0:
        errors.append("Diesel price cannot be negative")
    
    if data.get('fuel_cons', 0) < 0:
        errors.append("Fuel consumption cannot be negative")
    
    # Check for reasonable ranges
    if data.get('diesel_price', 0) > 200:
        errors.append("Diesel price seems unusually high")
    
    if data.get('lkgtc', 0) > 1000:
        errors.append("LKGTC seems unusually high")
    
    return errors

def safe_divide(numerator, denominator, default=0):
    """Safe division with error handling"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError, ZeroDivisionError):
        return default

# --------------------
# Local persistence functions with error handling
# --------------------
def save_locations_data():
    """Save locations data to local JSON file with error handling"""
    try:
        with open(LOCATIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.locations_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error saving locations data: {e}")

def load_locations_data():
    """Load locations data from local JSON with error handling"""
    try:
        if os.path.exists(LOCATIONS_FILE):
            with open(LOCATIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading locations data: {e}")

    # Default locations data
    return {
        'DIRECT MILLSITE': {'region': 'NORTH', 'barangay_fee': 0.0, 'rental_rate': 0.0, 'tls_opn': 0.0, 'drivers_hauler': 0.0, 'fuel_cons': 0.0, 'diesel_price': 0.0, 'ta_inc': 0.0, 'lkgtc': 0.0},
        'CROSSING VITO': {'region': 'NORTH', 'barangay_fee': 0.0, 'rental_rate': 0.0, 'tls_opn': 0.0, 'drivers_hauler': 0.0, 'fuel_cons': 0.0, 'diesel_price': 0.0, 'ta_inc': 0.0, 'lkgtc': 0.0},
        'BATO': {'region': 'NORTH', 'barangay_fee': 0.0, 'rental_rate': 0.0, 'tls_opn': 0.0, 'drivers_hauler': 0.0, 'fuel_cons': 0.0, 'diesel_price': 0.0, 'ta_inc': 0.0, 'lkgtc': 0.0},
        'ESCALANTE': {'region': 'NORTH', 'barangay_fee': 0.0, 'rental_rate': 0.0, 'tls_opn': 0.0, 'drivers_hauler': 0.0, 'fuel_cons': 0.0, 'diesel_price': 0.0, 'ta_inc': 0.0, 'lkgtc': 0.0},
        'SAN JOSE': {'region': 'NORTH', 'barangay_fee': 0.0, 'rental_rate': 0.0, 'tls_opn': 0.0, 'drivers_hauler': 0.0, 'fuel_cons': 0.0, 'diesel_price': 0.0, 'ta_inc': 0.0, 'lkgtc': 0.0},
        'PALAU': {'region': 'NORTH', 'barangay_fee': 0.0, 'rental_rate': 0.0, 'tls_opn': 0.0, 'drivers_hauler': 0.0, 'fuel_cons': 0.0, 'diesel_price': 0.0, 'ta_inc': 0.0, 'lkgtc': 0.0},
        'BAGAWINES': {'region': 'NORTH', 'barangay_fee': 0.0, 'rental_rate': 0.0, 'tls_opn': 0.0, 'drivers_hauler': 0.0, 'fuel_cons': 0.0, 'diesel_price': 0.0, 'ta_inc': 0.0, 'lkgtc': 0.0},
        'CANIBUNGAN': {'region': 'SOUTH', 'barangay_fee': 0.0, 'rental_rate': 0.0, 'tls_opn': 0.0, 'drivers_hauler': 0.0, 'fuel_cons': 0.0, 'diesel_price': 0.0, 'ta_inc': 0.0, 'lkgtc': 0.0},
        'MANAPLA': {'region': 'SOUTH', 'barangay_fee': 0.0, 'rental_rate': 0.0, 'tls_opn': 0.0, 'drivers_hauler': 0.0, 'fuel_cons': 0.0, 'diesel_price': 0.0, 'ta_inc': 0.0, 'lkgtc': 0.0},
        'SAN ISIDRO': {'region': 'SOUTH', 'barangay_fee': 0.0, 'rental_rate': 0.0, 'tls_opn': 0.0, 'drivers_hauler': 0.0, 'fuel_cons': 0.0, 'diesel_price': 0.0, 'ta_inc': 0.0, 'lkgtc': 0.0},
        'SARAVIA': {'region': 'SOUTH', 'barangay_fee': 0.0, 'rental_rate': 0.0, 'tls_opn': 0.0, 'drivers_hauler': 0.0, 'fuel_cons': 0.0, 'diesel_price': 0.0, 'ta_inc': 0.0, 'lkgtc': 0.0},
        'MURCIA': {'region': 'SOUTH', 'barangay_fee': 0.0, 'rental_rate': 0.0, 'tls_opn': 0.0, 'drivers_hauler': 0.0, 'fuel_cons': 0.0, 'diesel_price': 0.0, 'ta_inc': 0.0, 'lkgtc': 0.0},
        'MA-AO': {'region': 'SOUTH', 'barangay_fee': 0.0, 'rental_rate': 0.0, 'tls_opn': 0.0, 'drivers_hauler': 0.0, 'fuel_cons': 0.0, 'diesel_price': 0.0, 'ta_inc': 0.0, 'lkgtc': 0.0},
        'LA CASTELLANA': {'region': 'SOUTH', 'barangay_fee': 0.0, 'rental_rate': 0.0, 'tls_opn': 0.0, 'drivers_hauler': 0.0, 'fuel_cons': 0.0, 'diesel_price': 0.0, 'ta_inc': 0.0, 'lkgtc': 0.0}
    }

def save_app_settings():
    """Save app settings with error handling"""
    try:
        settings = {
            'current_date': st.session_state.current_date.isoformat(),
            'current_week': st.session_state.current_week
        }
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        st.error(f"Error saving settings: {e}")

def load_app_settings():
    """Load app settings with error handling"""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            return {
                'current_date': datetime.fromisoformat(settings['current_date']).date(),
                'current_week': settings['current_week']
            }
    except Exception as e:
        st.error(f"Error loading settings: {e}")
    return None

# --------------------
# HISTORY: Supabase-backed with improved error handling
# --------------------
def save_history_snapshots_local():
    """Write history snapshots to local JSON with error handling"""
    try:
        serializable_snapshots = []
        for snap in st.session_state.history_snapshots:
            serializable_snap = snap.copy()
            if 'rankings_df' in serializable_snap and isinstance(serializable_snap['rankings_df'], pd.DataFrame):
                serializable_snap['rankings_df'] = serializable_snap['rankings_df'].to_dict('records')
            if 'analysis_df' in serializable_snap and isinstance(serializable_snap['analysis_df'], pd.DataFrame):
                serializable_snap['analysis_df'] = serializable_snap['analysis_df'].to_dict('records')
            serializable_snapshots.append(serializable_snap)
        
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(serializable_snapshots, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error saving history locally: {e}")

def load_history_snapshots():
    """Load history snapshots with improved error handling"""
    try:
        if supabase:
            try:
                res = supabase.table("history_snapshots").select("*").order("id", desc=True).execute()
                rows = res.data or []
                snaps = []
                for r in rows:
                    try:
                        snap = {
                            'timestamp': r.get('timestamp'),
                            'date': r.get('date'),
                            'week_number': r.get('week_number'),
                            'week_range': r.get('week_range'),
                            'rankings_df': pd.DataFrame(r.get('rankings_json') or []),
                            'analysis_df': pd.DataFrame(r.get('analysis_json') or [])
                        }
                        snaps.append(snap)
                    except Exception as e:
                        st.warning(f"Error processing history record: {e}")
                        continue
                return snaps
            except Exception as e:
                st.warning(f"Error loading from Supabase: {e}. Falling back to local storage.")
        
        # Fallback to local storage
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                snapshots = json.load(f)
            # Convert lists back to DataFrames
            for snap in snapshots:
                try:
                    if 'rankings_df' in snap and isinstance(snap['rankings_df'], list):
                        snap['rankings_df'] = pd.DataFrame(snap['rankings_df'])
                    if 'analysis_df' in snap and isinstance(snap['analysis_df'], list):
                        snap['analysis_df'] = pd.DataFrame(snap['analysis_df'])
                except Exception as e:
                    st.warning(f"Error processing snapshot: {e}")
                    continue
            return snapshots
    except Exception as e:
        st.error(f"Error loading history: {e}")
    return []

def save_snapshot(rankings_df, analysis_df):
    """Save snapshot with comprehensive error handling"""
    try:
        week_start, week_end = get_week_range(st.session_state.current_date)
        snap = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'date': st.session_state.current_date.strftime('%Y-%m-%d'),
            'week_number': st.session_state.current_week,
            'week_range': format_week_display(week_start, week_end),
            'rankings_df': rankings_df.copy(),
            'analysis_df': analysis_df.copy()
        }

        # Append to session history
        if 'history_snapshots' not in st.session_state:
            st.session_state.history_snapshots = []
        st.session_state.history_snapshots.insert(0, snap)

        # Insert to Supabase if available
        if supabase:
            try:
                insert_obj = {
                    "timestamp": snap['timestamp'],
                    "date": snap['date'],
                    "week_number": snap['week_number'],
                    "week_range": snap['week_range'],
                    "rankings_json": rankings_df.to_dict('records'),
                    "analysis_json": analysis_df.to_dict('records')
                }
                supabase.table("history_snapshots").insert(insert_obj).execute()
                st.success("Snapshot saved to database.")
            except Exception as e:
                st.warning(f"Error saving to database: {e}. Saved locally only.")

        # Always save local backup
        save_history_snapshots_local()
        
    except Exception as e:
        st.error(f"Error saving snapshot: {e}")

# --------------------
# Session initialization with error handling
# --------------------
def initialize_session():
    """Initialize session state with error handling"""
    try:
        if 'locations_data' not in st.session_state:
            st.session_state.locations_data = load_locations_data()

        if 'history_snapshots' not in st.session_state:
            st.session_state.history_snapshots = load_history_snapshots()

        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False

        # Load app settings or set defaults
        saved_settings = load_app_settings()
        if saved_settings:
            if 'current_date' not in st.session_state:
                st.session_state.current_date = saved_settings['current_date']
            if 'current_week' not in st.session_state:
                st.session_state.current_week = saved_settings['current_week']
        else:
            if 'current_date' not in st.session_state:
                st.session_state.current_date = datetime.now().date()
            if 'current_week' not in st.session_state:
                current_date = datetime.now()
                if current_date.month >= 10:
                    crop_year_start = datetime(current_date.year, 10, 1)
                else:
                    crop_year_start = datetime(current_date.year - 1, 10, 1)
                days_diff = (current_date - crop_year_start).days
                st.session_state.current_week = max(1, (days_diff // 7) + 1)
                
    except Exception as e:
        st.error(f"Error initializing session: {e}")

# Initialize session
initialize_session()

# --------------------
# Date / Week helper functions with error handling
# --------------------
def get_week_number(date):
    """Calculate week number with error handling"""
    try:
        if date.month >= 10:
            crop_year_start = datetime(date.year, 10, 1).date()
        else:
            crop_year_start = datetime(date.year - 1, 10, 1).date()
        days_diff = (date - crop_year_start).days
        week_num = (days_diff // 7) + 1
        return max(1, week_num)
    except Exception as e:
        st.error(f"Error calculating week number: {e}")
        return 1

def get_week_range(date):
    """Get week range with error handling"""
    try:
        week_num = get_week_number(date)
        if date.month >= 10:
            crop_year_start = datetime(date.year, 10, 1).date()
        else:
            crop_year_start = datetime(date.year - 1, 10, 1).date()
        week_start = crop_year_start + timedelta(days=(week_num - 1) * 7)
        week_end = week_start + timedelta(days=6)
        return week_start, week_end
    except Exception as e:
        st.error(f"Error calculating week range: {e}")
        return date, date

def format_date_display(date):
    """Format date for display"""
    try:
        return date.strftime("%B %d, %Y (%A)")
    except Exception as e:
        return str(date)

def format_week_display(week_start, week_end):
    """Format week range for display"""
    try:
        return f"{week_start.strftime('%b %d')} - {week_end.strftime('%b %d, %Y')}"
    except Exception as e:
        return f"{week_start} - {week_end}"

def date_week_selector():
    """Date and week selector with error handling"""
    try:
        current_week_start, current_week_end = get_week_range(st.session_state.current_date)
        current_week_num = get_week_number(st.session_state.current_date)
        
        st.markdown(f'''
        <div class="date-header">
            <h3 style="color: white; margin: 0;">📅 Current Period Information</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0;">
                <strong>Date:</strong> {format_date_display(st.session_state.current_date)}<br>
                <strong>Week #{current_week_num}</strong> | {format_week_display(current_week_start, current_week_end)}
            </p>
        </div>
        ''', unsafe_allow_html=True)

        with st.expander("📅 Change Date & Week", expanded=False):
            st.markdown("### 🗓️ Date & Week Settings")
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_date = st.date_input("Select Date", value=st.session_state.current_date, key="date_selector")
            with col2:
                if st.button("Update Date & Week", type="primary"):
                    st.session_state.current_date = selected_date
                    st.session_state.current_week = get_week_number(selected_date)
                    save_app_settings()
                    st.success(f"Updated to Week #{st.session_state.current_week}")
                    st.rerun()
                    
            if selected_date != st.session_state.current_date:
                preview_week_start, preview_week_end = get_week_range(selected_date)
                preview_week_num = get_week_number(selected_date)
                st.info(f"Preview: Week #{preview_week_num} | {format_week_display(preview_week_start, preview_week_end)}")
                
    except Exception as e:
        st.error(f"Error in date selector: {e}")

# --------------------
# KPI & Ranking functions with improved error handling
# --------------------
def calculate_metrics(location_data):
    """Calculate metrics with error handling and validation"""
    try:
        # Validate input data
        errors = validate_location_data(location_data)
        if errors:
            st.warning(f"Data validation warnings: {', '.join(errors)}")
        
        fuel_cost = safe_divide(location_data['fuel_cons'] * location_data['diesel_price'], 32)
        total_cost = location_data['tls_opn'] + location_data['drivers_hauler'] + fuel_cost + location_data['ta_inc']
        cost_per_lkg = safe_divide(total_cost, location_data['lkgtc'])
        lkg_per_php = safe_divide(location_data['lkgtc'], cost_per_lkg) if cost_per_lkg > 0 else 0
        
        return {
            'fuel_cost': fuel_cost,
            'total_cost': total_cost,
            'cost_per_lkg': cost_per_lkg,
            'lkg_per_php': lkg_per_php
        }
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        return {'fuel_cost': 0, 'total_cost': 0, 'cost_per_lkg': 0, 'lkg_per_php': 0}

def get_single_efficiency_class(kpi_score, all_scores):
    """Get efficiency class with error handling"""
    try:
        if not all_scores or len(all_scores) < 2:
            return ('efficiency-average', '📊 Average', 'Not enough data for comparison')
        
        q75, q50, q25 = np.percentile(all_scores, [75, 50, 25])
        
        if kpi_score >= q75:
            return ('efficiency-excellent', '🥇 Excellent', f'Top 25% performer (Score: {kpi_score:.3f})')
        elif kpi_score >= q50:
            return ('efficiency-good', '🥈 Good', f'Above average performer (Score: {kpi_score:.3f})')
        elif kpi_score >= q25:
            return ('efficiency-average', '🥉 Average', f'Average performer (Score: {kpi_score:.3f})')
        else:
            return ('efficiency-poor', '❌ Poor', f'Below average performer (Score: {kpi_score:.3f})')
    except Exception as e:
        st.error(f"Error calculating efficiency class: {e}")
        return ('efficiency-average', '📊 Average', 'Error in calculation')

def get_efficiency_classes(scores):
    """Get efficiency classes for multiple scores"""
    try:
        if not scores:
            return []
        
        q75, q50, q25 = np.percentile(scores, [75, 50, 25])
        classes = []
        
        for s in scores:
            if s >= q75:
                classes.append(('efficiency-excellent', '🥇 Excellent'))
            elif s >= q50:
                classes.append(('efficiency-good', '🥈 Good'))
            elif s >= q25:
                classes.append(('efficiency-average', '🥉 Average'))
            else:
                classes.append(('efficiency-poor', '❌ Poor'))
                
        return classes
    except Exception as e:
        st.error(f"Error calculating efficiency classes: {e}")
        return []

def normalize_cost_scores(costs):
    """Normalize cost scores with error handling"""
    try:
        if not costs:
            return []
        
        mx, mn = max(costs), min(costs)
        if mx == mn:
            return [0.5] * len(costs)
        
        return [(mx - c) / (mx - mn) for c in costs]
    except Exception as e:
        st.error(f"Error normalizing cost scores: {e}")
        return []

def normalize_lkg_scores(lkgs):
    """Normalize LKG scores with error handling"""
    try:
        if not lkgs:
            return []
        
        mx, mn = max(lkgs), min(lkgs)
        if mx == mn:
            return [0.5] * len(lkgs)
        
        return [(l - mn) / (mx - mn) for l in lkgs]
    except Exception as e:
        st.error(f"Error normalizing LKG scores: {e}")
        return []

def compute_kpis(cost_scores, lkg_scores):
    """Compute KPI scores with error handling"""
    try:
        if len(cost_scores) != len(lkg_scores):
            st.error("Cost and LKG score arrays have different lengths")
            return []
        
        return [0.5 * cs + 0.5 * ls for cs, ls in zip(cost_scores, lkg_scores)]
    except Exception as e:
        st.error(f"Error computing KPIs: {e}")
        return []

def calculate_all_kpis():
    """Calculate all KPIs with error handling"""
    try:
        data = []
        for loc, vals in st.session_state.locations_data.items():
            if vals['lkgtc'] > 0:
                metrics = calculate_metrics(vals)
                data.append({
                    'location': loc,
                    'total_cost': metrics['total_cost'],
                    'lkgtc': vals['lkgtc']
                })
        
        if len(data) < 2:
            return {}
        
        costs = [d['total_cost'] for d in data]
        lkgs = [d['lkgtc'] for d in data]
        cost_scores = normalize_cost_scores(costs)
        lkg_scores = normalize_lkg_scores(lkgs)
        kpis = compute_kpis(cost_scores, lkg_scores)
        
        result = {}
        for i, d in enumerate(data):
            if i < len(kpis):
                result[d['location']] = kpis[i]
        
        return result
    except Exception as e:
        st.error(f"Error calculating all KPIs: {e}")
        return {}

def rank_tls(data):
    """Rank TLS locations with comprehensive error handling"""
    try:
        if not data:
            return pd.DataFrame()
        
        costs = [d.get('Total Cost', 0) for d in data]
        lkgs = [d.get('LKGTC', 0) for d in data]
        cost_scores = normalize_cost_scores(costs)
        lkg_scores = normalize_lkg_scores(lkgs)
        kpis = compute_kpis(cost_scores, lkg_scores)
        
        for i, kpi in enumerate(kpis):
            if i < len(data):
                data[i]['KPI Score'] = kpi
        
        df = pd.DataFrame(data)
        
        if 'KPI Score' in df.columns and not df['KPI Score'].empty:
            df['Overall Rank'] = df['KPI Score'].rank(method='dense', ascending=False).astype(int)
            df['Regional Rank'] = df.groupby('Region')['KPI Score'].rank(method='dense', ascending=False).astype(int)
            df['Global Class'] = get_efficiency_classes(df['KPI Score'].tolist())
            
            # Regional classes
            regional_classes = []
            for region, group in df.groupby('Region'):
                region_classes = get_efficiency_classes(group['KPI Score'].tolist())
                regional_classes.extend(region_classes)
            df['Regional Class'] = regional_classes
        
        return df
    except Exception as e:
        st.error(f"Error ranking TLS locations: {e}")
        return pd.DataFrame()

# --------------------
# Authentication with error handling
# --------------------
def login_page():
    """Login page with improved styling"""
    try:
        st.markdown('''
        <div style="background: #60a5fa; padding: 2rem; border-radius: 10px; text-align: center; margin-bottom: 2rem;">
            <h1 style="color: white; margin: 0; font-size: 2.5rem;">🚛 TLS Cost Input & Ranking System</h1>
        </div>
        ''', unsafe_allow_html=True)
        
        st.header('🔐 Login')
        
        with st.form('login_form'):
            username = st.text_input('Username')
            password = st.text_input('Password', type='password')
            submitted = st.form_submit_button('Login', type='primary', use_container_width=True)
            
            if submitted:
                if username == USERNAME and password == PASSWORD:
                    st.session_state.authenticated = True
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error('Invalid username or password')
    except Exception as e:
        st.error(f"Error in login page: {e}")

def account_page():
    """Account page with logout functionality"""
    try:
        st.header('👤 Account')
        st.write(f'Logged in as **{USERNAME}**')
        
        if st.button('Logout'):
            st.session_state.authenticated = False
            st.success("Logged out successfully!")
            st.rerun()
    except Exception as e:
        st.error(f"Error in account page: {e}")

# --------------------
# Application pages with comprehensive error handling
# --------------------
def cost_input_page():
    """Cost input page with error boundaries"""
    try:
        st.markdown('''
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: #1e40af; margin: 0; font-size: 2.5rem;">🚛 TLS Cost Input & Ranking System</h1>
        </div>
        ''', unsafe_allow_html=True)
        
        date_week_selector()
        
        st.header('📝 Cost Input')
        st.write('Enter the cost breakdown for each unloading point. LKGTC values are required for efficiency calculation.')

        all_kpis = calculate_all_kpis()
        all_kpi_scores = list(all_kpis.values()) if all_kpis else []

        # Add new location section
        with st.expander('➕ Add New Location'):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                new_location_name = st.text_input('Location Name').upper().strip()
            with col2:
                new_region = st.selectbox('Region', ['NORTH', 'SOUTH'])
            with col3:
                if st.button('Add Location'):
                    if new_location_name and new_location_name not in st.session_state.locations_data:
                        st.session_state.locations_data[new_location_name] = {
                            'region': new_region,
                            'barangay_fee': 0.0, 'rental_rate': 0.0, 'tls_opn': 0.0,
                            'drivers_hauler': 0.0, 'fuel_cons': 0.0, 'diesel_price': 0.0,
                            'ta_inc': 0.0, 'lkgtc': 0.0
                        }
                        save_locations_data()
                        st.success(f'Added {new_location_name}')
                        st.rerun()
                    elif new_location_name in st.session_state.locations_data:
                        st.warning('Location already exists')
                    else:
                        st.warning('Please enter a location name')

        # Location input forms
        for location_name, data in st.session_state.locations_data.items():
            with st.expander(f'📍 {location_name} ({data["region"]})', expanded=False):
                try:
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        st.session_state.locations_data[location_name]['barangay_fee'] = st.number_input(
                            'Barangay Fee', value=float(data.get('barangay_fee', 0)), step=1.0, 
                            key=f'barangay_{location_name}', on_change=save_locations_data
                        )
                        st.session_state.locations_data[location_name]['rental_rate'] = st.number_input(
                            'Rental Rate', value=float(data.get('rental_rate', 0)), step=1.0, 
                            key=f'rental_{location_name}', on_change=save_locations_data
                        )
                        st.session_state.locations_data[location_name]['tls_opn'] = st.number_input(
                            'TLS Opn', value=float(data.get('tls_opn', 0)), step=1.0, 
                            key=f'tls_{location_name}', on_change=save_locations_data
                        )
                    
                    with col2:
                        st.session_state.locations_data[location_name]['drivers_hauler'] = st.number_input(
                            'Drivers/Hauler', value=float(data.get('drivers_hauler', 0)), step=1.0, 
                            key=f'drivers_{location_name}', on_change=save_locations_data
                        )
                        st.session_state.locations_data[location_name]['fuel_cons'] = st.number_input(
                            'Fuel Cons', value=float(data.get('fuel_cons', 0)), step=1.0, 
                            key=f'fuel_{location_name}', on_change=save_locations_data
                        )
                        st.session_state.locations_data[location_name]['diesel_price'] = st.number_input(
                            'Diesel Price', value=float(data.get('diesel_price', 0)), step=1.0, 
                            key=f'diesel_{location_name}', on_change=save_locations_data
                        )
                    
                    with col3:
                        st.session_state.locations_data[location_name]['ta_inc'] = st.number_input(
                            'T.A. & Inc', value=float(data.get('ta_inc', 0)), step=1.0, 
                            key=f'ta_{location_name}', on_change=save_locations_data
                        )
                        st.session_state.locations_data[location_name]['lkgtc'] = st.number_input(
                            'LKGTC (Today)', value=float(data.get('lkgtc', 0)), step=0.001, format="%.3f", 
                            key=f'lkgtc_{location_name}', on_change=save_locations_data
                        )

                        # Calculate and display metrics
                        current_data = st.session_state.locations_data[location_name]
                        metrics = calculate_metrics(current_data)

                        st.markdown('📊 **Calculated Values:**')
                        st.write(f"Fuel Cost: ₱{metrics['fuel_cost']:.2f}")
                        st.write(f"Total Cost: ₱{metrics['total_cost']:.2f}")
                        st.write(f"Cost per LKG: ₱{metrics['cost_per_lkg']:.4f}")
                        st.write(f"LKG per PHP: {metrics['lkg_per_php']:.5f}")

                        # Display efficiency rating
                        if current_data['lkgtc'] > 0 and location_name in all_kpis:
                            kpi_score = all_kpis[location_name]
                            css_class, rating, description = get_single_efficiency_class(kpi_score, all_kpi_scores)
                            st.markdown(f'''
                            <div class="{css_class}">
                                <strong>{rating}</strong><br>
                                <small>{description}</small>
                            </div>
                            ''', unsafe_allow_html=True)
                        elif current_data['lkgtc'] > 0:
                            st.markdown('''
                            <div class="efficiency-average">
                                <strong>📊 Pending</strong><br>
                                <small>Need more locations for comparison</small>
                            </div>
                            ''', unsafe_allow_html=True)
                        else:
                            st.markdown('''
                            <div class="efficiency-poor">
                                <strong>⚠️ Incomplete</strong><br>
                                <small>Please enter LKGTC value</small>
                            </div>
                            ''', unsafe_allow_html=True)

                    # Delete location button
                    if st.button(f"🗑️ Delete {location_name}", key=f"del_{location_name}"):
                        del st.session_state.locations_data[location_name]
                        save_locations_data()
                        st.success(f"Deleted {location_name}")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error processing location {location_name}: {e}")
                    
    except Exception as e:
        st.markdown(f'''
        <div class="error-boundary">
            <h3>⚠️ Error in Cost Input Page</h3>
            <p>An error occurred: {e}</p>
            <p>Please refresh the page and try again.</p>
        </div>
        ''', unsafe_allow_html=True)

def ranking_page():
    """Ranking page with error boundaries"""
    try:
        date_week_selector()
        
        st.header('🏆 Efficiency Ranking')
        
        # Prepare data
        data = []
        for loc, vals in st.session_state.locations_data.items():
            try:
                metrics = calculate_metrics(vals)
                data.append({
                    'Date': st.session_state.current_date.strftime('%Y-%m-%d'),
                    'Location': loc,
                    'Region': vals['region'],
                    'Barangay Fee': vals['barangay_fee'],
                    'Rental Rate': vals['rental_rate'],
                    'TLS Opn': vals['tls_opn'],
                    'Drivers/Hauler': vals['drivers_hauler'],
                    'Fuel Cons': vals['fuel_cons'],
                    'Diesel Price': vals['diesel_price'],
                    'T.A.&Inc': vals['ta_inc'],
                    'LKGTC': vals['lkgtc'],
                    'Fuel Cost': metrics['fuel_cost'],
                    'Total Cost': metrics['total_cost'],
                    'Cost per LKG': metrics['cost_per_lkg'],
                    'LKG per PHP': metrics['lkg_per_php']
                })
            except Exception as e:
                st.warning(f"Error processing {loc}: {e}")
                continue

        if not data:
            st.warning('No valid data available for ranking.')
            return

        # Generate rankings
        rankings_df = rank_tls(data)
        
        if rankings_df.empty:
            st.warning('Unable to generate rankings.')
            return

        # Round LKG per PHP for display
        if 'LKG per PHP' in rankings_df.columns:
            rankings_df['LKG per PHP'] = rankings_df['LKG per PHP'].round(5)

        st.dataframe(rankings_df, use_container_width=True)

        # Save snapshot button
        if st.button('💾 Save Snapshot'):
            try:
                analysis_df = rankings_df[['Date', 'Location', 'Region', 'Total Cost', 'LKGTC', 'KPI Score']].copy()
                save_snapshot(rankings_df, analysis_df)
                st.success('Snapshot saved successfully!')
            except Exception as e:
                st.error(f"Error saving snapshot: {e}")
                
    except Exception as e:
        st.markdown(f'''
        <div class="error-boundary">
            <h3>⚠️ Error in Ranking Page</h3>
            <p>An error occurred: {e}</p>
            <p>Please refresh the page and try again.</p>
        </div>
        ''', unsafe_allow_html=True)

def analysis_page():
    """Analysis page with error boundaries"""
    try:
        st.header('📊 Cost Analysis')
        
        # Prepare data for analysis
        data = []
        for loc, vals in st.session_state.locations_data.items():
            try:
                metrics = calculate_metrics(vals)
                data.append({
                    'Location': loc,
                    'Region': vals['region'],
                    'Total Cost': metrics['total_cost'],
                    'LKGTC': vals['lkgtc'],
                    'Cost per LKG': metrics['cost_per_lkg'],
                    'LKG per PHP': round(metrics['lkg_per_php'], 5)
                })
            except Exception as e:
                st.warning(f"Error processing {loc}: {e}")
                continue

        if not data:
            st.warning('No data available for analysis.')
            return

        df = pd.DataFrame(data)

        # Filter out locations with zero LKGTC for meaningful analysis
        df_filtered = df[df['LKGTC'] > 0]
        
        if df_filtered.empty:
            st.warning('No locations with LKGTC data for analysis.')
            return

        # Total Cost vs LKGTC scatter plot
        st.subheader('Total Cost vs LKGTC')
        try:
            fig = px.scatter(
                df_filtered, 
                x='Total Cost', 
                y='LKGTC', 
                color='Region', 
                text='Location',
                title='Cost Efficiency Analysis'
            )
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating scatter plot: {e}")

        # Cost per LKG by Region bar chart
        st.subheader('Cost per LKG by Region')
        try:
            fig2 = px.bar(
                df_filtered, 
                x='Location', 
                y='Cost per LKG', 
                color='Region',
                title='Cost per LKG Comparison'
            )
            fig2.update_xaxes(tickangle=45)
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating bar chart: {e}")

        # Summary statistics
        st.subheader('📈 Summary Statistics')
        try:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Total Cost", f"₱{df_filtered['Total Cost'].mean():.2f}")
                st.metric("Average LKGTC", f"{df_filtered['LKGTC'].mean():.3f}")
            with col2:
                st.metric("Average Cost per LKG", f"₱{df_filtered['Cost per LKG'].mean():.4f}")
                st.metric("Average LKG per PHP", f"{df_filtered['LKG per PHP'].mean():.5f}")
        except Exception as e:
            st.error(f"Error calculating summary statistics: {e}")
            
    except Exception as e:
        st.markdown(f'''
        <div class="error-boundary">
            <h3>⚠️ Error in Analysis Page</h3>
            <p>An error occurred: {e}</p>
            <p>Please refresh the page and try again.</p>
        </div>
        ''', unsafe_allow_html=True)

def history_page():
    """History page with error boundaries"""
    try:
        st.header('📜 History')
        
        if not st.session_state.history_snapshots:
            st.info('No history snapshots saved yet. Go to the Ranking page and click "Save Snapshot" to create your first snapshot.')
            return

        st.write(f"**Total Snapshots:** {len(st.session_state.history_snapshots)}")

        # Display each snapshot
        for i, snap in enumerate(st.session_state.history_snapshots):
            try:
                with st.expander(f"📸 Snapshot {i+1} — {snap.get('timestamp', 'Unknown time')}", expanded=False):
                    if 'date' in snap and 'week_number' in snap:
                        st.write(f"**Date:** {snap['date']} | **Week:** #{snap['week_number']} | **Range:** {snap.get('week_range', 'N/A')}")
                    
                    # Display rankings dataframe
                    display_df = snap['rankings_df'].copy() if isinstance(snap.get('rankings_df'), pd.DataFrame) else pd.DataFrame(snap.get('rankings_df', []))
                    
                    if not display_df.empty:
                        if 'LKG per PHP' in display_df.columns:
                            display_df['LKG per PHP'] = display_df['LKG per PHP'].round(5)
                        st.dataframe(display_df, use_container_width=True)
                    else:
                        st.warning("No data in this snapshot")
                        
            except Exception as e:
                st.error(f"Error displaying snapshot {i+1}: {e}")

        # Download all history button
        try:
            if st.button("⬇️ Download All History (CSV)"):
                csv_data = ""
                for i, s in enumerate(st.session_state.history_snapshots):
                    try:
                        if isinstance(s['rankings_df'], pd.DataFrame):
                            csv_data += f"\n=== Snapshot {i+1} - {s.get('timestamp', '')} ===\n"
                            csv_data += s['rankings_df'].to_csv(index=False) + "\n"
                        else:
                            df = pd.DataFrame(s['rankings_df'])
                            csv_data += f"\n=== Snapshot {i+1} - {s.get('timestamp', '')} ===\n"
                            csv_data += df.to_csv(index=False) + "\n"
                    except Exception as e:
                        st.warning(f"Error processing snapshot {i+1} for download: {e}")
                        continue
                
                st.download_button(
                    label="📥 Download CSV File",
                    data=csv_data,
                    file_name=f"tls_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error preparing download: {e}")

        # Danger zone - delete all history
        st.markdown("### ⚠️ Danger Zone")
        with st.form("delete_history"):
            st.warning("This action will permanently delete all history snapshots!")
            passcode = st.text_input("Enter passcode to clear history", type="password")
            confirm = st.form_submit_button("🗑️ Delete All History", type="secondary")
            
            if confirm:
                try:
                    if passcode == HISTORY_DELETE_PASSCODE:
                        # Delete from Supabase if configured
                        if supabase:
                            try:
                                supabase.table("history_snapshots").delete().neq("id", 0).execute()
                                st.success("All history deleted from database.")
                            except Exception as e:
                                st.error(f"Error deleting from database: {e}")
                        
                        # Clear local session & file
                        st.session_state.history_snapshots = []
                        save_history_snapshots_local()
                        st.success("All history cleared locally.")
                        st.rerun()
                    else:
                        st.error("Incorrect passcode.")
                except Exception as e:
                    st.error(f"Error deleting history: {e}")
                    
    except Exception as e:
        st.markdown(f'''
        <div class="error-boundary">
            <h3>⚠️ Error in History Page</h3>
            <p>An error occurred: {e}</p>
            <p>Please refresh the page and try again.</p>
        </div>
        ''', unsafe_allow_html=True)

def calculation_tester_page():
    """Calculation tester page for debugging"""
    try:
        st.header("🧮 Calculation Tester")
        st.write("Use this tool to test individual calculations and verify formulas.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Values")
            total_cost = st.number_input("Total Cost", value=0.0, step=1.0)
            lkgtc = st.number_input("LKGTC", value=0.0, step=0.001, format="%.3f")
            
            fuel_cons = st.number_input("Fuel Consumption", value=0.0, step=1.0)
            diesel_price = st.number_input("Diesel Price", value=0.0, step=1.0)
        
        with col2:
            st.subheader("Calculated Results")
            
            if lkgtc > 0:
                cost_per_lkg = safe_divide(total_cost, lkgtc)
                lkg_per_php = safe_divide(lkgtc, cost_per_lkg) if cost_per_lkg > 0 else 0
                fuel_cost = safe_divide(fuel_cons * diesel_price, 32)
                
                st.metric("Cost per LKG", f"₱{cost_per_lkg:.4f}")
                st.metric("LKG per PHP", f"{lkg_per_php:.5f}")
                st.metric("Fuel Cost", f"₱{fuel_cost:.2f}")
            else:
                st.info("Enter LKGTC value > 0 to see calculations")
                
        # Formula explanations
        st.subheader("📋 Formula Reference")
        st.markdown("""
        **Key Formulas:**
        - **Fuel Cost** = (Fuel Consumption × Diesel Price) ÷ 32
        - **Total Cost** = TLS Opn + Drivers/Hauler + Fuel Cost + T.A.&Inc
        - **Cost per LKG** = Total Cost ÷ LKGTC
        - **LKG per PHP** = LKGTC ÷ Cost per LKG
        - **KPI Score** = 0.5 × (Normalized Cost Score) + 0.5 × (Normalized LKG Score)
        """)
        
    except Exception as e:
        st.error(f"Error in calculation tester: {e}")

# --------------------
# Main application flow
# --------------------
def main():
    """Main application with error boundaries"""
    try:
        if not st.session_state.authenticated:
            login_page()
            return

        # Navigation sidebar
        st.sidebar.title("🧭 Navigation")
        st.sidebar.markdown("---")
        
        page = st.sidebar.radio(
            "Go to", 
            ["Cost Input", "Efficiency Ranking", "Cost Analysis", "History", "Calculation Tester", "Account"]
        )
        
        # Display current connection status
        st.sidebar.markdown("---")
        st.sidebar.markdown("**System Status:**")
        if supabase:
            st.sidebar.success("🟢 Database Connected")
        else:
            st.sidebar.warning("🟡 Local Storage Only")
        
        # Page routing with error boundaries
        try:
            if page == "Cost Input":
                cost_input_page()
            elif page == "Efficiency Ranking":
                ranking_page()
            elif page == "Cost Analysis":
                analysis_page()
            elif page == "History":
                history_page()
            elif page == "Calculation Tester":
                calculation_tester_page()
            elif page == "Account":
                account_page()
        except Exception as e:
            st.markdown(f'''
            <div class="error-boundary">
                <h3>⚠️ Page Error</h3>
                <p>An error occurred while loading the {page} page: {e}</p>
                <p>Please try refreshing the page or contact support if the issue persists.</p>
            </div>
            ''', unsafe_allow_html=True)
            
    except Exception as e:
        st.markdown(f'''
        <div class="error-boundary">
            <h3>🚨 Application Error</h3>
            <p>A critical error occurred: {e}</p>
            <p>Please refresh the page. If the problem persists, check your configuration.</p>
        </div>
        ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()