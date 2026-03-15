"""
Vehicle Counter Dashboard
- Xem dữ liệu realtime và lịch sử
- Biểu đồ thống kê
- Export dữ liệu
"""
import streamlit as st
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import time
import os

st.set_page_config(
    page_title="Vehicle Counter Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- DATABASE CONFIG ---
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "vehicle_counter",
    "user": "postgres",
    "password": "Sinhnam2004"
}


def get_db_connection():
    """Kết nối PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            database=DB_CONFIG["database"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"]
        )
        return conn
    except Exception as e:
        st.error(f"❌ Không thể kết nối database: {e}")
        st.info("""
        **Kiểm tra:**
        - PostgreSQL đang chạy?
        - Database `vehicle_counter` đã được tạo?
        - Username/password đúng?
        
        **Tạo database:**
        ```sql
        CREATE DATABASE vehicle_counter;
        ```
        """)
        return None


def load_recent_data(hours: int = 24) -> pd.DataFrame:
    """Load dữ liệu trong N giờ gần nhất"""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    
    query = f"""
        SELECT timestamp, count_in, count_out, total, fps, session_id
        FROM vehicle_counts
        WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
        ORDER BY timestamp DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


def load_sessions() -> pd.DataFrame:
    """Load danh sách sessions"""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    
    query = """
        SELECT id, start_time, end_time, total_in, total_out, status
        FROM sessions
        ORDER BY start_time DESC
        LIMIT 50
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df


def load_events(limit: int = 100) -> pd.DataFrame:
    """Load events gần nhất"""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    
    query = f"""
        SELECT timestamp, track_id, direction, session_id
        FROM vehicle_events
        ORDER BY timestamp DESC
        LIMIT {limit}
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


def get_today_summary() -> dict:
    """Thống kê hôm nay"""
    conn = get_db_connection()
    if conn is None:
        return {'total_in': 0, 'total_out': 0, 'total': 0}
    
    query = """
        SELECT 
            COALESCE(SUM(count_in), 0) as total_in,
            COALESCE(SUM(count_out), 0) as total_out
        FROM vehicle_counts
        WHERE DATE(timestamp) = CURRENT_DATE
    """
    cursor = conn.cursor()
    cursor.execute(query)
    row = cursor.fetchone()
    conn.close()
    
    return {
        'total_in': int(row[0]) if row else 0,
        'total_out': int(row[1]) if row else 0,
        'total': int((row[0] or 0) + (row[1] or 0))
    }


def get_hourly_data(date: str = None) -> pd.DataFrame:
    """Thống kê theo giờ"""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    
    if date is None:
        date_filter = "DATE(timestamp) = CURRENT_DATE"
    else:
        date_filter = f"DATE(timestamp) = '{date}'"
    
    query = f"""
        SELECT 
            EXTRACT(HOUR FROM timestamp)::INTEGER as hour,
            SUM(count_in) as total_in,
            SUM(count_out) as total_out,
            SUM(total) as total
        FROM vehicle_counts
        WHERE {date_filter}
        GROUP BY EXTRACT(HOUR FROM timestamp)
        ORDER BY hour
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df


def get_daily_data(days: int = 30) -> pd.DataFrame:
    """Thống kê theo ngày"""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    
    query = f"""
        SELECT 
            DATE(timestamp) as date,
            SUM(count_in) as total_in,
            SUM(count_out) as total_out,
            SUM(total) as total
        FROM vehicle_counts
        WHERE timestamp >= NOW() - INTERVAL '{days} days'
        GROUP BY DATE(timestamp)
        ORDER BY date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df


# --- MAIN UI ---
st.title("📊 Vehicle Counter Dashboard")

# Sidebar
with st.sidebar:
    st.header("⚙️ Cài đặt")
    
    auto_refresh = st.checkbox("Tự động refresh", value=False)
    refresh_interval = st.slider("Refresh interval (giây)", 5, 60, 10)
    
    st.markdown("---")
    
    st.subheader("🐘 PostgreSQL")
    st.code(f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    
    if st.button("🔄 Refresh ngay"):
        st.rerun()
    
    st.markdown("---")
    st.subheader("🔧 Cấu hình DB")
    with st.expander("Sửa cấu hình"):
        DB_CONFIG["host"] = st.text_input("Host", value=DB_CONFIG["host"])
        DB_CONFIG["port"] = st.number_input("Port", value=DB_CONFIG["port"])
        DB_CONFIG["database"] = st.text_input("Database", value=DB_CONFIG["database"])
        DB_CONFIG["user"] = st.text_input("User", value=DB_CONFIG["user"])
        DB_CONFIG["password"] = st.text_input("Password", value=DB_CONFIG["password"], type="password")

# Check database connection
test_conn = get_db_connection()
if test_conn is None:
    st.stop()
else:
    test_conn.close()

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Tổng quan", "📊 Biểu đồ", "📋 Dữ liệu chi tiết", "🔧 Sessions"])

with tab1:
    st.subheader("📈 Tổng quan hôm nay")
    
    today = get_today_summary()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🟢 Tổng VÀO", today['total_in'])
    with col2:
        st.metric("🔴 Tổng RA", today['total_out'])
    with col3:
        st.metric("📊 Tổng cộng", today['total'])
    
    st.markdown("---")
    
    # Events gần nhất
    st.subheader("🚗 Xe đi qua gần nhất")
    events = load_events(20)
    
    if not events.empty:
        for _, row in events.iterrows():
            direction_icon = "🟢" if row['direction'] == 'IN' else "🔴"
            direction_text = "VÀO" if row['direction'] == 'IN' else "RA"
            st.text(f"{direction_icon} {row['timestamp'].strftime('%H:%M:%S')} - ID: {row['track_id']} - {direction_text}")
    else:
        st.info("Chưa có dữ liệu")

with tab2:
    st.subheader("📊 Biểu đồ thống kê")
    
    chart_type = st.selectbox("Loại biểu đồ", ["Theo giờ (Hôm nay)", "Theo ngày (30 ngày)"])
    
    if chart_type == "Theo giờ (Hôm nay)":
        hourly = get_hourly_data()
        
        if not hourly.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=hourly['hour'], y=hourly['total_in'], name='VÀO', marker_color='green'))
            fig.add_trace(go.Bar(x=hourly['hour'], y=hourly['total_out'], name='RA', marker_color='red'))
            fig.update_layout(
                title="Số lượng xe theo giờ",
                xaxis_title="Giờ",
                yaxis_title="Số lượng",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Chưa có dữ liệu hôm nay")
    
    else:
        daily = get_daily_data(30)
        
        if not daily.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily['date'], y=daily['total_in'], name='VÀO', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=daily['date'], y=daily['total_out'], name='RA', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=daily['date'], y=daily['total'], name='Tổng', line=dict(color='blue', dash='dash')))
            fig.update_layout(
                title="Số lượng xe theo ngày",
                xaxis_title="Ngày",
                yaxis_title="Số lượng"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Chưa có dữ liệu")

with tab3:
    st.subheader("📋 Dữ liệu chi tiết")
    
    hours = st.slider("Hiển thị dữ liệu trong ... giờ gần nhất", 1, 72, 24)
    
    df = load_recent_data(hours)
    
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        
        # Export
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Tải CSV",
            csv,
            f"vehicle_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )
    else:
        st.info("Chưa có dữ liệu")

with tab4:
    st.subheader("🔧 Sessions")
    
    sessions = load_sessions()
    
    if not sessions.empty:
        st.dataframe(sessions, use_container_width=True)
    else:
        st.info("Chưa có sessions")

# Auto refresh
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# Footer
st.markdown("---")
st.caption(f"🕐 Cập nhật lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
