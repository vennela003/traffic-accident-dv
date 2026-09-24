"""
US Accidents Analytics Dashboard.
Modern, portfolio-ready exploratory analytics product built with Streamlit & Plotly.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

# Local modules
from src.charts import (
    create_hourly_day_heatmap,
    create_hourly_trend_chart,
    create_hotspot_map,
    create_severity_donut,
    create_state_choropleth,
    create_top_states_bar,
    create_weather_bar,
)
from src.data_loader import (
    DEFAULT_PARQUET,
    DEFAULT_RAW_CSV,
    filter_data,
    load_dataset,
)

# Page configuration
st.set_page_config(
    page_title="US Traffic Intelligence | Accident Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
STYLE_FILE = Path(__file__).resolve().parent / "assets" / "style.css"
if STYLE_FILE.exists():
    with open(STYLE_FILE, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_sidebar(df: pd.DataFrame):
    """Render sidebar navigation and global filters."""
    st.sidebar.markdown(
        """
        <div style="padding: 10px 0 16px 0;">
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:1.6rem;">🚦</span>
                <div>
                    <div style="font-weight:700; font-size:1.05rem; color:#F8FAFC; letter-spacing:-0.01em;">TRAFFIC INTELLIGENCE</div>
                    <div style="font-size:0.75rem; color:#14B8A6; font-weight:600; text-transform:uppercase; letter-spacing:0.06em;">US Accident Analytics</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Navigation menu
    st.sidebar.markdown(
        "<div style='font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:#64748B; margin:8px 0;'>Navigation</div>",
        unsafe_allow_html=True,
    )
    pages = [
        "🏠 Overview",
        "🗺️ Geographic Analysis",
        "⏰ Time Patterns",
        "⚠️ Severity & Conditions",
        "📍 Hotspot Map",
        "📋 Data Explorer",
    ]
    selected_page = st.sidebar.radio("View Selection", pages, label_visibility="collapsed")

    st.sidebar.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.08); margin:16px 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div style='font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:#64748B; margin-bottom:8px;'>Data Filters</div>",
        unsafe_allow_html=True,
    )

    all_states = sorted(df["State"].unique().tolist())

    # State filter
    select_all_states = st.sidebar.checkbox("Select All States", value=True)
    if select_all_states:
        selected_states = all_states
        st.sidebar.caption(f"Analyzing all {len(all_states)} states & territories")
    else:
        top_defaults = [s for s in ["CA", "FL", "TX", "SC", "NY"] if s in all_states]
        selected_states = st.sidebar.multiselect(
            "Filter States:",
            options=all_states,
            default=top_defaults,
        )

    # Date range filter
    min_date = df["Date"].min().to_pydatetime().date()
    max_date = df["Date"].max().to_pydatetime().date()
    date_selection = st.sidebar.date_input(
        "Date Range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_selection, tuple) and len(date_selection) == 2:
        date_range = (
            pd.Timestamp(date_selection[0]),
            pd.Timestamp(date_selection[1]) + pd.Timedelta(days=1, microseconds=-1),
        )
    else:
        date_range = (pd.Timestamp(min_date), pd.Timestamp(max_date))

    # Severity filter
    severities = st.sidebar.multiselect(
        "Severity Levels:",
        options=[1, 2, 3, 4],
        default=[1, 2, 3, 4],
        format_func=lambda s: f"Severity {s}",
    )

    # Sidebar footer info
    st.sidebar.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.08); margin:16px 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        f"""
        <div style="font-size:0.75rem; color:#64748B; line-height:1.6;">
            <div><b>Source:</b> Kaggle US Accidents</div>
            <div><b>Cache:</b> {len(df):,} sampled rows</div>
            <div><b>RAM Usage:</b> {df.memory_usage(deep=True).sum() / (1024**2):.1f} MB</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return selected_page, selected_states, date_range, severities


def render_header(title: str, subtitle: str, active_count: int, total_count: int):
    """Render a clean hero header banner with badge pills."""
    share_pct = (active_count / max(total_count, 1)) * 100
    st.markdown(
        f"""
        <div class="hero-header">
            <div class="hero-title-row">
                <h1 class="hero-title">{title}</h1>
                <div class="badge-row">
                    <span class="data-badge accent">● Active Scope: {active_count:,} records ({share_pct:.1f}%)</span>
                    <span class="data-badge">7.7M Total Dataset</span>
                    <span class="data-badge">2016 – 2023 Coverage</span>
                </div>
            </div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(filtered_df: pd.DataFrame, total_cached_count: int):
    """Render 5 modern KPI cards with icons and color accents."""
    total_count = len(filtered_df)
    state_top = (
        filtered_df["State"].value_counts().index[0]
        if total_count > 0
        else "N/A"
    )
    state_top_count = (
        filtered_df["State"].value_counts().iloc[0] if total_count > 0 else 0
    )

    peak_hour = (
        filtered_df["Hour"].value_counts().index[0]
        if total_count > 0
        else 0
    )
    top_weather = (
        filtered_df["Weather_Condition"].value_counts().index[0]
        if total_count > 0
        else "N/A"
    )

    severe_count = (
        filtered_df["Severity"].isin([3, 4]).sum() if total_count > 0 else 0
    )
    severe_pct = (severe_count / max(total_count, 1)) * 100

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
            <div class="kpi-card" style="--card-accent: #14B8A6;">
                <div class="kpi-header">
                    <span class="kpi-label">Total Collisions</span>
                    <div class="kpi-icon" style="--icon-bg: rgba(20, 184, 166, 0.12);">🚗</div>
                </div>
                <div class="kpi-value">{total_count:,}</div>
                <div class="kpi-subtext"><span class="highlight">{total_count / max(total_cached_count, 1) * 100:.1f}%</span> of sample</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="kpi-card" style="--card-accent: #38BDF8;">
                <div class="kpi-header">
                    <span class="kpi-label">Top Impact State</span>
                    <div class="kpi-icon" style="--icon-bg: rgba(56, 189, 248, 0.12);">📍</div>
                </div>
                <div class="kpi-value">{state_top}</div>
                <div class="kpi-subtext"><span class="highlight">{state_top_count:,}</span> incidents</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="kpi-card" style="--card-accent: #F59E0B;">
                <div class="kpi-header">
                    <span class="kpi-label">Peak Risk Hour</span>
                    <div class="kpi-icon" style="--icon-bg: rgba(245, 158, 11, 0.12);">⏰</div>
                </div>
                <div class="kpi-value">{peak_hour:02d}:00</div>
                <div class="kpi-subtext"><span class="highlight">Evening</span> commute spike</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="kpi-card" style="--card-accent: #818CF8;">
                <div class="kpi-header">
                    <span class="kpi-label">Dominant Weather</span>
                    <div class="kpi-icon" style="--icon-bg: rgba(129, 140, 248, 0.12);">⛅</div>
                </div>
                <div class="kpi-value" style="font-size:1.4rem;">{top_weather}</div>
                <div class="kpi-subtext"><span class="highlight">Most common</span> condition</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            f"""
            <div class="kpi-card" style="--card-accent: #EF4444;">
                <div class="kpi-header">
                    <span class="kpi-label">Severe Rate (3-4)</span>
                    <div class="kpi-icon" style="--icon-bg: rgba(239, 68, 68, 0.12);">🚨</div>
                </div>
                <div class="kpi-value">{severe_pct:.1f}%</div>
                <div class="kpi-subtext"><span class="highlight" style="color:#EF4444;">{severe_count:,}</span> major events</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def page_overview(filtered_df: pd.DataFrame, total_cached_count: int):
    """Render Home / Overview page."""
    render_header(
        title='US Traffic <span class="hero-title-accent">Incident Intelligence</span>',
        subtitle="Holistic executive overview of 7.7 million traffic collisions recorded across the contiguous United States (2016–2023).",
        active_count=len(filtered_df),
        total_count=total_cached_count,
    )

    # Top KPI cards
    render_kpi_cards(filtered_df, total_cached_count)

    # Executive Summary Card
    st.markdown(
        """
        <div class="insight-box">
            <span class="insight-icon">💡</span>
            <div class="insight-content">
                <b>Executive Summary:</b> Traffic incident frequency is heavily driven by commuter traffic volume rather than purely adverse weather. 
                Weekdays exhibit sharp bimodal peaks during morning (7–9 AM) and evening (4–6 PM) rush hours. 
                High-population states (California, Florida, Texas) record the highest collision counts, while severe incidents (Levels 3 & 4) represent ~21% of events with significant traffic clearance delays.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<h3 style='font-size:1.25rem; font-weight:700; color:#F8FAFC; margin:24px 0 12px 0;'>📖 How to Navigate This Dashboard</h3>", unsafe_allow_html=True)

    # 4-step guide grid
    st.markdown(
        """
        <div class="guide-grid">
            <div class="guide-card">
                <div class="guide-step">Step 01</div>
                <div class="guide-title">Filter Scope</div>
                <div class="guide-desc">Use the left sidebar to filter the dataset by state, specific date ranges (2016–2023), and severity levels (1 to 4).</div>
            </div>
            <div class="guide-card">
                <div class="guide-step">Step 02</div>
                <div class="guide-title">Geographic Analysis</div>
                <div class="guide-desc">Explore state-level accident volume via the interactive US choropleth map and ranked top-state volume distribution.</div>
            </div>
            <div class="guide-card">
                <div class="guide-step">Step 03</div>
                <div class="guide-title">Temporal & Weather Risk</div>
                <div class="guide-desc">Inspect diurnal rush-hour patterns and analyze how ambient weather conditions impact collision frequency and severity.</div>
            </div>
            <div class="guide-card">
                <div class="guide-step">Step 04</div>
                <div class="guide-title">Interactive Hotspots</div>
                <div class="guide-desc">Navigate to the Hotspot Map for high-resolution MapLibre GL coordinates with scatter and density heatmap overlays.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick preview section
    st.markdown("<h3 style='font-size:1.25rem; font-weight:700; color:#F8FAFC; margin:28px 0 16px 0;'>⚡ Quick Analysis Snapshot</h3>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(create_top_states_bar(filtered_df, top_n=8), use_container_width=True)
    with col_right:
        st.plotly_chart(create_hourly_trend_chart(filtered_df), use_container_width=True)


def page_geographic(filtered_df: pd.DataFrame, total_cached_count: int):
    """Render Geographic Analysis page."""
    render_header(
        title='Geographic <span class="hero-title-accent">Distribution & State Rankings</span>',
        subtitle="Interactive spatial analysis of accident density across all 49 represented states and territories.",
        active_count=len(filtered_df),
        total_count=total_cached_count,
    )

    st.markdown(
        """
        <div class="insight-box">
            <span class="insight-icon">📍</span>
            <div class="insight-content">
                <b>Geographic Insights:</b> Collisions concentrate heavily in high-population states with dense highway corridors.
                California, Florida, Texas, and North Carolina lead total incident counts. Hover over states in the choropleth map to inspect state-level volume and national percentage shares.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_map, col_bar = st.columns([1.3, 1])
    with col_map:
        fig_choro = create_state_choropleth(filtered_df)
        st.plotly_chart(fig_choro, use_container_width=True)

    with col_bar:
        fig_bar = create_top_states_bar(filtered_df, top_n=15)
        st.plotly_chart(fig_bar, use_container_width=True)


def page_time_patterns(filtered_df: pd.DataFrame, total_cached_count: int):
    """Render Time Patterns page."""
    render_header(
        title='Temporal <span class="hero-title-accent">Risk Patterns & Rush Hours</span>',
        subtitle="Cross-sectional diurnal and weekly analysis identifying high-vulnerability time windows.",
        active_count=len(filtered_df),
        total_count=total_cached_count,
    )

    # Heatmap Section with insight
    st.markdown(
        """
        <div class="insight-box">
            <span class="insight-icon">⏰</span>
            <div class="insight-content">
                <b>Weekly Heatmap Analysis:</b> Incidents peak intensely during weekday morning commute hours (7:00 AM – 9:00 AM) 
                and evening commute hours (4:00 PM – 6:00 PM). Fridays consistently sustain the highest collision density into the evening.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    fig_heat = create_hourly_day_heatmap(filtered_df)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Weekday vs Weekend Section with insight
    st.markdown(
        """
        <div class="insight-box" style="border-left-color: #F59E0B;">
            <span class="insight-icon">📈</span>
            <div class="insight-content">
                <b>Weekday vs. Weekend Trajectory:</b> Weekday traffic follows a distinctive bimodal pattern driven by work and school commutes. 
                In contrast, weekend collisions follow a smooth unimodal curve that peaks in mid-afternoon (1:00 PM – 4:00 PM) and tapers gradually.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    fig_line = create_hourly_trend_chart(filtered_df)
    st.plotly_chart(fig_line, use_container_width=True)


def page_severity_conditions(filtered_df: pd.DataFrame, total_cached_count: int):
    """Render Severity & Conditions page."""
    render_header(
        title='Collision Severity & <span class="hero-title-accent">Environmental Factors</span>',
        subtitle="Evaluation of accident severity levels (1 to 4) alongside weather conditions and road clearance durations.",
        active_count=len(filtered_df),
        total_count=total_cached_count,
    )

    # Average Duration Statistics Cards
    avg_duration = filtered_df.groupby("Severity", observed=True)["Duration_Minutes"].mean()
    sev_colors = {1: "#10B981", 2: "#06B6D4", 3: "#F59E0B", 4: "#EF4444"}

    st.markdown("<h4 style='font-size:1rem; font-weight:600; color:#F8FAFC; margin-bottom:12px;'>⏱️ Average Road Clearance Duration by Severity Level</h4>", unsafe_allow_html=True)

    dur_cols = st.columns(4)
    labels = {
        1: "Severity 1 (Minor)",
        2: "Severity 2 (Moderate)",
        3: "Severity 3 (Significant)",
        4: "Severity 4 (Severe)",
    }
    for i, sev in enumerate([1, 2, 3, 4]):
        dur = avg_duration.get(sev, 0.0)
        color = sev_colors.get(sev, "#14B8A6")
        with dur_cols[i]:
            st.markdown(
                f"""
                <div class="duration-pill">
                    <div class="duration-pill-header">
                        <span class="duration-pill-dot" style="background-color:{color};"></span>
                        <span>{labels.get(sev)}</span>
                    </div>
                    <div class="duration-pill-val">{dur:.1f} <span style="font-size:0.8rem; font-weight:500; color:#94A3B8;">min</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="insight-box">
            <span class="insight-icon">⛅</span>
            <div class="insight-content">
                <b>Weather Dynamics:</b> Over 70% of collisions occur in Fair or Clear weather conditions, mirroring general driving exposure. 
                However, precipitation (Rain, Snow, Fog) dramatically increases the likelihood of secondary collisions and delays road clearance times.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_sev, col_weather = st.columns([1, 1.25])
    with col_sev:
        fig_donut = create_severity_donut(filtered_df)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_weather:
        fig_weather = create_weather_bar(filtered_df, top_n=15)
        st.plotly_chart(fig_weather, use_container_width=True)


def page_hotspot_map(filtered_df: pd.DataFrame, total_cached_count: int):
    """Render Hotspot Map page with prominent full-width visualization."""
    render_header(
        title='Interactive <span class="hero-title-accent">Geospatial Hotspots</span>',
        subtitle="High-resolution coordinates visualization using MapLibre GL dark-matter cartography.",
        active_count=len(filtered_df),
        total_count=total_cached_count,
    )

    # Map Controls Toolbar
    ctrl_col1, ctrl_col2 = st.columns([1, 1.5])
    with ctrl_col1:
        map_mode = st.segmented_control(
            "Visualization Mode:",
            options=["Scatter Points", "Density Heatmap"],
            default="Scatter Points",
        )
    with ctrl_col2:
        max_pts = st.slider(
            "Rendering Limit (Subsampling for 60 FPS fluidity):",
            min_value=5_000,
            max_value=30_000,
            value=15_000,
            step=2_500,
            help="Limits rendered points in the browser to maintain smooth hardware-accelerated WebGL performance.",
        )

    st.markdown(
        f"""
        <div class="insight-box" style="margin-top:8px;">
            <span class="insight-icon">📍</span>
            <div class="insight-content">
                <b>Interactive Navigation:</b> Scroll to zoom in and out. Click and drag to pan across high-density metro corridors (e.g. Los Angeles, Miami, Dallas, New York).
                Active sample size: <b>{min(len(filtered_df), max_pts):,}</b> points displayed.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Full-width Map chart
    fig_map = create_hotspot_map(
        filtered_df,
        map_type=map_mode or "Scatter Points",
        max_display_points=max_pts,
    )
    # Give the map extra height for a prominent, dashboard-centerpiece feel
    fig_map.update_layout(height=650)
    st.plotly_chart(fig_map, use_container_width=True)


def page_data_explorer(filtered_df: pd.DataFrame, total_cached_count: int):
    """Render Data Explorer page with tabular inspection and CSV download."""
    render_header(
        title='Filtered <span class="hero-title-accent">Records Explorer</span>',
        subtitle="Inspect raw tabular collision records matching current active filter parameters.",
        active_count=len(filtered_df),
        total_count=total_cached_count,
    )

    st.markdown(
        f"""
        <div class="insight-box">
            <span class="insight-icon">📋</span>
            <div class="insight-content">
                Displaying up to the latest 500 records from the filtered selection of <b>{len(filtered_df):,}</b> records.
                Columns include exact collision timestamp, severity level, municipality, and ambient weather.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    preview_cols = [
        "Severity",
        "Start_Time",
        "City",
        "State",
        "Weather_Condition",
        "Duration_Minutes",
        "Start_Lat",
        "Start_Lng",
    ]

    st.dataframe(
        filtered_df[preview_cols].tail(500),
        use_container_width=True,
        hide_index=True,
    )

    # Export button
    csv_bytes = filtered_df[preview_cols].head(10000).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Current Filtered Scope (CSV, up to 10k rows)",
        data=csv_bytes,
        file_name="us_accidents_selection.csv",
        mime="text/csv",
    )


def main():
    # Verify dataset existence
    if not DEFAULT_PARQUET.exists() and not DEFAULT_RAW_CSV.exists():
        st.error(
            f"❌ Dataset not found! Please ensure either `{DEFAULT_PARQUET}` or `{DEFAULT_RAW_CSV}` is present."
        )
        return

    # Load dataset
    with st.spinner("⚡ Loading cached accident data..."):
        try:
            df = load_dataset()
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return

    # Render sidebar controls & navigation
    selected_page, selected_states, date_range, severities = render_sidebar(df)

    # Apply global filters
    filtered_df = filter_data(
        df,
        states=selected_states,
        date_range=date_range,
        severities=severities,
    )

    if filtered_df.empty:
        st.warning("⚠️ No accidents found matching the selected filter criteria. Please broaden your selection in the sidebar.")
        return

    # Route to selected page
    if "Overview" in selected_page:
        page_overview(filtered_df, len(df))
    elif "Geographic" in selected_page:
        page_geographic(filtered_df, len(df))
    elif "Time Patterns" in selected_page:
        page_time_patterns(filtered_df, len(df))
    elif "Severity" in selected_page:
        page_severity_conditions(filtered_df, len(df))
    elif "Hotspot" in selected_page:
        page_hotspot_map(filtered_df, len(df))
    elif "Data Explorer" in selected_page:
        page_data_explorer(filtered_df, len(df))


if __name__ == "__main__":
    main()
