"""
Interactive Plotly visualizations for the US Accidents Dashboard.
Polished with a modern dark slate / teal analytics theme.
Includes:
- State Choropleth & Bar Charts
- Hourly & Day-of-Week Heatmaps and Rush-Hour Curves
- Severity Distribution Donut & Breakdown
- Weather Condition Impact Bar Charts
- Geospatial Hotspot Maps (Scatter & Density)
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Color schemes aligned with modern dark theme
SEVERITY_COLORS = {
    1: "#10B981",  # Emerald / Minor
    2: "#06B6D4",  # Cyan / Moderate
    3: "#F59E0B",  # Amber / Significant
    4: "#EF4444",  # Rose Red / Severe
}

PLOT_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94A3B8", family="Inter, -apple-system, Segoe UI, sans-serif", size=12),
    margin=dict(l=40, r=40, t=55, b=40),
)


def create_state_choropleth(df: pd.DataFrame) -> go.Figure:
    """Generate US Choropleth map of accident volumes by state."""
    state_counts = (
        df.groupby("State", observed=True)
        .size()
        .reset_index(name="Accident_Count")
    )
    total_accidents = len(df)
    state_counts["Percentage"] = (
        state_counts["Accident_Count"] / max(total_accidents, 1) * 100
    ).round(2)

    fig = px.choropleth(
        state_counts,
        locations="State",
        locationmode="USA-states",
        color="Accident_Count",
        scope="usa",
        color_continuous_scale=[
            (0.0, "#0F2B36"),
            (0.3, "#0D9488"),
            (0.7, "#14B8A6"),
            (1.0, "#5EEAD4"),
        ],
        hover_data={"State": True, "Accident_Count": ":,", "Percentage": ":.2f%"},
        labels={"Accident_Count": "Accidents", "Percentage": "Share (%)"},
        title="<b>Accident Frequency by US State</b>",
    )

    fig.update_layout(
        **PLOT_LAYOUT_DEFAULTS,
        title_font=dict(color="#F8FAFC", size=15),
        geo=dict(
            scope="usa",
            projection_type="albers usa",
            showlakes=True,
            lakecolor="#0B0F19",
            bgcolor="rgba(0,0,0,0)",
            landcolor="#1E293B",
            showland=True,
            showcountries=False,
            coastlinecolor="rgba(255, 255, 255, 0.1)",
        ),
        coloraxis_colorbar=dict(
            title=dict(text="Accidents", font=dict(color="#94A3B8", size=11)),
            tickfont=dict(color="#94A3B8", size=10),
            thickness=14,
            len=0.75,
        ),
    )
    return fig


def create_top_states_bar(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart showing top states by accident volume."""
    state_counts = (
        df.groupby("State", observed=True)
        .size()
        .reset_index(name="Accident_Count")
        .sort_values(by="Accident_Count", ascending=False)
        .head(top_n)
    )
    # Sort ascending for clean bottom-up horizontal bar
    state_counts = state_counts.sort_values("Accident_Count", ascending=True)

    fig = px.bar(
        state_counts,
        x="Accident_Count",
        y="State",
        orientation="h",
        color="Accident_Count",
        color_continuous_scale=[
            (0.0, "#0E7490"),
            (0.5, "#0D9488"),
            (1.0, "#14B8A6"),
        ],
        text="Accident_Count",
        labels={"Accident_Count": "Accident Count", "State": "State"},
        title=f"<b>Top {top_n} States by Incident Volume</b>",
    )

    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        cliponaxis=False,
        marker=dict(line=dict(width=0)),
    )
    fig.update_layout(
        **PLOT_LAYOUT_DEFAULTS,
        title_font=dict(color="#F8FAFC", size=15),
        coloraxis_showscale=False,
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.06)",
            zeroline=False,
            tickfont=dict(color="#94A3B8"),
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color="#F8FAFC", size=11),
        ),
    )
    return fig


def create_hourly_day_heatmap(df: pd.DataFrame) -> go.Figure:
    """Generate 2D heatmap cross-tabulating Day of Week vs. Hour of Day."""
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    matrix = (
        df.groupby(["Day_Name", "Hour"], observed=False)
        .size()
        .unstack(fill_value=0)
    )
    matrix = matrix.reindex(day_order)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=[f"{h:02d}:00" for h in matrix.columns],
            y=matrix.index.tolist(),
            colorscale=[
                (0.0, "#0B0F19"),
                (0.2, "#132338"),
                (0.45, "#0E7490"),
                (0.7, "#14B8A6"),
                (1.0, "#F59E0B"),
            ],
            hoverongaps=False,
            hovertemplate="<b>Day:</b> %{y}<br><b>Time:</b> %{x}<br><b>Accidents:</b> %{z:,}<extra></extra>",
            colorbar=dict(
                title=dict(text="Incidents", font=dict(color="#94A3B8", size=11)),
                tickfont=dict(color="#94A3B8", size=10),
                thickness=14,
                len=0.75,
            ),
        )
    )

    fig.update_layout(
        **PLOT_LAYOUT_DEFAULTS,
        title="<b>Accident Concentration: Day of Week vs. Hour of Day</b>",
        title_font=dict(color="#F8FAFC", size=15),
        xaxis=dict(
            title=dict(text="Hour of Day (24h)", font=dict(color="#94A3B8", size=11)),
            tickangle=-45,
            gridcolor="rgba(255, 255, 255, 0.05)",
            tickfont=dict(color="#94A3B8", size=10),
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
            tickfont=dict(color="#F8FAFC", size=11),
        ),
    )
    return fig


def create_hourly_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Line chart comparing accident distribution by hour for Weekdays vs. Weekends."""
    df_copy = df[["Hour", "Day_of_Week"]].copy()
    df_copy["Day_Type"] = df_copy["Day_of_Week"].apply(
        lambda d: "Weekend (Sat-Sun)" if d >= 5 else "Weekday (Mon-Fri)"
    )

    hourly = (
        df_copy.groupby(["Hour", "Day_Type"], observed=False)
        .size()
        .reset_index(name="Accident_Count")
    )

    fig = px.line(
        hourly,
        x="Hour",
        y="Accident_Count",
        color="Day_Type",
        markers=True,
        color_discrete_map={
            "Weekday (Mon-Fri)": "#14B8A6",
            "Weekend (Sat-Sun)": "#F59E0B",
        },
        title="<b>Hourly Accident Trajectory: Weekdays vs. Weekends</b>",
        labels={"Hour": "Hour of Day (0-23)", "Accident_Count": "Accidents"},
    )

    fig.update_traces(line=dict(width=2.5), marker=dict(size=6))
    fig.update_layout(
        **PLOT_LAYOUT_DEFAULTS,
        title_font=dict(color="#F8FAFC", size=15),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text="",
            font=dict(color="#F8FAFC", size=11),
        ),
        xaxis=dict(
            tickmode="linear",
            tick0=0,
            dtick=2,
            gridcolor="rgba(255, 255, 255, 0.06)",
            showgrid=True,
            tickfont=dict(color="#94A3B8"),
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            showgrid=True,
            tickfont=dict(color="#94A3B8"),
        ),
    )
    return fig


def create_severity_donut(df: pd.DataFrame) -> go.Figure:
    """Generate Donut chart showing distribution across Severity levels (1 to 4)."""
    sev_counts = (
        df["Severity"]
        .value_counts()
        .sort_index()
        .reset_index(name="Count")
    )
    sev_counts.columns = ["Severity", "Count"]

    severity_labels = {
        1: "Severity 1 (Minor impact / Short delay)",
        2: "Severity 2 (Moderate impact)",
        3: "Severity 3 (Significant delay)",
        4: "Severity 4 (Severe road closure)",
    }
    sev_counts["Label"] = sev_counts["Severity"].map(severity_labels).fillna(
        sev_counts["Severity"].astype(str)
    )
    colors = [SEVERITY_COLORS.get(int(s), "#9E9E9E") for s in sev_counts["Severity"]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=sev_counts["Label"],
                values=sev_counts["Count"],
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="#0B0F19", width=2.5)),
                textinfo="percent",
                hoverinfo="label+value+percent",
                hovertemplate="<b>%{label}</b><br>Incidents: %{value:,}<br>Share: %{percent}<extra></extra>",
            )
        ]
    )

    total_incidents = len(df)
    fig.update_layout(
        **PLOT_LAYOUT_DEFAULTS,
        title="<b>Incident Severity Breakdown</b>",
        title_font=dict(color="#F8FAFC", size=15),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5,
            font=dict(color="#94A3B8", size=11),
        ),
        annotations=[
            dict(
                text=f"<b>{total_incidents:,}</b><br><span style='font-size:10px;color:#94A3B8;'>Incidents</span>",
                x=0.5,
                y=0.5,
                font=dict(size=16, color="#F8FAFC"),
                showarrow=False,
            )
        ],
    )
    return fig


def create_weather_bar(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart showing top weather conditions during accidents."""
    weather_counts = (
        df.groupby("Weather_Condition", observed=True)
        .size()
        .reset_index(name="Accident_Count")
        .sort_values(by="Accident_Count", ascending=False)
        .head(top_n)
    )
    weather_counts = weather_counts.sort_values("Accident_Count", ascending=True)

    fig = px.bar(
        weather_counts,
        x="Accident_Count",
        y="Weather_Condition",
        orientation="h",
        color="Accident_Count",
        color_continuous_scale=[
            (0.0, "#1E293B"),
            (0.5, "#F59E0B"),
            (1.0, "#FBBF24"),
        ],
        text="Accident_Count",
        labels={
            "Accident_Count": "Accident Count",
            "Weather_Condition": "Weather Condition",
        },
        title=f"<b>Top {top_n} Atmospheric Conditions at Collision Time</b>",
    )

    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        cliponaxis=False,
        marker=dict(line=dict(width=0)),
    )
    fig.update_layout(
        **PLOT_LAYOUT_DEFAULTS,
        title_font=dict(color="#F8FAFC", size=15),
        coloraxis_showscale=False,
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.06)",
            zeroline=False,
            tickfont=dict(color="#94A3B8"),
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color="#F8FAFC", size=11),
        ),
    )
    return fig


def create_hotspot_map(
    df: pd.DataFrame,
    map_type: str = "Scatter Points",
    max_display_points: int = 15_000,
) -> go.Figure:
    """
    Render MapLibre geospatial map of accident hotspots.
    Dynamically downsamples when point count is very high to maintain 60 FPS.
    """
    sample_df = df
    if len(df) > max_display_points:
        sample_df = df.sample(n=max_display_points, random_state=42)

    center_lat = float(sample_df["Start_Lat"].median()) if len(sample_df) else 39.8283
    center_lon = float(sample_df["Start_Lng"].median()) if len(sample_df) else -98.5795

    if map_type == "Density Heatmap":
        fig = px.density_map(
            sample_df,
            lat="Start_Lat",
            lon="Start_Lng",
            z="Severity",
            radius=12,
            center=dict(lat=center_lat, lon=center_lon),
            zoom=3.8,
            map_style="carto-darkmatter",
            title=f"<b>Accident Hotspot Density (Rendering {len(sample_df):,} Points)</b>",
        )
    else:
        sample_df = sample_df.copy()
        sample_df["Severity_Label"] = sample_df["Severity"].apply(lambda s: f"Severity {s}")
        color_map = {f"Severity {k}": v for k, v in SEVERITY_COLORS.items()}

        fig = px.scatter_map(
            sample_df,
            lat="Start_Lat",
            lon="Start_Lng",
            color="Severity_Label",
            color_discrete_map=color_map,
            hover_name="City",
            hover_data={
                "State": True,
                "Weather_Condition": True,
                "Start_Lat": ":.3f",
                "Start_Lng": ":.3f",
                "Severity_Label": False,
            },
            center=dict(lat=center_lat, lon=center_lon),
            zoom=3.8,
            map_style="carto-darkmatter",
            title=f"<b>Accident Hotspots by Severity (Rendering {len(sample_df):,} Points)</b>",
        )
        fig.update_traces(marker=dict(size=4.5, opacity=0.75))

    fig.update_layout(
        **PLOT_LAYOUT_DEFAULTS,
        title_font=dict(color="#F8FAFC", size=15),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text="",
            font=dict(color="#F8FAFC", size=11),
        ),
    )
    return fig
