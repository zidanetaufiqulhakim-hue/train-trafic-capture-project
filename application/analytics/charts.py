"""
Plotly chart builders for the analytics section.

Each function receives an aggregated DataFrame and returns a go.Figure.
All charts share _BASE_LAYOUT for consistent BI-standard styling.

Public functions:
  passenger_timeline_chart(time_series, chart_type) — passenger increases over time by destination (primary)
  passenger_heatmap_chart(heatmap)                  — station × time crowd heatmap
  passenger_pct_stacked_bar(time_agg)               — % share per station per time bucket
  passenger_volume_bar(station_agg)                 — total absolute counts per station
  station_share_pie(station_agg)                    — overall share donut chart
  figure_to_bytes(fig, fmt)                         — renders to image bytes for download
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


# ── Design system ──────────────────────────────────────────────────────────────

# Professional palette anchored in the app's forest green brand.
# Selected for visual distinction and approximate colorblind safety.
CHART_PALETTE = [
    "#166534",  # forest green   (brand primary)
    "#0369A1",  # ocean blue
    "#B45309",  # amber
    "#7C3AED",  # violet
    "#0F766E",  # dark teal
    "#BE185D",  # rose
    "#4D7C0F",  # olive
    "#C2410C",  # burnt orange
    "#1D4ED8",  # royal blue
    "#6D28D9",  # purple
]

# Shared layout applied to every chart via fig.update_layout(**_BASE_LAYOUT, ...).
# Colors here target light mode; CSS in app.py re-tints Plotly's SVG in dark mode.
# The hover tooltip is dark in both modes — it reads as a distinctive popover
# in light mode and matches the surface in dark mode.
_BASE_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, system-ui, sans-serif", size=13, color="#334155"),
    title_font=dict(size=15, color="#0F172A", weight=700),
    margin=dict(t=58, b=48, l=56, r=28),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    hoverlabel=dict(
        bgcolor="rgba(15,23,42,.96)",
        font_color="#FFFFFF",
        font_size=12,
        bordercolor="rgba(100,116,139,.4)",
        namelength=-1,
    ),
)

_LEGEND_BASE = dict(
    bgcolor="rgba(0,0,0,0)",
    borderwidth=0,
    font=dict(size=12, color="#334155"),
    title_font=dict(size=12, color="#475569", weight=700),
)


# ── Chart builders ─────────────────────────────────────────────────────────────


def passenger_heatmap_chart(heatmap: pd.DataFrame) -> go.Figure:
    """Crowd heatmap: station rows × HH:MM:SS time columns.

    Cell intensity encodes passenger count at that (station, time) intersection.
    Reveals when and where crowding occurs across the full recording."""
    if heatmap.empty:
        return _empty_figure("Run analysis to see the crowd heatmap")

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap.values,
            x=heatmap.columns.tolist(),
            y=heatmap.index.tolist(),
            colorscale=[
                [0.00, "#F0FDF4"],
                [0.25, "#86EFAC"],
                [0.55, "#22C55E"],
                [0.85, "#15803D"],
                [1.00, "#14532D"],
            ],
            colorbar=dict(
                title=dict(text="Passengers", font=dict(size=11, color="#475569", weight=700)),
                tickfont=dict(size=11, color="#64748B"),
                thickness=12,
                len=0.82,
                outlinewidth=0,
            ),
            xgap=1,
            ygap=1,
            hovertemplate="<b>%{y}</b><br>Time: %{x}<br>Passengers: %{z:,}<extra></extra>",
        )
    )
    fig.update_layout(
        **_BASE_LAYOUT,
        title="Crowd Heatmap — Passenger Volume by Station and Time",
        xaxis=dict(
            title="In-game time (HH:MM:SS)",
            tickangle=-45,
            showgrid=False,
            tickfont=dict(size=10),
            linecolor="rgba(148,163,184,.3)",
        ),
        yaxis=dict(
            title=None,
            showgrid=False,
            tickfont=dict(size=12),
            autorange="reversed",
        ),
    )
    return fig


def passenger_timeline_chart(time_series: pd.DataFrame, chart_type: str = "bar") -> go.Figure:
    """Passenger increases over in-game time, grouped by destination station.

    X = HH:MM:SS in-game clock, Y = total passenger count, color = destination station.
    chart_type: 'bar' (grouped) or 'line' (with markers)."""
    if time_series.empty:
        return _empty_figure("Run analysis to see the passenger flow timeline")

    common = dict(
        data_frame=time_series,
        x="time",
        y="passenger_increse",
        color="station_name",
        labels={
            "time": "In-game time",
            "passenger_increse": "Passengers",
            "station_name": "Destination",
        },
        title="Passenger Flow Timeline by Destination",
        color_discrete_sequence=CHART_PALETTE,
    )

    if chart_type == "line":
        fig = px.line(**common, markers=True)
        fig.update_traces(
            line_width=2.5,
            marker_size=7,
            hovertemplate="<b>%{fullData.name}</b><br>Time: %{x}<br>Passengers: %{y:,}<extra></extra>",
        )
    else:
        fig = px.bar(**common, barmode="group")
        fig.update_traces(
            hovertemplate="<b>%{fullData.name}</b><br>Time: %{x}<br>Passengers: %{y:,}<extra></extra>",
        )

    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis=dict(
            title="In-game time (HH:MM:SS)",
            tickangle=-35,
            showgrid=False,
            linecolor="#E2E8F0",
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title="Passenger increases",
            gridcolor="#F1F5F9",
            tickfont=dict(size=12),
            tickformat=",",
            rangemode="tozero",
        ),
        bargap=0.22,
        bargroupgap=0.06,
        legend=dict(**_LEGEND_BASE, title="Destination"),
    )
    return fig


def passenger_pct_stacked_bar(time_agg: pd.DataFrame) -> go.Figure:
    """Stacked 100% bar chart: % of each time bucket's passengers going to each station."""
    if time_agg.empty:
        return _empty_figure("Run analysis to see the passenger distribution chart")

    fig = px.bar(
        time_agg,
        x="time_bucket",
        y="pct_of_bucket",
        color="station_name",
        barmode="stack",
        labels={
            "time_bucket": "Time (in-game clock)",
            "pct_of_bucket": "Share of passengers (%)",
            "station_name": "Destination",
            "passengers": "Passengers",
        },
        title="Passenger Destination Share by Time Interval",
        hover_data={"passengers": True, "pct_of_bucket": ":.1f"},
        color_discrete_sequence=CHART_PALETTE,
    )
    fig.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>%{x}<br>%{customdata[1]:.1f}% · %{customdata[0]:,} pax<extra></extra>",
    )
    fig.update_layout(
        **_BASE_LAYOUT,
        yaxis=dict(
            range=[0, 100],
            ticksuffix="%",
            title="Share of passengers",
            gridcolor="#F1F5F9",
            tickfont=dict(size=12),
        ),
        xaxis=dict(
            title="Time (in-game clock)",
            tickangle=-30,
            showgrid=False,
            linecolor="#E2E8F0",
        ),
        bargap=0.18,
        legend=dict(**_LEGEND_BASE, title="Destination"),
    )
    return fig


def passenger_volume_bar(station_agg: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart: total passenger increases per station, labelled with % share."""
    if station_agg.empty:
        return _empty_figure("Run analysis to see the volume chart")

    fig = px.bar(
        station_agg.sort_values("total_passengers"),
        x="total_passengers",
        y="station_name",
        orientation="h",
        text="pct_of_total",
        color="total_passengers",
        color_continuous_scale=[[0, "#D1FAE5"], [0.5, "#16A34A"], [1, "#14532D"]],
        labels={
            "total_passengers": "Total Passenger Increases",
            "station_name": "Station",
        },
        title="Total Passenger Increases by Station",
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        textfont=dict(size=12, color="#334155"),
        hovertemplate="<b>%{y}</b><br>%{x:,} passengers · %{text:.1f}%<extra></extra>",
    )
    fig.update_layout(
        **_BASE_LAYOUT,
        coloraxis_showscale=False,
        yaxis=dict(title=None, showgrid=False, tickfont=dict(size=12)),
        xaxis=dict(title="Passenger increases", gridcolor="#F1F5F9"),
    )
    return fig


def station_share_pie(station_agg: pd.DataFrame) -> go.Figure:
    """Donut chart: each station's share of total passenger increases across the session."""
    if station_agg.empty:
        return _empty_figure("Run analysis to see the station share chart")

    fig = px.pie(
        station_agg,
        values="total_passengers",
        names="station_name",
        hole=0.42,
        title="Overall Station Share of Passenger Increases",
        color_discrete_sequence=CHART_PALETTE,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>%{value:,} passengers · %{percent}<extra></extra>",
    )
    fig.update_layout(
        **_BASE_LAYOUT,
        legend=dict(**_LEGEND_BASE, title="Station"),
    )
    return fig


def figure_to_bytes(fig: go.Figure, fmt: str = "png") -> bytes:
    """Renders a Plotly figure to image bytes.

    Requires kaleido >=1.0 and a Chrome binary. If Chrome is missing, run
    `plotly_get_chrome` from the active Python env."""
    try:
        return fig.to_image(format=fmt)
    except Exception as e:
        msg = str(e).lower()
        if "chrome" in msg:
            raise RuntimeError(
                "PNG export needs Chrome. Run `plotly_get_chrome` in the yolo_env to install it."
            ) from e
        if "kaleido" in msg or "orca" in msg:
            raise RuntimeError(
                "PNG export needs kaleido. Run `pip install 'kaleido>=1.0'` in the yolo_env."
            ) from e
        raise


# ── Internal helpers ───────────────────────────────────────────────────────────


def _empty_figure(message: str) -> go.Figure:
    """Returns a blank figure with a centred placeholder message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=14, color="#94A3B8", family="Inter, system-ui, sans-serif"),
    )
    fig.update_layout(
        **{**_BASE_LAYOUT, "margin": dict(t=40, b=20, l=20, r=20)},
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig
