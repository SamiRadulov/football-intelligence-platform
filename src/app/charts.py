"""Plotly chart builders for the app.

Charts show **percentiles** rather than z-scores wherever a user is reading a
profile: "84th percentile among centre-backs" is immediately meaningful in a way
that "z = +1.02" is not. Raw values are always available alongside.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Colour-blind safe qualitative palette, used consistently across the app.
SERIES_COLOURS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]
GRID_COLOUR = "rgba(128,128,128,0.25)"


def _base_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor=GRID_COLOUR, zerolinecolor=GRID_COLOUR)
    fig.update_yaxes(gridcolor=GRID_COLOUR, zerolinecolor=GRID_COLOUR)
    return fig


def radar(labels: list[str], series: dict[str, list[float]], title: str = "") -> go.Figure:
    """Percentile radar comparing one or two profiles (values 0-100)."""
    fig = go.Figure()
    for i, (name, values) in enumerate(series.items()):
        colour = SERIES_COLOURS[i % len(SERIES_COLOURS)]
        fig.add_trace(
            go.Scatterpolar(
                r=values + values[:1],          # close the loop
                theta=labels + labels[:1],
                name=name,
                fill="toself",
                line=dict(color=colour, width=2),
                opacity=0.55,
                hovertemplate="%{theta}<br>%{r:.0f}th percentile<extra>" + name + "</extra>",
            )
        )
    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=GRID_COLOUR),
            angularaxis=dict(gridcolor=GRID_COLOUR),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return _base_layout(fig, height=520)


def percentile_bars(
    frame: pd.DataFrame, label_column: str, value_column: str,
    colour_column: str | None = None, title: str = "",
) -> go.Figure:
    """Horizontal percentile bars (0-100), highest at the top."""
    ordered = frame.sort_values(value_column)
    fig = px.bar(
        ordered, x=value_column, y=label_column, orientation="h",
        color=colour_column, color_discrete_sequence=SERIES_COLOURS,
        range_x=[0, 100], title=title,
    )
    fig.add_vline(x=50, line_dash="dot", line_color=GRID_COLOUR)
    fig.update_traces(hovertemplate="%{y}: %{x:.0f}th percentile<extra></extra>")
    fig.update_layout(yaxis_title=None, xaxis_title="Percentile within group")
    return _base_layout(fig, height=max(320, 22 * len(ordered)))


def grouped_bars(
    frame: pd.DataFrame, label_column: str, value_columns: list[str], title: str = ""
) -> go.Figure:
    """Side-by-side comparison bars for two players/teams."""
    melted = frame.melt(id_vars=label_column, value_vars=value_columns,
                        var_name="who", value_name="percentile")
    fig = px.bar(
        melted, x="percentile", y=label_column, color="who", orientation="h",
        barmode="group", color_discrete_sequence=SERIES_COLOURS,
        range_x=[0, 100], title=title,
    )
    fig.add_vline(x=50, line_dash="dot", line_color=GRID_COLOUR)
    fig.update_traces(hovertemplate="%{y}<br>%{x:.0f}th percentile<extra>%{fullData.name}</extra>")
    fig.update_layout(yaxis_title=None, xaxis_title="Percentile within role")
    return _base_layout(fig, height=max(360, 30 * frame[label_column].nunique()))


def style_scatter(
    frame: pd.DataFrame, x: str, y: str, label_column: str, colour_column: str,
    highlight: str | None = None, x_title: str = "", y_title: str = "",
) -> go.Figure:
    """PCA style map: one point per team, coloured by cluster, labelled."""
    fig = px.scatter(
        frame, x=x, y=y, color=colour_column, text=label_column,
        color_discrete_sequence=SERIES_COLOURS,
        hover_data={x: ":.2f", y: ":.2f", colour_column: True, label_column: False},
    )
    fig.update_traces(marker=dict(size=13, line=dict(width=1, color="rgba(0,0,0,0.35)")),
                      textposition="top center", textfont=dict(size=11))
    if highlight is not None and highlight in frame[label_column].values:
        point = frame[frame[label_column] == highlight].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[point[x]], y=[point[y]], mode="markers", name=highlight,
                marker=dict(size=24, color="rgba(0,0,0,0)",
                            line=dict(width=3, color="#E45756")),
                hoverinfo="skip", showlegend=False,
            )
        )
    fig.add_hline(y=0, line_dash="dot", line_color=GRID_COLOUR)
    fig.add_vline(x=0, line_dash="dot", line_color=GRID_COLOUR)
    fig.update_layout(xaxis_title=x_title or x, yaxis_title=y_title or y)
    return _base_layout(fig, height=620)


def variability_bars(frame: pd.DataFrame, label_column: str, value_column: str,
                     title: str = "") -> go.Figure:
    """Match-to-match standard deviation, as a rigid-vs-adaptable read."""
    ordered = frame.sort_values(value_column)
    fig = px.bar(ordered, x=value_column, y=label_column, orientation="h",
                 color_discrete_sequence=[SERIES_COLOURS[2]], title=title)
    fig.update_traces(hovertemplate="%{y}: sd %{x:.3f}<extra></extra>")
    fig.update_layout(yaxis_title=None, xaxis_title="Match-to-match standard deviation")
    return _base_layout(fig, height=max(320, 22 * len(ordered)))
