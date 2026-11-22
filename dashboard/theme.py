"""
FraudShield dashboard - shared theme, palette and chart helpers.

The colour palette is a validated categorical palette: the hues are assigned in
a fixed order (never cycled), the fraud/legitimate pair is a status-coloured
dichotomy, and every chart also ships readable labels or a table view so that
colour is never the only way to read a value.
"""

from __future__ import annotations

import plotly.graph_objects as go

# --- surfaces and ink -------------------------------------------------------
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8982"
GRID = "#e6e5e0"

# --- categorical slots (fixed order) ---------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

# --- status colours (reserved; always shipped with a label) -----------------
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_SERIOUS = "#ec835a"
STATUS_CRITICAL = "#d03b3b"

# fraud vs legitimate: a two-state dichotomy, not a series pair
COLOR_FRAUD = STATUS_CRITICAL
COLOR_LEGIT = "#2a78d6"

RISK_COLORS = {
    "High Risk": STATUS_CRITICAL,
    "Medium Risk": STATUS_SERIOUS,
    "Low Risk": STATUS_WARNING,
    "No Alert": INK_MUTED,
}

TYPE_COLORS = {
    "CASH_OUT": SERIES[0],
    "PAYMENT": SERIES[1],
    "CASH_IN": SERIES[2],
    "TRANSFER": SERIES[3],
    "DEBIT": SERIES[4],
}

MODEL_COLORS = {
    "LogisticRegression": SERIES[0],
    "RandomForestClassifier": SERIES[2],
}


def style(fig: go.Figure, height: int = 360, showlegend: bool = False) -> go.Figure:
    """Apply the shared chart chrome: recessive grid, quiet axes, no clutter."""
    fig.update_layout(
        height=height,
        showlegend=showlegend,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="-apple-system, Segoe UI, Roboto, sans-serif",
                  size=13, color=INK_SECONDARY),
        margin=dict(l=8, r=8, t=36, b=8),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID,
                        font=dict(color=INK, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, title_text=""),
        # Plotly renders the literal string "undefined" when a title dict is
        # supplied without any text, so carry over whatever title the chart set
        # (or an empty string) explicitly.
        title=dict(
            text=(fig.layout.title.text or ""),
            font=dict(size=14, color=INK),
            x=0,
            xanchor="left",
        ),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRID,
                     ticks="outside", tickcolor=GRID, title_font=dict(size=12))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     linecolor="rgba(0,0,0,0)", title_font=dict(size=12))
    return fig


CSS = f"""
<style>
  .block-container {{ padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1400px; }}
  #MainMenu, footer {{ visibility: hidden; }}

  .fs-header {{
      display: flex; align-items: baseline; gap: .75rem;
      border-bottom: 1px solid {GRID}; padding-bottom: .7rem; margin-bottom: 1.3rem;
  }}
  .fs-title {{ font-size: 1.65rem; font-weight: 700; color: {INK}; letter-spacing: -.02em; }}
  .fs-sub {{ font-size: .9rem; color: {INK_MUTED}; }}

  .fs-card {{
      background: {SURFACE}; border: 1px solid {GRID}; border-radius: 10px;
      padding: 1rem 1.1rem; height: 100%;
  }}
  .fs-card-label {{
      font-size: .74rem; font-weight: 600; letter-spacing: .07em;
      text-transform: uppercase; color: {INK_MUTED}; margin-bottom: .35rem;
  }}
  .fs-card-value {{ font-size: 1.6rem; font-weight: 700; color: {INK}; line-height: 1.15; }}
  .fs-card-note {{ font-size: .78rem; color: {INK_SECONDARY}; margin-top: .3rem; }}
  .fs-accent-fraud {{ border-left: 3px solid {STATUS_CRITICAL}; }}
  .fs-accent-ok {{ border-left: 3px solid {STATUS_GOOD}; }}
  .fs-accent-blue {{ border-left: 3px solid {COLOR_LEGIT}; }}
  .fs-accent-warn {{ border-left: 3px solid {STATUS_WARNING}; }}

  .fs-section {{
      font-size: 1.05rem; font-weight: 700; color: {INK};
      margin: 1.6rem 0 .6rem 0;
  }}
  .fs-note {{
      font-size: .82rem; color: {INK_SECONDARY}; background: #f3f5f8;
      border-left: 3px solid {COLOR_LEGIT}; border-radius: 6px;
      padding: .6rem .8rem; margin: .4rem 0 1rem 0;
  }}
  .fs-pill {{
      display: inline-block; font-size: .72rem; font-weight: 600;
      padding: .15rem .55rem; border-radius: 999px; margin-right: .35rem;
  }}
  div[data-testid="stMetricValue"] {{ font-size: 1.4rem; }}
</style>
"""


def card(label: str, value: str, note: str = "", accent: str = "blue") -> str:
    """One KPI card as HTML (values are always written out, never colour-only)."""
    return (
        f'<div class="fs-card fs-accent-{accent}">'
        f'<div class="fs-card-label">{label}</div>'
        f'<div class="fs-card-value">{value}</div>'
        f'<div class="fs-card-note">{note}</div>'
        f"</div>"
    )
