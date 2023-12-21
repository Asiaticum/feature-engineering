from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def scatter_3d_by_px(
    df: pd.DataFrame,
    df_label: pd.DataFrame,
    color: Optional[str] = None,
    symbol: Optional[str] = None,
    color_discrete_sequence: List[str] = ["red", "green", "blue", "yellow", "black"],
) -> go.Figure:
    df[color] = df_label[color].astype("category") if color is not None else None
    df[symbol] = df_label[symbol].astype("category") if symbol is not None else None
    fig = px.scatter_3d(
        df,
        x=df.columns[0],
        y=df.columns[1],
        z=df.columns[2],
        color=color,
        symbol=symbol,
        color_discrete_sequence=color_discrete_sequence,
    )
    fig.update_layout(title=f"3D UMAP Dimension Reduction with {color} and {symbol}")
    # カラーバーを非表示
    fig.update(layout_coloraxis_showscale=False)

    return fig


def scatter_2d_by_px(
    df: pd.DataFrame,
    df_label: pd.DataFrame,
    color: Optional[str] = None,
    symbol: Optional[str] = None,
    color_discrete_sequence: List[str] = ["red", "green", "blue", "yellow", "black"],
) -> go.Figure:
    df[color] = df_label[color].astype("category") if color is not None else None
    df[symbol] = df_label[symbol].astype("category") if symbol is not None else None
    fig = px.scatter(
        df,
        x=df.columns[0],
        y=df.columns[1],
        color=color,
        symbol=symbol,
        color_discrete_sequence=color_discrete_sequence,
    )

    fig.update_layout(title=f"2D UMAP Dimension Reduction with {color} and {symbol}")
    # カラーバーを非表示
    fig.update(layout_coloraxis_showscale=False)

    return fig
