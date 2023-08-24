from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import plot
import numpy as np
from bball_data_handling import data, teams, team_dict, team_color_map, team_names_df

external_stylesheets = [
    {
        "href": (
            "https://fonts.googleapis.com/css2?"
            "family=Lato:wght@400;700&display=swap"
        ),
        "rel": "stylesheet",
    },
]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Baseball Analytics: Understand Your Team!"

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.P(children="⚾", className="header-emoji"),
                html.H1(
                    children="Baseball Analytics", className="header-title"
                ),
                html.P(
                    children=(
                        "EDA for baseball teams"
                    ),
                    className="header-description",
                ),
            ],
            className="header",
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(children="All Teams", className="menu-title"),

                        dcc.Checklist(
                          id="my-checklist",
                          options=team_dict,
                          value=["NYA"],
                          className='my_box_container',
                          labelStyle={"display": "inline-block"},
                        )
                    ]
                ),
                html.Div(
                    children=[
                        html.Div(
                            children="Date Range", className="menu-title"
                        ),
                        dcc.DatePickerRange(
                            id="date-range",
                            min_date_allowed=data["Date"].min().date(),
                            max_date_allowed=data["Date"].max().date(),
                            start_date=data["Date"].min().date(),
                            end_date=data["Date"].max().date(),
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Div(
                            children="Through N games", className="menu-title"
                        ),
                        dcc.Input(
                            id='range',
                            type='number',
                            min=1,
                            max=162,
                            step=1
                        ),
                    ]
                ),
            ],
            className="menu",
        ),
        html.Div(
            children=[
                html.Div(
                    children=dcc.Graph(
                        id="win-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="diff-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="best-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="worst-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                )
            ],
            className="wrapper",
        ),
    ]
)



@app.callback(
    Output("win-chart", "figure"),
    Output("diff-chart", "figure"),
    Output("best-chart","figure"),
    Output("worst-chart", "figure"),
    Input("my-checklist","value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("range","value")
)
def update_charts(options, start_date, end_date, num_games):
    filtered_data = data[(data['Team'].isin(options)) & (data['Date'] >= start_date) & (data['Date'] <= end_date)]
    filtered_data['Game No.'] = list(pd.concat([pd.DataFrame(filtered_data[filtered_data["Team"] == team][['Team','Differential']].reset_index(drop=True).groupby(by="Team").cumsum().reset_index()['index']) for team in options])['index'])
    filtered_data['Game No.'] = filtered_data['Game No.'] + 1
    filtered_data["Year"] = filtered_data['Date'].dt.year
    filtered_data["Wins"] = filtered_data[["Team","Result"]].groupby(by=["Team"]).cumsum()
    filtered_data["Diff_Cum"] = filtered_data[["Team","Differential"]].groupby(by=["Team"]).cumsum()
    win_chart_figure = px.line(filtered_data, x="Game No.", y="Wins", color="Team Name",
                               title='Wins above or below 0.500', color_discrete_map=team_color_map)
    diff_chart_figure = px.line(filtered_data, x="Game No.", y="Diff_Cum", color="Team Name",
                                title="Team's Run Differential", color_discrete_map=team_color_map)
    filtered_data = filtered_data.drop(columns=['Wins', 'Game No.'])
    filtered_data['Game No.'] = list(pd.concat([pd.DataFrame(
        filtered_data[(filtered_data["Team"] == team) & (filtered_data["Year"] == year)][
            ['Team', 'Year', 'Differential']].reset_index(drop=True).groupby(
            by=["Team", "Year"]).cumsum().reset_index()['index']) for team in options for year in range(1990, 2023)])['index'])
    filtered_data2 = filtered_data[filtered_data['Game No.'] <= num_games]
    filtered_data2["Wins"] = filtered_data2[["Team", "Year", "Result"]].groupby(by=["Team", "Year"]).cumsum()
    filtered_data3 = filtered_data2[filtered_data2["Game No."] == num_games]
    max_wins_df = filtered_data3[["Team", "Wins"]].groupby("Team").max().reset_index()  # .drop_duplicates()

    semi_fin_df = pd.merge(filtered_data3[["Team", "Year", "Wins"]], max_wins_df, on=["Team", "Wins"]).groupby(
        ["Team", "Wins"]).max().reset_index()[["Team", "Year"]]
    fin_df = pd.merge(semi_fin_df, filtered_data2, how='left', on=["Team", "Year"])
    fin_df["Team Name"] = fin_df['Year'].astype(str) + ' ' + fin_df['Team'].astype(str)
    best_chart_figure = px.line(fin_df, x="Game No.", y="Wins", color="Team Name", color_discrete_map=team_color_map)

    min_wins_df = filtered_data3[["Team", "Wins"]].groupby("Team").min().reset_index()  # .drop_duplicates()
    semi_fin_df = pd.merge(filtered_data3[["Team", "Year", "Wins"]], min_wins_df, on=["Team", "Wins"]).groupby(
        ["Team", "Wins"]).max().reset_index()[["Team", "Year"]]
    fin_df = pd.merge(semi_fin_df, filtered_data2, how='left', on=["Team", "Year"])
    fin_df["Team Name"] = fin_df['Year'].astype(str) + ' ' + fin_df['Team'].astype(str)
    worst_chart_figure = px.line(fin_df, x="Game No.", y="Wins", color="Team Name", color_discrete_map=team_color_map)
    return win_chart_figure, diff_chart_figure, best_chart_figure, worst_chart_figure


if __name__ == "__main__":
    app.run_server(debug=True)