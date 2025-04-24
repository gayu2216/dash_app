import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Initialize the Dash app with dark theme
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.DARKLY],
                use_pages=True)

# Custom CSS for the app
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #000000;
                color: #3498db;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .dash-dropdown .Select-control {
                background-color: #111111;
                color: #3498db;
                border-color: #3498db;
            }
            .dash-dropdown .Select-menu-outer {
                background-color: #111111;
                color: #3498db;
            }
            .dash-dropdown .Select-value-label {
                color: #3498db !important;
            }
            .dash-dropdown .Select-placeholder {
                color: #3498db !important;
            }
            .card {
                background-color: #111111;
                border: 1px solid #3498db;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 0 15px rgba(52, 152, 219, 0.3);
            }
            .nav-link {
                color: #3498db !important;
            }
            .nav-link.active {
                background-color: #3498db !important;
                color: #000000 !important;
            }
            .btn-primary {
                background-color: #3498db;
                border-color: #3498db;
            }
            .btn-primary:hover {
                background-color: #2980b9;
                border-color: #2980b9;
            }
            h1, h2, h3, h4, h5 {
                color: #3498db;
            }
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner table {
                background-color: #111111;
                color: #3498db;
            }
            .dash-spinner * {
                background-color: #3498db !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Define the navbar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.NavItem(dbc.NavLink("Predictor", href="/predictor")),
        dbc.NavItem(dbc.NavLink("Data Explorer", href="/explorer")),
        dbc.NavItem(dbc.NavLink("About", href="/about")),
    ],
    brand="ChildHealth_AI",
    brand_href="/",
    color="dark",
    dark=True,
    className="mb-4",
)

# Define the layout for the app
app.layout = html.Div([
    navbar,
    dash.page_container
])

# Register pages
import pages.home
import pages.predictor
import pages.explorer
import pages.about

# Run the app
if __name__ == '__main__':
    app.run(debug=True)