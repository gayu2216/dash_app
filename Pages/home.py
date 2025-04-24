import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/')

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("Health Conditions Prediction Dashboard", 
                        className="text-center mb-4", 
                        style={"font-weight": "bold", "font-size": "3rem"}),
                html.H3("Analyze and Predict Health Conditions Using Machine Learning", 
                        className="text-center mb-4"),
                html.Div([
                    html.P("This dashboard allows you to explore health condition data and predict various health outcomes based on demographic, environmental, and health factors."),
                    html.P("Use the navigation bar to access different sections of the application:"),
                    html.Ul([
                        html.Li(["", html.B("Predictor"), ": Select values to predict risk of different health conditions"]),
                        html.Li(["", html.B("Data Explorer"), ": Explore relationships between variables in our dataset"]),
                        html.Li(["", html.B("About"), ": Learn more about this project and how the predictions work"]),
                    ]),
                ], className="text-left"),
                html.Div([
                    dbc.Button("Go to Predictor", color="primary", size="lg", href="/predictor", className="me-3"),
                    dbc.Button("Explore Data", color="info", size="lg", href="/explorer"),
                ], className="d-flex justify-content-center mt-5"),
            ], className="card p-5"),
        ], width={"size": 10, "offset": 1}),
    ], className="my-5"),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("Key Features", className="mb-3"),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H5("Health Prediction", className="mb-2"),
                                html.P("Predict likelihood of various health conditions using our trained models.")
                            ], className="card p-3 h-100")
                        ], width=4),
                        dbc.Col([
                            html.Div([
                                html.H5("Data Visualization", className="mb-2"),
                                html.P("Explore relationships between health factors and conditions with interactive charts.")
                            ], className="card p-3 h-100")
                        ], width=4),
                        dbc.Col([
                            html.Div([
                                html.H5("User-Friendly Interface", className="mb-2"),
                                html.P("Simple selection boxes to input data and get instant predictions and analyses.")
                            ], className="card p-3 h-100")
                        ], width=4),
                    ]),
                ])
            ], className="card p-4"),
        ], width={"size": 10, "offset": 1}),
    ], className="mb-5"),
])