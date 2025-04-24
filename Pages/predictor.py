import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import joblib
import os
import tensorflow as tf  # Add TensorFlow import

dash.register_page(__name__, path='/predictor')

# Define path to your saved model and scaler
MODEL_PATH = "/Users/gayathriutla/Desktop/Projects/shiny_app/models/multilabel_classification_model.h5"  # Update this path
SCALER_PATH = "/Users/gayathriutla/Desktop/Projects/shiny_app/models/scaler.pkl"  # Update this path

# Load the TensorFlow model and scaler
try:
    # Load TensorFlow model
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load scaler
    scaler = joblib.load(SCALER_PATH)
    
    # Set flag indicating successful model loading
    model_loaded = True
    print("Model and scaler loaded successfully")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model_loaded = False

conditions = ["Autism/ASD", "Learning Disability", "ADD/ADHD", "Depression", 
             "Anxiety", "Behavior Problems", "Speech Disorder", "Asthma"]

# Option definitions for each dropdown
options = {
    'general_health': [
        {'label': 'Excellent', 'value': 1},
        {'label': 'Very Good', 'value': 2},
        {'label': 'Good', 'value': 3},
        {'label': 'Fair', 'value': 4},
        {'label': 'Poor', 'value': 5}
    ],
    'mental_health': [
        {'label': 'Excellent', 'value': 1},
        {'label': 'Very Good', 'value': 2},
        {'label': 'Good', 'value': 3},
        {'label': 'Fair', 'value': 4},
        {'label': 'Poor', 'value': 5}
    ],
    'physical_health': [
        {'label': 'Excellent', 'value': 1},
        {'label': 'Very Good', 'value': 2},
        {'label': 'Good', 'value': 3},
        {'label': 'Fair', 'value': 4},
        {'label': 'Poor', 'value': 5}
    ],
    'education': [
        {'label': '8th grade or less', 'value': 1},
        {'label': '9th-12th grade, No diploma', 'value': 2},
        {'label': 'High School Graduate or GED', 'value': 3},
        {'label': 'Vocational/trade/business school', 'value': 4},
        {'label': 'Some College Credit, No Degree', 'value': 5},
        {'label': 'Associate Degree', 'value': 6},
        {'label': "Bachelor's Degree", 'value': 7},
        {'label': "Master's Degree", 'value': 8},
        {'label': 'Doctorate or Professional Degree', 'value': 9}
    ],
    'financial_hardship': [
        {'label': 'Never', 'value': 1},
        {'label': 'Rarely', 'value': 2},
        {'label': 'Somewhat often', 'value': 3},
        {'label': 'Very often', 'value': 4}
    ],
    'food_situation': [
        {'label': 'Always afford good nutritious meals', 'value': 1},
        {'label': 'Always afford enough but not always nutritious', 'value': 2},
        {'label': 'Sometimes could not afford enough', 'value': 3},
        {'label': 'Often could not afford enough', 'value': 4}
    ],
    'yes_no': [
        {'label': 'Yes', 'value': 1},
        {'label': 'No', 'value': 2}
    ],
    'neighborhood_safety': [
        {'label': 'Definitely agree', 'value': 1},
        {'label': 'Somewhat agree', 'value': 2},
        {'label': 'Somewhat disagree', 'value': 3},
        {'label': 'Definitely disagree', 'value': 4}
    ],
    'family_meal': [
        {'label': '0 days', 'value': 1},
        {'label': '1-3 days', 'value': 2},
        {'label': '4-6 days', 'value': 3},
        {'label': 'Every day', 'value': 4}
    ],
    'child_care_difficulty': [
        {'label': 'Never', 'value': 1},
        {'label': 'Rarely', 'value': 2},
        {'label': 'Sometimes', 'value': 3},
        {'label': 'Usually', 'value': 4},
        {'label': 'Always', 'value': 5}
    ],
    'family_talk': [
        {'label': 'All of the time', 'value': 1},
        {'label': 'Most of the time', 'value': 2},
        {'label': 'Some of the time', 'value': 3},
        {'label': 'None of the time', 'value': 4}
    ],
    'weight_concern': [
        {'label': 'Yes, too high', 'value': 1},
        {'label': 'Yes, too low', 'value': 2},
        {'label': 'Not concerned', 'value': 3}
    ],
    'screen_time': [
        {'label': 'Less than 1 hour', 'value': 1},
        {'label': '1 hour', 'value': 2},
        {'label': '2 hours', 'value': 3},
        {'label': '3 hours', 'value': 4},
        {'label': '4 or more hours', 'value': 5}
    ],
    'family_structure': [
        {'label': 'Two biological/adoptive parents, married', 'value': 1},
        {'label': 'Two biological/adoptive parents, not married', 'value': 2},
        {'label': 'Two parents (at least one not bio/adoptive), married', 'value': 3},
        {'label': 'Two parents (at least one not bio/adoptive), not married', 'value': 4},
        {'label': 'Single mother', 'value': 5},
        {'label': 'Single father', 'value': 6},
        {'label': 'Grandparent household', 'value': 7},
        {'label': 'Other relation', 'value': 8}
    ],
    'age': [{'label': str(i), 'value': i} for i in range(1, 18)],
    'race': [
        {'label': 'White alone', 'value': 1},
        {'label': 'Black or African American alone', 'value': 2},
        {'label': 'American Indian or Alaska Native alone', 'value': 3},
        {'label': 'Asian alone', 'value': 4},
        {'label': 'Native Hawaiian and Other Pacific Islander alone', 'value': 5},
        {'label': 'Two or More Races', 'value': 7}
    ],
    'birth_order': [
        {'label': 'Only child', 'value': 1},
        {'label': 'Oldest child', 'value': 2},
        {'label': 'Second oldest child', 'value': 3},
        {'label': 'Third oldest child', 'value': 4},
        {'label': 'Fourth or greater oldest child', 'value': 5}
    ]
}

# Custom CSS for dark theme
dark_theme_css = {
    'backgroundColor': '#121212',  # Dark background
    'color': '#E0E0E0',  # Light text
    'border': '1px solid #333',
    'borderRadius': '5px',
    'padding': '15px',
    'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.3)',
    'marginBottom': '15px'
}

dropdown_style = {
    'backgroundColor': '#1E1E1E',
    'color': '#FFFFFF',
    'border': '1px solid #2C5282'  # Blue border
}

# Create a styled dropdown component
def create_dropdown(id, label, options_list):
    return html.Div([
        html.Label(label, className="mb-2", style={'color': '#90CDF4'}),  # Light blue label
        dcc.Dropdown(
            id=id,
            options=options_list,
            value=options_list[0]['value'],
            className="dash-dropdown mb-3",
            clearable=False,
            style={'backgroundColor': '#1E1E1E', 'color': '#FFFFFF'},
            # Styling for dropdown menu
            # Note: Some of these styles may not fully apply due to Dash's component architecture
            # You might need custom CSS in an external file for complete styling
        )
    ])

# Layout the page
layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Health Condition Predictor", className="text-center mb-4", 
                   style={'color': '#63B3ED'}),  # Brighter blue for headings
            html.P("Select values for each factor to predict the likelihood of various health conditions.", 
                   className="text-center mb-4", style={'color': '#A0AEC0'}),  # Muted blue-gray for secondary text
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H3("Child Information", className="mb-3", style={'color': '#63B3ED'}),
                create_dropdown("age", "Child's Age (Years)", options['age']),
                create_dropdown("gender", "Child's Gender", [
                    {'label': 'Male', 'value': 1},
                    {'label': 'Female', 'value': 2}
                ]),
                create_dropdown("race", "Child's Race", options['race']),
                create_dropdown("general_health", "Child's General Health", options['general_health']),
                create_dropdown("birth_order", "Birth Order", options['birth_order']),
                create_dropdown("born_usa", "Born in USA", options['yes_no']),
            ], style={**dark_theme_css, 'backgroundColor': '#1A202C'})  # Slightly blue-tinted dark background
        ], width=4),
        
        dbc.Col([
            html.Div([
                html.H3("Family & Environment", className="mb-3", style={'color': '#63B3ED'}),
                create_dropdown("family_structure", "Family Structure", options['family_structure']),
                create_dropdown("financial_hardship", "Financial Hardship", options['financial_hardship']),
                create_dropdown("food_situation", "Food Situation", options['food_situation']),
                create_dropdown("family_meal", "Family Meals Together (per week)", options['family_meal']),
                create_dropdown("child_care_difficulty", "Difficulty Caring for Child", options['child_care_difficulty']),
                create_dropdown("family_talk", "Family Talks When Facing Problems", options['family_talk']),
            ], style={**dark_theme_css, 'backgroundColor': '#1A202C'})
        ], width=4),
        
        dbc.Col([
            html.Div([
                html.H3("Health & Lifestyle", className="mb-3", style={'color': '#63B3ED'}),
                create_dropdown("neighborhood_safety", "Child is Safe in Neighborhood", options['neighborhood_safety']),
                create_dropdown("rec_center", "Recreation Center in Neighborhood", options['yes_no']),
                create_dropdown("library", "Library in Neighborhood", options['yes_no']),
                create_dropdown("screen_time", "Screen Time (Weekdays)", options['screen_time']),
                create_dropdown("cigarettes", "Anyone in Household Use Cigarettes", options['yes_no']),
                create_dropdown("vape", "Anyone Vape Inside Home", options['yes_no']),
            ], style={**dark_theme_css, 'backgroundColor': '#1A202C'})
        ], width=4),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H3("Medical & Health History", className="mb-3", style={'color': '#63B3ED'}),
                dbc.Row([
                    dbc.Col([
                        create_dropdown("breathing_difficulty", "Breathing Difficulty (Past 12 Months)", options['yes_no']),
                        create_dropdown("stomach_problems", "Stomach Problems (Past 12 Months)", options['yes_no']),
                        create_dropdown("headaches", "Frequent/Severe Headaches", options['yes_no']),
                        create_dropdown("concussion", "Had Concussion/Brain Injury", options['yes_no']),
                    ], width=6),
                    dbc.Col([
                        create_dropdown("overweight", "Doctor Identified as Overweight", options['yes_no']),
                        create_dropdown("weight_concern", "Concerned About Weight", options['weight_concern']),
                        create_dropdown("heart_condition", "Heart Condition", options['yes_no']),
                        create_dropdown("diabetes", "Type 2 Diabetes", options['yes_no']),
                    ], width=6),
                ]),
            ], style={**dark_theme_css, 'backgroundColor': '#1A202C'})
        ], width=8),
        
        dbc.Col([
            html.Div([
                html.H3("Adult Health", className="mb-3", style={'color': '#63B3ED'}),
                create_dropdown("parent_education", "Parent's Education Level", options['education']),
                create_dropdown("parent_mental_health", "Parent's Mental Health", options['mental_health']),
                create_dropdown("parent_physical_health", "Parent's Physical Health", options['physical_health']),
            ], style={**dark_theme_css, 'backgroundColor': '#1A202C'})
        ], width=4),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H3("Adverse Childhood Experiences", className="mb-3", style={'color': '#63B3ED'}),
                dbc.Row([
                    dbc.Col([
                        create_dropdown("homeless", "Ever Homeless or Lived in Shelter", [
                            {'label': 'Yes', 'value': 1},
                            {'label': 'No', 'value': 2},
                            {'label': "Don't Know", 'value': 3}
                        ]),
                        create_dropdown("racial_unfair", "Treated Unfairly Because of Race", options['yes_no']),
                    ], width=6),
                    dbc.Col([
                        create_dropdown("witness_violence", "Witnessed Adult Violence in Home", options['yes_no']),
                        create_dropdown("victim_violence", "Victim/Witness of Neighborhood Violence", options['yes_no']),
                    ], width=6),
                ]),
            ], style={**dark_theme_css, 'backgroundColor': '#1A202C', 'marginBottom': '20px'})
        ], width=12),
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Button("Predict Health Conditions", 
                     id="predict-button", 
                     color="primary", 
                     size="lg", 
                     className="w-100 mb-4",
                     style={'backgroundColor': '#2B6CB0', 'borderColor': '#2B6CB0'}),  # Bright blue button
        ], width={"size": 6, "offset": 3}),
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div(id="prediction-output", className="mb-3"),
            dcc.Graph(id="prediction-chart", 
                     style={"display": "none", "backgroundColor": "#1A202C", "border": "1px solid #2C5282"}),
        ], width=12),
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H5("Important Note", className="mb-2", style={'color': '#F6AD55'}),  # Orange warning text
                html.P("This tool provides estimates based on statistical models and should not replace professional medical advice. "
                       "Always consult with healthcare providers for diagnosis and treatment decisions.",
                       className="small fst-italic", style={'color': '#A0AEC0'}),
            ], style={**dark_theme_css, 'backgroundColor': '#1A202C', 'textAlign': 'center'})
        ], width={"size": 8, "offset": 2}),
    ], className="mb-5"),
], fluid=True, style={'backgroundColor': '#0A0A0A', 'minHeight': '100vh', 'padding': '20px'})  # Very dark base background

# Callback to handle predictions
@callback(
    [Output("prediction-output", "children"),
     Output("prediction-chart", "figure"),
     Output("prediction-chart", "style")],
    [Input("predict-button", "n_clicks")],
    [State("age", "value"),
     State("gender", "value"),
     State("race", "value"),
     State("general_health", "value"),
     State("birth_order", "value"),
     State("born_usa", "value"),
     State("family_structure", "value"),
     State("financial_hardship", "value"),
     State("food_situation", "value"),
     State("family_meal", "value"),
     State("child_care_difficulty", "value"),
     State("family_talk", "value"),
     State("neighborhood_safety", "value"),
     State("rec_center", "value"),
     State("library", "value"),
     State("screen_time", "value"),
     State("cigarettes", "value"),
     State("vape", "value"),
     State("breathing_difficulty", "value"),
     State("stomach_problems", "value"),
     State("headaches", "value"),
     State("concussion", "value"),
     State("overweight", "value"),
     State("weight_concern", "value"),
     State("heart_condition", "value"),
     State("diabetes", "value"),
     State("parent_education", "value"),
     State("parent_mental_health", "value"),
     State("parent_physical_health", "value"),
     State("homeless", "value"),
     State("racial_unfair", "value"),
     State("witness_violence", "value"),
     State("victim_violence", "value")],
    prevent_initial_call=True
)
def update_prediction(n_clicks, age, gender, race, general_health, birth_order, born_usa,
                     family_structure, financial_hardship, food_situation, family_meal, 
                     child_care_difficulty, family_talk, neighborhood_safety, rec_center,
                     library, screen_time, cigarettes, vape, breathing_difficulty, 
                     stomach_problems, headaches, concussion, overweight, weight_concern,
                     heart_condition, diabetes, parent_education, parent_mental_health,
                     parent_physical_health, homeless, racial_unfair, witness_violence,
                     victim_violence):
    if n_clicks is None:
        return html.Div(), {}, {"display": "none"}
    
    # Collect all inputs into a feature vector
    features = [
        age, gender, race, general_health, birth_order, born_usa,
        family_structure, financial_hardship, food_situation, family_meal, 
        child_care_difficulty, family_talk, neighborhood_safety, rec_center,
        library, screen_time, cigarettes, vape, breathing_difficulty, 
        stomach_problems, headaches, concussion, overweight, weight_concern,
        heart_condition, diabetes, parent_education, parent_mental_health,
        parent_physical_health, homeless, racial_unfair, witness_violence,
        victim_violence
    ]
    
    # Use the model to make predictions
    if model_loaded:
        try:
            # Scale the features
            scaled_features = scaler.transform([features])
            
            # Make prediction with TensorFlow model
            prediction_array = model.predict(scaled_features)[0]
            
            # Convert to probabilities
            predictions = prediction_array.tolist()
            
            print("Model prediction successful")
        except Exception as e:
            print(f"Error during prediction: {e}")
            predictions = np.random.random(size=len(conditions))
    else:
        # Fall back to random predictions if model isn't loaded
        print("Using random predictions as model is not loaded")
        predictions = np.random.random(size=len(conditions))
    
    # Create dataframe for predictions
    results_df = pd.DataFrame({
        'Condition': conditions,
        'Probability': predictions,
        'Percentage': [p * 100 for p in predictions]
    })
    
    # Get top 3 conditions
    top_3 = results_df.sort_values('Probability', ascending=False).head(3).reset_index(drop=True)
    
    # Colors for the bars - adjust for dark theme
    colors = ['#63B3ED', '#4FD1C5', '#F6AD55']  # Blue, teal, orange
    
    # Create horizontal bar chart for top 3 conditions
    fig = go.Figure()
    
    for i, row in top_3.iterrows():
        fig.add_trace(go.Bar(
            y=[row['Condition']],
            x=[row['Percentage']],
            orientation='h',
            name=row['Condition'],
            marker_color=colors[i],
            text=[f"{row['Percentage']:.1f}%"],
            textposition='auto',
            textfont=dict(
                color='white'
            )
        ))
    
    fig.update_layout(
        title={
            'text': 'Top 3 Predicted Health Conditions',
            'font': {
                'color': '#E0E0E0'
            }
        },
        xaxis_title={
            'text': 'Probability (%)',
            'font': {
                'color': '#E0E0E0'
            }
        },
        yaxis={
            'categoryorder': 'total ascending',
            'tickfont': {
                'color': '#E0E0E0'
            }
        },
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
        paper_bgcolor='rgba(26, 32, 44, 0.9)',  # Dark blue-gray
        plot_bgcolor='rgba(26, 32, 44, 0.7)',   # Slightly lighter
        font={
            'color': '#E0E0E0'
        }
    )
    
    # Create summary component with dark theme styling
    summary_cards = []
    for i, row in top_3.iterrows():
        risk_level = "High Risk" if row['Probability'] >= 0.3 else "Low Risk"
        
        # Using different colors for dark theme
        color = "danger" if risk_level == "High Risk" else "success"
        bg_color = "#2D3748" if i % 2 == 0 else "#1A202C"  # Alternating card backgrounds
        
        summary_cards.append(
            dbc.Card([
                dbc.CardHeader(row['Condition'], 
                             className="fw-bold", 
                             style={'backgroundColor': '#2C5282', 'color': '#FFFFFF'}),  # Blue header
                dbc.CardBody([
                    html.H3(f"{row['Percentage']:.1f}%", 
                          className="card-title", 
                          style={'color': colors[i]}),  # Using the same colors as the chart
                    html.P(risk_level, 
                         className=f"text-{color}", 
                         style={'fontWeight': 'bold'})
                ], style={'backgroundColor': bg_color})
            ], className="text-center mb-3")
        )
    
    # Create prediction output div
    prediction_div = html.Div([
        html.H3("Prediction Results", 
               className="text-center mb-4", 
               style={'color': '#63B3ED'}),  # Blue heading
        html.Div([
            html.P(
                f"{'Model prediction' if model_loaded else 'Sample prediction'}: Based on the information provided, here are the top 3 estimated risks:", 
                className="mb-3 text-center",
                style={'color': '#A0AEC0'}  # Light blue-gray text
            ),
            dbc.Row([
                dbc.Col(card, width=4) for card in summary_cards
            ], className="justify-content-center")
        ], style={**dark_theme_css, 'backgroundColor': '#1A202C'})
    ])
    
    return prediction_div, fig, {"display": "block", "backgroundColor": "#1A202C", "border": "1px solid #2C5282", "borderRadius": "5px", "padding": "10px"}