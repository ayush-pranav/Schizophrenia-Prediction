from flask import Flask, render_template_string, request, redirect, url_for, session
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
import sqlite3
from functools import wraps
from flask import send_file

# Initialize Flask
server = Flask(__name__)
server.secret_key = 'your_secret_key_here_change_this'  # Change this to a random secret key

# Database Setup
def setup_database():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )
    """)
    cursor.execute("SELECT * FROM users")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", ("admin", "password"))
        conn.commit()
    conn.close()

setup_database()
@server.route('/background.jpg')
def serve_background():
    return send_file('background.jpg')
# Login HTML Template
LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Login Page</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: Arial, sans-serif;
            background: url('background.jpg') no-repeat center center fixed;
            background-size: cover;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .login-container {
            background: rgba(0, 0, 0, 0.8);
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            width: 400px;
        }
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 30px;
            font-size: 28px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            color: white;
            display: block;
            margin-bottom: 8px;
            font-size: 14px;
        }
        input[type="text"], input[type="password"] {
            width: 100%;
            padding: 12px;
            border: none;
            border-radius: 5px;
            font-size: 14px;
            background: rgba(255, 255, 255, 0.9);
        }
        input[type="text"]:focus, input[type="password"]:focus {
            outline: 2px solid #7b4fff;
        }
        button {
            width: 100%;
            padding: 14px;
            background: #7b4fff;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            margin-top: 10px;
            transition: background 0.3s;
        }
        button:hover {
            background: #6a3fee;
        }
        .error {
            background: #ff4444;
            color: white;
            padding: 12px;
            border-radius: 5px;
            margin-bottom: 20px;
            text-align: center;
        }
        .info {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
            font-size: 12px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <h1>🔐 Login</h1>
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
        <form method="POST" action="/login">
            <div class="form-group">
                <label>Username</label>
                <input type="text" name="username" required placeholder="Enter your username">
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" name="password" required placeholder="Enter your password">
            </div>
            <button type="submit">Login</button>
        </form>
        <div class="info">
            <strong>Test Credentials:</strong><br>
            Username: admin<br>
            Password: password
        </div>
    </div>
</body>
</html>
"""

# Flask Routes
@server.route('/')
def index():
    if 'username' in session:
        return redirect('/dashboard')
    return redirect('/login')

@server.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            session['username'] = username
            return redirect('/dashboard')
        else:
            return render_template_string(LOGIN_TEMPLATE, error="Invalid username or password")
    
    return render_template_string(LOGIN_TEMPLATE, error=None)

@server.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/login')

# Initialize Dash app
app = dash.Dash(__name__, server=server, url_base_pathname='/dashboard/')

# Load Data and Models
try:
    df = pd.read_csv('SchizophreniaSymptomnsData.csv')
    df.columns = df.columns.str.strip()
    
    model_data = joblib.load("model.pkl")
    model = model_data["model"]
    scaler = model_data["scaler"]
    le_gender = model_data["le_gender"]
    le_marital_status = model_data["le_marital_status"]
    le_schizophrenia = model_data["le_schizophrenia"]
except Exception as e:
    print(f"Error loading data/models: {e}")
    df = pd.DataFrame(columns=["Age", "Fatigue", "Schizophrenia", "Name", "Gender", "Slowing", "Pain", "Hygiene", "Movement"])
    model = None

# Prediction Function
def predict_schizophrenia(age, gender, marital_status, fatigue, slowing, pain, hygiene):
    try:
        input_data = np.array([[
            age,
            le_gender.transform([gender])[0],
            le_marital_status.transform([marital_status])[0],
            round(fatigue, 2),
            round(slowing, 2),
            round(pain, 2),
            round(hygiene, 2),
        ]])
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        predicted_stage = le_schizophrenia.inverse_transform(prediction)[0]
        return predicted_stage
    except Exception as e:
        return "Error in Prediction"

# Precautions Function
def get_precautions(level):
    precautions = {
        "Elevated Proneness": "Regular checkups, meditation, moderate physical activity.",
        "Very High Proneness": "Immediate medical attention, medication as prescribed, high supervision, 8-9 hours of sleep.",
        "High Proneness": "Monitor closely, engage in therapy sessions, ensure proper sleep (7-8 hours), avoid stress.",
        "Low Proneness": "Light physical activity, engage in mindfulness practices, adequate sleep (7-8 hours).",
        "Moderate Proneness": "Engage in group therapy, maintain a routine, avoid stress, 7-8 hours of sleep."
    }
    return precautions.get(level, "No precautions available.")

# Dash Layout - REPLACE YOUR EXISTING app.layout WITH THIS:
app.layout = html.Div([
    html.Div([
        html.H1("🧠 Schizophrenia Prediction Dashboard", 
                style={'textAlign': 'center', 'color': 'white', 'marginBottom': '10px', 
                       'textShadow': '2px 2px 4px rgba(0,0,0,0.8)'}),
        html.Div([
            html.Span(id='username-display', 
                     style={'marginRight': '20px', 'fontSize': '16px', 'color': 'white',
                            'fontWeight': 'bold', 'textShadow': '1px 1px 2px rgba(0,0,0,0.8)'}),
            html.A('Logout', href='/logout', style={
                'padding': '8px 16px',
                'background': '#e74c3c',
                'color': 'white',
                'textDecoration': 'none',
                'borderRadius': '5px',
                'fontSize': '14px'
            })
        ], style={'textAlign': 'right', 'marginBottom': '20px'})
    ]),
    
    dcc.Graph(id='scatter-plot', style={'marginBottom': '30px', 'background': 'rgba(255,255,255,0.95)', 
                                        'borderRadius': '10px', 'padding': '10px'}),
    
    html.Div([
        html.H2("Patient Information", style={'color': '#34495e', 'marginBottom': '20px'}),
        
        html.Div([
            html.Div([
                html.Label("Name:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                dcc.Input(id='name', type='text', placeholder='Enter Name', 
                         style={'width': '100%', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
            ], style={'marginBottom': '15px'}),
            
            html.Div([
                html.Label("Age:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                dcc.Input(id='age', type='number', placeholder='Enter Age',
                         style={'width': '100%', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
            ], style={'marginBottom': '15px'}),
            
            html.Div([
                html.Label("Gender:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                dcc.Dropdown(id='gender', 
                           options=[{'label': g, 'value': g} for g in ['Male', 'Female']], 
                           placeholder="Select Gender",
                           style={'width': '100%'}),
            ], style={'marginBottom': '15px'}),
            
            html.Div([
                html.Label("Marital Status:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                dcc.Dropdown(id='marital-status',
                           options=[{'label': m, 'value': m} for m in ['Single', 'Married', 'Widowed', 'Divorced']],
                           placeholder="Select Marital Status",
                           style={'width': '100%'}),
            ], style={'marginBottom': '15px'}),
            
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '4%'}),
        
        html.Div([
            html.Div([
                html.Label("Fatigue (0-10):", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                dcc.Input(id='fatigue', type='number', placeholder='Enter Fatigue Level',
                         style={'width': '100%', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
            ], style={'marginBottom': '15px'}),
            
            html.Div([
                html.Label("Slowing (0-10):", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                dcc.Input(id='slowing', type='number', placeholder='Enter Slowing Level',
                         style={'width': '100%', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
            ], style={'marginBottom': '15px'}),
            
            html.Div([
                html.Label("Pain (0-10):", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                dcc.Input(id='pain', type='number', placeholder='Enter Pain Level',
                         style={'width': '100%', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
            ], style={'marginBottom': '15px'}),
            
            html.Div([
                html.Label("Hygiene (0-10):", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                dcc.Input(id='hygiene', type='number', placeholder='Enter Hygiene Level',
                         style={'width': '100%', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
            ], style={'marginBottom': '15px'}),
            
            html.Div([
                html.Label("Movement (0-10):", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                dcc.Input(id='movement', type='number', placeholder='Enter Movement Level',
                         style={'width': '100%', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
            ], style={'marginBottom': '15px'}),
            
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
    ], style={'background': 'rgba(255, 255, 255, 0.95)', 'padding': '30px', 'borderRadius': '10px', 
              'marginBottom': '20px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.2)'}),
    
    html.Button('Submit Prediction', id='submit-button', n_clicks=0, 
                style={
                    'width': '100%',
                    'padding': '15px',
                    'background': '#7b4fff',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '5px',
                    'fontSize': '18px',
                    'fontWeight': 'bold',
                    'cursor': 'pointer',
                    'marginBottom': '20px',
                    'boxShadow': '0 4px 8px rgba(0,0,0,0.2)'
                }),
    
    html.Div(id='prediction-output', style={
        'padding': '20px',
        'background': 'rgba(232, 245, 233, 0.95)',
        'borderRadius': '5px',
        'marginBottom': '10px',
        'fontSize': '18px',
        'fontWeight': 'bold',
        'textAlign': 'center',
        'boxShadow': '0 4px 8px rgba(0,0,0,0.2)'
    }),
    
    html.Div(id='precautions-output', style={
        'padding': '20px',
        'background': 'rgba(255, 243, 224, 0.95)',
        'borderRadius': '5px',
        'fontSize': '16px',
        'boxShadow': '0 4px 8px rgba(0,0,0,0.2)'
    })
], style={
    'maxWidth': '1600px', 
    'margin': '0 auto', 
    'padding': '20px',
    'minHeight': '100vh',
    'background': 'url(/background.jpg) no-repeat center center fixed',
    'backgroundSize': 'cover'
})


# Callbacks
@app.callback(
    Output('username-display', 'children'),
    Input('submit-button', 'n_clicks')
)
def display_username(n_clicks):
    username = session.get('username', 'Guest')
    return f"👤 Welcome, {username}!"

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('submit-button', 'n_clicks')],
    [State('name', 'value'), State('age', 'value'), State('gender', 'value'),
     State('marital-status', 'value'), State('fatigue', 'value'), State('slowing', 'value'),
     State('pain', 'value'), State('hygiene', 'value'), State('movement', 'value')]
)
def update_graph(n_clicks, name, age, gender, marital_status, fatigue, slowing, pain, hygiene, movement):
    global df
    if n_clicks > 0 and all(v is not None for v in [name, age, gender, marital_status, fatigue, slowing, pain, hygiene]):
        predicted_stage = predict_schizophrenia(age, gender, marital_status, fatigue, slowing, pain, hygiene)
        new_data = pd.DataFrame([{
            "Name": name, "Age": age, "Gender": gender, "Marital_Status": marital_status,
            "Fatigue": fatigue, "Slowing": slowing, "Pain": pain, "Hygiene": hygiene,
            "Movement": movement, "Schizophrenia": predicted_stage
        }])
        df = pd.concat([df, new_data], ignore_index=True)
    
    fig = px.scatter(df, x="Age", y="Fatigue", color="Schizophrenia" if "Schizophrenia" in df.columns else None,
                     hover_data=["Name"] if "Name" in df.columns else None,
                     title="Patient Schizophrenia Levels",
                     color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=500)
    return fig

@app.callback(
    [Output('prediction-output', 'children'), Output('precautions-output', 'children')],
    [Input('submit-button', 'n_clicks')],
    [State('name', 'value'), State('age', 'value'), State('gender', 'value'),
     State('marital-status', 'value'), State('fatigue', 'value'), State('slowing', 'value'),
     State('pain', 'value'), State('hygiene', 'value'), State('movement', 'value')]
)
def predict_precautions(n_clicks, name, age, gender, marital_status, fatigue, slowing, pain, hygiene, movement):
    if n_clicks > 0 and all(v is not None for v in ([name, age, gender, marital_status, fatigue, slowing, pain, hygiene, movement])):
        predicted_stage = predict_schizophrenia(age, gender, marital_status, fatigue, slowing, pain, hygiene)
        precautions = get_precautions(predicted_stage)
        return f"🎯 Prediction for {name}: {predicted_stage}", f"⚕️ Precautions: {precautions}"
    return "", ""

# Run the app
if __name__ == "__main__":
    server.run(debug=True, host='0.0.0.0', port=8050)
