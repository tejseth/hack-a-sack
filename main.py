import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
from xgboost import Booster, DMatrix
import plotly.graph_objs as go
import matplotlib as plt
import pandas as pd
import dash_table as dt
import plotly.express as px


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Hack-a-Sack'
server = app.server

mod = Booster({'nthread': 8})
mod.load_model('xgb_sack')

app.layout = html.Div([
    dbc.Row([html.H3(children='Hack-a-Sack')]),
    dbc.Row([
        dbc.Col(html.Label(children='Down'), width={"order": "first"}),
        dcc.Dropdown(['1', '2', '3', '4'], '3', id='down'),
        dbc.Col(html.Label(children='Offensive Personnel (RB-TE)'), width={"order": "first"}),
        dcc.Dropdown(['11', '12', '21', '13', '10', '22', '01', '20', '11*', '02', '12*'], '11', id='o_dropdown'),
        dbc.Col(html.Label(children='Defensive Formation (DL-LB-DB)'), width={"order": "first"}),    
        dcc.Dropdown(['4-2-5', '2-4-5', '3-3-5', '2-3-6', '4-3-4', '3-4-4', '4-1-6', '3-2-6', '1-4-6', '1-5-5'], '4-2-5', id='d_dropdown'),
        dbc.Col(html.Label(children='Ball Spot'), width={"order": "first"}),
        dcc.Dropdown(['Left Hash', 'Middle', 'Right Hash'], 'Middle', id='the_hash'),
        dbc.Col(html.Label(children='Offense Formation'), width={"order": "first"}),
        dcc.Dropdown(['Empty', 'Shotgun', 'I Formation', 'Jumbo', 'Pistol', 'Singleback', 'Wildcat'], 'Shotgun', id='offenseFormation')
    ]),
    dbc.Row([
        dbc.Col(html.Label(children='Yards to Go'), width={"order": "first"}),
        dbc.Col(dcc.Slider(min=1, max=15, step = 1, value = 2, id='yardsToGo')),
        html.Br()
    ]),
    dbc.Row([
        dbc.Col(html.Label(children='Yards From Endzone'), width={"order": "first"}),    
        dbc.Col(dcc.Slider(min = 1, max = 99, step = 5, value = 59, id = "absoluteYardlineNumber")),
        html.Br()
    ]),
    dbc.Row([
        dbc.Col(html.Label(children='Defenders in the Box'), width={"order": "first"}),    
        dbc.Col(dcc.Slider(min = 4, max = 7, step = 1, value = 6, id = "defendersInBox")),
        html.Br()
    ]),
    dbc.Row([
        dbc.Col(html.Label(children="Player 1's Distance from MoF (-26.66 to 26.66)"), width={"order": "first"}),    
        dcc.Input(id='rel_y_1', value=-5.03, type='number', min = -26.66, max = 26.66),
        dbc.Col(html.Label(children="Player 1's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_1', value=1.74, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 1's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'DE', id='official_position_1')
    ]),
    dbc.Row([
        dbc.Col(html.Label(children="Player 2's Distance from MoF (-26.66 to 26.66)"), width={"order": "first"}),    
        dcc.Input(id='rel_y_2', value=8.71, type='number', min = -26.66, max = 26.66),
        dbc.Col(html.Label(children="Player 2's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_2', value=2.34, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 2's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'DE', id='official_position_2')
    ]),
    dbc.Row([
        dbc.Col(html.Label(children="Player 3's Distance from MoF (-26.66 to 26.66)"), width={"order": "first"}),    
        dcc.Input(id='rel_y_3', value=5.43, type='number', min = -26.66, max = 26.66),
        dbc.Col(html.Label(children="Player 3's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_3', value=3.12, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 3's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'MLB', id='official_position_3')
    ]),
    dbc.Row([
        dbc.Col(html.Label(children="Player 4's Distance from MoF (-26.66 to 26.66)"), width={"order": "first"}),    
        dcc.Input(id='rel_y_4', value=13.70, type='number', min = -26.66, max = 26.66),
        dbc.Col(html.Label(children="Player 4's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_4', value=2.05, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 4's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'CB', id='official_position_4')
    ]),
    dbc.Row([
        dbc.Col(html.Label(children="Player 5's Distance from MoF (-26.66 to 26.66)"), width={"order": "first"}),    
        dcc.Input(id='rel_y_5', value=12.44, type='number', min = -26.66, max = 26.66),
        dbc.Col(html.Label(children="Player 5's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_5', value=3.05, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 5's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'CB', id='official_position_5')
    ]),
    dbc.Row([
        dbc.Col(html.Label(children="Player 6's Distance from MoF (-26.66 to 26.66)"), width={"order": "first"}),    
        dcc.Input(id='rel_y_6', value=1.26, type='number', min = -26.66, max = 26.66),
        dbc.Col(html.Label(children="Player 6's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_6', value=1.79, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 6's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'DE', id='official_position_6')
    ]),
    dbc.Row([
        dbc.Col(html.Label(children="Player 7's Distance from MoF (-26.66 to 26.66)"), width={"order": "first"}),    
        dcc.Input(id='rel_y_7', value=3.26, type='number', min = -26.66, max = 26.66),
        dbc.Col(html.Label(children="Player 7's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_7', value=17.43, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 7's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'SS', id='official_position_7')
    ]),
    dbc.Row([
        dbc.Col(html.Label(children="Player 8's Distance from MoF (-26.66 to 26.66)"), width={"order": "first"}),    
        dcc.Input(id='rel_y_8', value=-1.51, type='number', min = -26.66, max = 26.66),
        dbc.Col(html.Label(children="Player 8's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_8', value=6.51, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 8's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'SS', id='official_position_8')
    ]),
    dbc.Row([
        dbc.Col(html.Label(children="Player 9's Distance from MoF (-26.66 to 26.66)"), width={"order": "first"}),    
        dcc.Input(id='rel_y_9', value=-8.04, type='number', min = -26.66, max = 26.66),
        dbc.Col(html.Label(children="Player 9's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_9', value=1.79, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 9's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'CB', id='official_position_9')
    ]),
    dbc.Row([
        dbc.Col(html.Label(children="Player 10's Distance from MoF (-26.66 to 26.66)"), width={"order": "first"}),    
        dcc.Input(id='rel_y_10', value=-1.99, type='number', min = -26.66, max = 26.66),
        dbc.Col(html.Label(children="Player 10's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_10', value=2.12, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 10's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'OLB', id='official_position_10')
    ]),
    dbc.Row([
        dbc.Col(html.Label(children="Player 11's Distance from MoF (-26.66 to 26.66)"), width={"order": "first"}),    
        dcc.Input(id='rel_y_11', value=2.14, type='number', min = -26.66, max = 26.66),
        dbc.Col(html.Label(children="Player 11's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_11', value=2.75, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 11's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'DT', id='official_position_11')
    ]),
    dbc.Row([
        html.Br(),
        dbc.Row([dbc.Button('Submit', id='submit-val', n_clicks=0, color="primary")]),
        html.Br(),
        dbc.Row([html.Div(id='prediction output')])
    ]),
    dcc.Graph(id = 'play-diagram', style={'width': '90vh', 'height': '60vh'}) 
], style = {'padding': '0px 0px 0px 25px', 'width': '50%'})

@app.callback(
    # Output('graph', 'figure'),
    Output('prediction output', 'children'),
    Output('play-diagram', component_property= 'figure'),
    Input('submit-val', 'n_clicks'),
    State('yardsToGo', 'value'),
    State('absoluteYardlineNumber', 'value'),
    State('defendersInBox', 'value'),
    State('o_dropdown', 'value'),
    State('d_dropdown', 'value'),
    State('rel_x_1', 'value'),
    State('rel_y_1', 'value'),
    State('rel_x_2', 'value'),
    State('rel_y_2', 'value'),
    State('rel_x_3', 'value'),
    State('rel_y_3', 'value'),
    State('rel_x_4', 'value'),
    State('rel_y_4', 'value'),
    State('rel_x_5', 'value'),
    State('rel_y_5', 'value'),
    State('rel_x_6', 'value'),
    State('rel_y_6', 'value'),
    State('rel_x_7', 'value'),
    State('rel_y_7', 'value'),
    State('rel_x_8', 'value'),
    State('rel_y_8', 'value'),
    State('rel_x_9', 'value'),
    State('rel_y_9', 'value'),
    State('rel_x_10', 'value'),
    State('rel_y_10', 'value'),
    State('rel_x_11', 'value'),
    State('rel_y_11', 'value'),
    State('the_hash', 'value'),
    State('down', 'value'),
    State('official_position_1', 'value'),
    State('official_position_2', 'value'),
    State('official_position_3', 'value'),
    State('official_position_4', 'value'),
    State('official_position_5', 'value'),
    State('official_position_6', 'value'),
    State('official_position_7', 'value'),
    State('official_position_8', 'value'),
    State('official_position_9', 'value'),
    State('official_position_10', 'value'),
    State('official_position_11', 'value'),
    State('offenseFormation', 'value')
)
   
def update_output(n_clicks, yardsToGo, absoluteYardlineNumber, defendersInBox, o_dropdown, 
                  d_dropdown, rel_x_1, rel_y_1, rel_x_2, rel_y_2, rel_x_3, 
                  rel_y_3, rel_x_4, rel_y_4, rel_x_5, rel_y_5, rel_x_6, rel_y_6, 
                  rel_x_7, rel_y_7, rel_x_8, rel_y_8, rel_x_9, rel_y_9, rel_x_10, 
                  rel_y_10, rel_x_11, rel_y_11, the_hash, down, official_position_1, 
                  official_position_2, official_position_3, official_position_4, 
                  official_position_5, official_position_6, official_position_7, 
                  official_position_8, official_position_9, official_position_10,
                  official_position_11, offenseFormation):
    
    if o_dropdown == '12*':
        num_rb = float(1)
        num_te = float(2)
        num_wr = float(1)
    elif o_dropdown == '12':
        num_rb = float(1)
        num_te = float(2)
        num_wr = float(2)
    elif o_dropdown == '21':
        num_rb = float(2)
        num_te = float(1)
        num_wr = float(2)
    elif o_dropdown == '13':
        num_rb = float(1)
        num_te = float(3)
        num_wr = float(1)        
    elif o_dropdown == '10':
        num_rb = float(1)
        num_te = float(0)
        num_wr = float(4)       
    elif o_dropdown == '22':
        num_rb = float(2)
        num_te = float(2)
        num_wr = float(1)
    elif o_dropdown == '01':
        num_rb = float(0)
        num_te = float(1)
        num_wr = float(4)
    elif o_dropdown == '20':
        num_rb = float(2)
        num_te = float(0)
        num_wr = float(3)
    elif o_dropdown == '11*':
        num_rb = float(1)
        num_te = float(1)
        num_wr = float(2)
    elif o_dropdown == '02':
        num_rb = float(0)
        num_te = float(2)
        num_wr = float(3)
    else:
        num_rb = float(1)
        num_te = float(1)
        num_wr = float(3)
        
    if d_dropdown == '1-5-5':
        num_dl = float(1)
        num_lb = float(5)
        num_db = float(5)
    elif d_dropdown == '2-4-5':
        num_dl = float(2)
        num_lb = float(4)
        num_db = float(5)
    elif d_dropdown == '3-3-5':
        num_dl = float(3)
        num_lb = float(3)
        num_db = float(5)
    elif d_dropdown == '2-3-6':
        num_dl = float(2)
        num_lb = float(3)
        num_db = float(6)
    elif d_dropdown == '4-3-4':
        num_dl = float(4)
        num_lb = float(3)
        num_db = float(4)
    elif d_dropdown == '3-4-4':
        num_dl = float(3)
        num_lb = float(4)
        num_db = float(4)
    elif d_dropdown == '4-1-6':
        num_dl = float(4)
        num_lb = float(1)
        num_db = float(6)
    elif d_dropdown == '3-2-6':
        num_dl = float(3)
        num_lb = float(2)
        num_db = float(6)
    elif d_dropdown == '1-4-6':
        num_dl = float(1)
        num_lb = float(4)
        num_db = float(6)
    else:
        num_dl = float(4)
        num_lb = float(2)
        num_db = float(5)
        
    
    if the_hash == 'Right':
        ball_y_array = 29.7
    elif the_hash == 'Left':
        ball_y_array = 23.6
    else:
        ball_y_array = 27.0
    
    
    if float(down) == 4:
        down_1 = 0
        down_2 = 0
        down_3 = 0
        down_4 = 1
    elif float(down) == 3:
        down_1 = 0
        down_2 = 0
        down_3 = 1
        down_4 = 0      
    elif float(down) == 2:
        down_1 = 0
        down_2 = 1
        down_3 = 0
        down_4 = 0    
    else:
        down_1 = 1
        down_2 = 0
        down_3 = 0
        down_4 = 0    
    
    if official_position_1 == 'CB':
        cb_pos_1 = 1
        de_pos_1 = 0
        dt_pos_1 = 0
        fs_pos_1 = 0
        ilb_pos_1 = 0
        mlb_pos_1 = 0
        nt_pos_1 = 0
        olb_pos_1 = 0
        ss_pos_1 = 0
    elif official_position_1 == 'DE':
        cb_pos_1 = 0
        de_pos_1 = 1
        dt_pos_1 = 0
        fs_pos_1 = 0
        ilb_pos_1 = 0
        mlb_pos_1 = 0
        nt_pos_1 = 0
        olb_pos_1 = 0
        ss_pos_1 = 0
    elif official_position_1 == 'DT':
        cb_pos_1 = 0
        de_pos_1 = 0
        dt_pos_1 = 1
        fs_pos_1 = 0
        ilb_pos_1 = 0
        mlb_pos_1 = 0
        nt_pos_1 = 0
        olb_pos_1 = 0
        ss_pos_1 = 0
    elif official_position_1 == 'FS':
        cb_pos_1 = 0
        de_pos_1 = 0
        dt_pos_1 = 0
        fs_pos_1 = 1
        ilb_pos_1 = 0
        mlb_pos_1 = 0
        nt_pos_1 = 0
        olb_pos_1 = 0
        ss_pos_1 = 0
    elif official_position_1 == 'ILB':
        cb_pos_1 = 0
        de_pos_1 = 0
        dt_pos_1 = 0
        fs_pos_1 = 0
        ilb_pos_1 = 1
        mlb_pos_1 = 0
        nt_pos_1 = 0
        olb_pos_1 = 0
        ss_pos_1 = 0
    elif official_position_1 == 'MLB':
        cb_pos_1 = 0
        de_pos_1 = 0
        dt_pos_1 = 0
        fs_pos_1 = 0
        ilb_pos_1 = 0
        mlb_pos_1 = 1
        nt_pos_1 = 0
        olb_pos_1 = 0
        ss_pos_1 = 0
    elif official_position_1 == 'NT':
        cb_pos_1 = 0
        de_pos_1 = 0
        dt_pos_1 = 0
        fs_pos_1 = 0
        ilb_pos_1 = 0
        mlb_pos_1 = 0
        nt_pos_1 = 1
        olb_pos_1 = 0
        ss_pos_1 = 0
    elif official_position_1 == 'OLB':
        cb_pos_1 = 0
        de_pos_1 = 0
        dt_pos_1 = 0
        fs_pos_1 = 0
        ilb_pos_1 = 0
        mlb_pos_1 = 0
        nt_pos_1 = 0
        olb_pos_1 = 1
        ss_pos_1 = 0
    elif official_position_1 == 'SS':
        cb_pos_1 = 0
        de_pos_1 = 0
        dt_pos_1 = 0
        fs_pos_1 = 0
        ilb_pos_1 = 0
        mlb_pos_1 = 0
        nt_pos_1 = 0
        olb_pos_1 = 0
        ss_pos_1 = 1

    if official_position_2 == 'CB':
        cb_pos_2 = 1
        de_pos_2 = 0
        dt_pos_2 = 0
        fs_pos_2 = 0
        ilb_pos_2 = 0
        mlb_pos_2 = 0
        nt_pos_2 = 0
        olb_pos_2 = 0
        ss_pos_2 = 0
    elif official_position_2 == 'DE':
        cb_pos_2 = 0
        de_pos_2 = 1
        dt_pos_2 = 0
        fs_pos_2 = 0
        ilb_pos_2 = 0
        mlb_pos_2 = 0
        nt_pos_2 = 0
        olb_pos_2 = 0
        ss_pos_2 = 0
    elif official_position_2 == 'DT':
        cb_pos_2 = 0
        de_pos_2 = 0
        dt_pos_2 = 1
        fs_pos_2 = 0
        ilb_pos_2 = 0
        mlb_pos_2 = 0
        nt_pos_2 = 0
        olb_pos_2 = 0
        ss_pos_2 = 0
    elif official_position_2 == 'FS':
        cb_pos_2 = 0
        de_pos_2 = 0
        dt_pos_2 = 0
        fs_pos_2 = 1
        ilb_pos_2 = 0
        mlb_pos_2 = 0
        nt_pos_2 = 0
        olb_pos_2 = 0
        ss_pos_2 = 0
    elif official_position_2 == 'ILB':
        cb_pos_2 = 0
        de_pos_2 = 0
        dt_pos_2 = 0
        fs_pos_2 = 0
        ilb_pos_2 = 1
        mlb_pos_2 = 0
        nt_pos_2 = 0
        olb_pos_2 = 0
        ss_pos_2 = 0
    elif official_position_2 == 'MLB':
        cb_pos_2 = 0
        de_pos_2 = 0
        dt_pos_2 = 0
        fs_pos_2 = 0
        ilb_pos_2 = 0
        mlb_pos_2 = 1
        nt_pos_2 = 0
        olb_pos_2 = 0
        ss_pos_2 = 0
    elif official_position_2 == 'NT':
        cb_pos_2 = 0
        de_pos_2 = 0
        dt_pos_2 = 0
        fs_pos_2 = 0
        ilb_pos_2 = 0
        mlb_pos_2 = 0
        nt_pos_2 = 1
        olb_pos_2 = 0
        ss_pos_2 = 0
    elif official_position_2 == 'OLB':
        cb_pos_2 = 0
        de_pos_2 = 0
        dt_pos_2 = 0
        fs_pos_2 = 0
        ilb_pos_2 = 0
        mlb_pos_2 = 0
        nt_pos_2 = 0
        olb_pos_2 = 1
        ss_pos_2 = 0
    elif official_position_2 == 'SS':
        cb_pos_2 = 0
        de_pos_2 = 0
        dt_pos_2 = 0
        fs_pos_2 = 0
        ilb_pos_2 = 0
        mlb_pos_2 = 0
        nt_pos_2 = 0
        olb_pos_2 = 0
        ss_pos_2 = 1
        
    if official_position_3 == 'CB':
        cb_pos_3 = 1
        de_pos_3 = 0
        dt_pos_3 = 0
        fs_pos_3 = 0
        ilb_pos_3 = 0
        mlb_pos_3 = 0
        nt_pos_3 = 0
        olb_pos_3 = 0
        ss_pos_3 = 0
    elif official_position_3 == 'DE':
        cb_pos_3 = 0
        de_pos_3 = 1
        dt_pos_3 = 0
        fs_pos_3 = 0
        ilb_pos_3 = 0
        mlb_pos_3 = 0
        nt_pos_3 = 0
        olb_pos_3 = 0
        ss_pos_3 = 0
    elif official_position_3 == 'DT':
        cb_pos_3 = 0
        de_pos_3 = 0
        dt_pos_3 = 1
        fs_pos_3 = 0
        ilb_pos_3 = 0
        mlb_pos_3 = 0
        nt_pos_3 = 0
        olb_pos_3 = 0
        ss_pos_3 = 0
    elif official_position_3 == 'FS':
        cb_pos_3 = 0
        de_pos_3 = 0
        dt_pos_3 = 0
        fs_pos_3 = 1
        ilb_pos_3 = 0
        mlb_pos_3 = 0
        nt_pos_3 = 0
        olb_pos_3 = 0
        ss_pos_3 = 0
    elif official_position_3 == 'ILB':
        cb_pos_3 = 0
        de_pos_3 = 0
        dt_pos_3 = 0
        fs_pos_3 = 0
        ilb_pos_3 = 1
        mlb_pos_3 = 0
        nt_pos_3 = 0
        olb_pos_3 = 0
        ss_pos_3 = 0
    elif official_position_3 == 'MLB':
        cb_pos_3 = 0
        de_pos_3 = 0
        dt_pos_3 = 0
        fs_pos_3 = 0
        ilb_pos_3 = 0
        mlb_pos_3 = 1
        nt_pos_3 = 0
        olb_pos_3 = 0
        ss_pos_3 = 0
    elif official_position_3 == 'NT':
        cb_pos_3 = 0
        de_pos_3 = 0
        dt_pos_3 = 0
        fs_pos_3 = 0
        ilb_pos_3 = 0
        mlb_pos_3 = 0
        nt_pos_3 = 1
        olb_pos_3 = 0
        ss_pos_3 = 0
    elif official_position_3 == 'OLB':
        cb_pos_3 = 0
        de_pos_3 = 0
        dt_pos_3 = 0
        fs_pos_3 = 0
        ilb_pos_3 = 0
        mlb_pos_3 = 0
        nt_pos_3 = 0
        olb_pos_3 = 1
        ss_pos_3 = 0
    elif official_position_3 == 'SS':
        cb_pos_3 = 0
        de_pos_3 = 0
        dt_pos_3 = 0
        fs_pos_3 = 0
        ilb_pos_3 = 0
        mlb_pos_3 = 0
        nt_pos_3 = 0
        olb_pos_3 = 0
        ss_pos_3 = 1
    
    if official_position_4 == 'CB':
        cb_pos_4 = 1
        de_pos_4 = 0
        dt_pos_4 = 0
        fs_pos_4 = 0
        ilb_pos_4 = 0
        mlb_pos_4 = 0
        nt_pos_4 = 0
        olb_pos_4 = 0
        ss_pos_4 = 0
    elif official_position_4 == 'DE':
        cb_pos_4 = 0
        de_pos_4 = 1
        dt_pos_4 = 0
        fs_pos_4 = 0
        ilb_pos_4 = 0
        mlb_pos_4 = 0
        nt_pos_4 = 0
        olb_pos_4 = 0
        ss_pos_4 = 0
    elif official_position_4 == 'DT':
        cb_pos_4 = 0
        de_pos_4 = 0
        dt_pos_4 = 1
        fs_pos_4 = 0
        ilb_pos_4 = 0
        mlb_pos_4 = 0
        nt_pos_4 = 0
        olb_pos_4 = 0
        ss_pos_4 = 0
    elif official_position_4 == 'FS':
        cb_pos_4 = 0
        de_pos_4 = 0
        dt_pos_4 = 0
        fs_pos_4 = 1
        ilb_pos_4 = 0
        mlb_pos_4 = 0
        nt_pos_4 = 0
        olb_pos_4 = 0
        ss_pos_4 = 0
    elif official_position_4 == 'ILB':
        cb_pos_4 = 0
        de_pos_4 = 0
        dt_pos_4 = 0
        fs_pos_4 = 0
        ilb_pos_4 = 1
        mlb_pos_4 = 0
        nt_pos_4 = 0
        olb_pos_4 = 0
        ss_pos_4 = 0
    elif official_position_4 == 'MLB':
        cb_pos_4 = 0
        de_pos_4 = 0
        dt_pos_4 = 0
        fs_pos_4 = 0
        ilb_pos_4 = 0
        mlb_pos_4 = 1
        nt_pos_4 = 0
        olb_pos_4 = 0
        ss_pos_4 = 0
    elif official_position_4 == 'NT':
        cb_pos_4 = 0
        de_pos_4 = 0
        dt_pos_4 = 0
        fs_pos_4 = 0
        ilb_pos_4 = 0
        mlb_pos_4 = 0
        nt_pos_4 = 1
        olb_pos_4 = 0
        ss_pos_4 = 0
    elif official_position_4 == 'OLB':
        cb_pos_4 = 0
        de_pos_4 = 0
        dt_pos_4 = 0
        fs_pos_4 = 0
        ilb_pos_4 = 0
        mlb_pos_4 = 0
        nt_pos_4 = 0
        olb_pos_4 = 1
        ss_pos_4 = 0
    elif official_position_4 == 'SS':
        cb_pos_4 = 0
        de_pos_4 = 0
        dt_pos_4 = 0
        fs_pos_4 = 0
        ilb_pos_4 = 0
        mlb_pos_4 = 0
        nt_pos_4 = 0
        olb_pos_4 = 0
        ss_pos_4 = 1
    
    if official_position_5 == 'CB':
        cb_pos_5 = 1
        de_pos_5 = 0
        dt_pos_5 = 0
        fs_pos_5 = 0
        ilb_pos_5 = 0
        mlb_pos_5 = 0
        nt_pos_5 = 0
        olb_pos_5 = 0
        ss_pos_5 = 0
    elif official_position_5 == 'DE':
        cb_pos_5 = 0
        de_pos_5 = 1
        dt_pos_5 = 0
        fs_pos_5 = 0
        ilb_pos_5 = 0
        mlb_pos_5 = 0
        nt_pos_5 = 0
        olb_pos_5 = 0
        ss_pos_5 = 0
    elif official_position_5 == 'DT':
        cb_pos_5 = 0
        de_pos_5 = 0
        dt_pos_5 = 1
        fs_pos_5 = 0
        ilb_pos_5 = 0
        mlb_pos_5 = 0
        nt_pos_5 = 0
        olb_pos_5 = 0
        ss_pos_5 = 0
    elif official_position_5 == 'FS':
        cb_pos_5 = 0
        de_pos_5 = 0
        dt_pos_5 = 0
        fs_pos_5 = 1
        ilb_pos_5 = 0
        mlb_pos_5 = 0
        nt_pos_5 = 0
        olb_pos_5 = 0
        ss_pos_5 = 0
    elif official_position_5 == 'ILB':
        cb_pos_5 = 0
        de_pos_5 = 0
        dt_pos_5 = 0
        fs_pos_5 = 0
        ilb_pos_5 = 1
        mlb_pos_5 = 0
        nt_pos_5 = 0
        olb_pos_5 = 0
        ss_pos_5 = 0
    elif official_position_5 == 'MLB':
        cb_pos_5 = 0
        de_pos_5 = 0
        dt_pos_5 = 0
        fs_pos_5 = 0
        ilb_pos_5 = 0
        mlb_pos_5 = 1
        nt_pos_5 = 0
        olb_pos_5 = 0
        ss_pos_5 = 0
    elif official_position_5 == 'NT':
        cb_pos_5 = 0
        de_pos_5 = 0
        dt_pos_5 = 0
        fs_pos_5 = 0
        ilb_pos_5 = 0
        mlb_pos_5 = 0
        nt_pos_5 = 1
        olb_pos_5 = 0
        ss_pos_5 = 0
    elif official_position_5 == 'OLB':
        cb_pos_5 = 0
        de_pos_5 = 0
        dt_pos_5 = 0
        fs_pos_5 = 0
        ilb_pos_5 = 0
        mlb_pos_5 = 0
        nt_pos_5 = 0
        olb_pos_5 = 1
        ss_pos_5 = 0
    elif official_position_5 == 'SS':
        cb_pos_5 = 0
        de_pos_5 = 0
        dt_pos_5 = 0
        fs_pos_5 = 0
        ilb_pos_5 = 0
        mlb_pos_5 = 0
        nt_pos_5 = 0
        olb_pos_5 = 0
        ss_pos_5 = 1

    if official_position_6 == 'CB':
        cb_pos_6 = 1
        de_pos_6 = 0
        dt_pos_6 = 0
        fs_pos_6 = 0
        ilb_pos_6 = 0
        mlb_pos_6 = 0
        nt_pos_6 = 0
        olb_pos_6 = 0
        ss_pos_6 = 0
    elif official_position_6 == 'DE':
        cb_pos_6 = 0
        de_pos_6 = 1
        dt_pos_6 = 0
        fs_pos_6 = 0
        ilb_pos_6 = 0
        mlb_pos_6 = 0
        nt_pos_6 = 0
        olb_pos_6 = 0
        ss_pos_6 = 0
    elif official_position_6 == 'DT':
        cb_pos_6 = 0
        de_pos_6 = 0
        dt_pos_6 = 1
        fs_pos_6 = 0
        ilb_pos_6 = 0
        mlb_pos_6 = 0
        nt_pos_6 = 0
        olb_pos_6 = 0
        ss_pos_6 = 0
    elif official_position_6 == 'FS':
        cb_pos_6 = 0
        de_pos_6 = 0
        dt_pos_6 = 0
        fs_pos_6 = 1
        ilb_pos_6 = 0
        mlb_pos_6 = 0
        nt_pos_6 = 0
        olb_pos_6 = 0
        ss_pos_6 = 0
    elif official_position_6 == 'ILB':
        cb_pos_6 = 0
        de_pos_6 = 0
        dt_pos_6 = 0
        fs_pos_6 = 0
        ilb_pos_6 = 1
        mlb_pos_6 = 0
        nt_pos_6 = 0
        olb_pos_6 = 0
        ss_pos_6 = 0
    elif official_position_6 == 'MLB':
        cb_pos_6 = 0
        de_pos_6 = 0
        dt_pos_6 = 0
        fs_pos_6 = 0
        ilb_pos_6 = 0
        mlb_pos_6 = 1
        nt_pos_6 = 0
        olb_pos_6 = 0
        ss_pos_6 = 0
    elif official_position_6 == 'NT':
        cb_pos_6 = 0
        de_pos_6 = 0
        dt_pos_6 = 0
        fs_pos_6 = 0
        ilb_pos_6 = 0
        mlb_pos_6 = 0
        nt_pos_6 = 1
        olb_pos_6 = 0
        ss_pos_6 = 0
    elif official_position_6 == 'OLB':
        cb_pos_6 = 0
        de_pos_6 = 0
        dt_pos_6 = 0
        fs_pos_6 = 0
        ilb_pos_6 = 0
        mlb_pos_6 = 0
        nt_pos_6 = 0
        olb_pos_6 = 1
        ss_pos_6 = 0
    elif official_position_6 == 'SS':
        cb_pos_6 = 0
        de_pos_6 = 0
        dt_pos_6 = 0
        fs_pos_6 = 0
        ilb_pos_6 = 0
        mlb_pos_6 = 0
        nt_pos_6 = 0
        olb_pos_6 = 0
        ss_pos_6 = 1

    if official_position_7 == 'CB':
        cb_pos_7 = 1
        de_pos_7 = 0
        dt_pos_7 = 0
        fs_pos_7 = 0
        ilb_pos_7 = 0
        mlb_pos_7 = 0
        nt_pos_7 = 0
        olb_pos_7 = 0
        ss_pos_7 = 0
    elif official_position_7 == 'DE':
        cb_pos_7 = 0
        de_pos_7 = 1
        dt_pos_7 = 0
        fs_pos_7 = 0
        ilb_pos_7 = 0
        mlb_pos_7 = 0
        nt_pos_7 = 0
        olb_pos_7 = 0
        ss_pos_7 = 0
    elif official_position_7 == 'DT':
        cb_pos_7 = 0
        de_pos_7 = 0
        dt_pos_7 = 1
        fs_pos_7 = 0
        ilb_pos_7 = 0
        mlb_pos_7 = 0
        nt_pos_7 = 0
        olb_pos_7 = 0
        ss_pos_7 = 0
    elif official_position_7 == 'FS':
        cb_pos_7 = 0
        de_pos_7 = 0
        dt_pos_7 = 0
        fs_pos_7 = 1
        ilb_pos_7 = 0
        mlb_pos_7 = 0
        nt_pos_7 = 0
        olb_pos_7 = 0
        ss_pos_7 = 0
    elif official_position_7 == 'ILB':
        cb_pos_7 = 0
        de_pos_7 = 0
        dt_pos_7 = 0
        fs_pos_7 = 0
        ilb_pos_7 = 1
        mlb_pos_7 = 0
        nt_pos_7 = 0
        olb_pos_7 = 0
        ss_pos_7 = 0
    elif official_position_7 == 'MLB':
        cb_pos_7 = 0
        de_pos_7 = 0
        dt_pos_7 = 0
        fs_pos_7 = 0
        ilb_pos_7 = 0
        mlb_pos_7 = 1
        nt_pos_7 = 0
        olb_pos_7 = 0
        ss_pos_7 = 0
    elif official_position_7 == 'NT':
        cb_pos_7 = 0
        de_pos_7 = 0
        dt_pos_7 = 0
        fs_pos_7 = 0
        ilb_pos_7 = 0
        mlb_pos_7 = 0
        nt_pos_7 = 1
        olb_pos_7 = 0
        ss_pos_7 = 0
    elif official_position_7 == 'OLB':
        cb_pos_7 = 0
        de_pos_7 = 0
        dt_pos_7 = 0
        fs_pos_7 = 0
        ilb_pos_7 = 0
        mlb_pos_7 = 0
        nt_pos_7 = 0
        olb_pos_7 = 1
        ss_pos_7 = 0
    elif official_position_7 == 'SS':
        cb_pos_7 = 0
        de_pos_7 = 0
        dt_pos_7 = 0
        fs_pos_7 = 0
        ilb_pos_7 = 0
        mlb_pos_7 = 0
        nt_pos_7 = 0
        olb_pos_7 = 0
        ss_pos_7 = 1

    if official_position_8 == 'CB':
        cb_pos_8 = 1
        de_pos_8 = 0
        dt_pos_8 = 0
        fs_pos_8 = 0
        ilb_pos_8 = 0
        mlb_pos_8 = 0
        nt_pos_8 = 0
        olb_pos_8 = 0
        ss_pos_8 = 0
    elif official_position_8 == 'DE':
        cb_pos_8 = 0
        de_pos_8 = 1
        dt_pos_8 = 0
        fs_pos_8 = 0
        ilb_pos_8 = 0
        mlb_pos_8 = 0
        nt_pos_8 = 0
        olb_pos_8 = 0
        ss_pos_8 = 0
    elif official_position_8 == 'DT':
        cb_pos_8 = 0
        de_pos_8 = 0
        dt_pos_8 = 1
        fs_pos_8 = 0
        ilb_pos_8 = 0
        mlb_pos_8 = 0
        nt_pos_8 = 0
        olb_pos_8 = 0
        ss_pos_8 = 0
    elif official_position_8 == 'FS':
        cb_pos_8 = 0
        de_pos_8 = 0
        dt_pos_8 = 0
        fs_pos_8 = 1
        ilb_pos_8 = 0
        mlb_pos_8 = 0
        nt_pos_8 = 0
        olb_pos_8 = 0
        ss_pos_8 = 0
    elif official_position_8 == 'ILB':
        cb_pos_8 = 0
        de_pos_8 = 0
        dt_pos_8 = 0
        fs_pos_8 = 0
        ilb_pos_8 = 1
        mlb_pos_8 = 0
        nt_pos_8 = 0
        olb_pos_8 = 0
        ss_pos_8 = 0
    elif official_position_8 == 'MLB':
        cb_pos_8 = 0
        de_pos_8 = 0
        dt_pos_8 = 0
        fs_pos_8 = 0
        ilb_pos_8 = 0
        mlb_pos_8 = 1
        nt_pos_8 = 0
        olb_pos_8 = 0
        ss_pos_8 = 0
    elif official_position_8 == 'NT':
        cb_pos_8 = 0
        de_pos_8 = 0
        dt_pos_8 = 0
        fs_pos_8 = 0
        ilb_pos_8 = 0
        mlb_pos_8 = 0
        nt_pos_8 = 1
        olb_pos_8 = 0
        ss_pos_8 = 0
    elif official_position_8 == 'OLB':
        cb_pos_8 = 0
        de_pos_8 = 0
        dt_pos_8 = 0
        fs_pos_8 = 0
        ilb_pos_8 = 0
        mlb_pos_8 = 0
        nt_pos_8 = 0
        olb_pos_8 = 1
        ss_pos_8 = 0
    elif official_position_8 == 'SS':
        cb_pos_8 = 0
        de_pos_8 = 0
        dt_pos_8 = 0
        fs_pos_8 = 0
        ilb_pos_8 = 0
        mlb_pos_8 = 0
        nt_pos_8 = 0
        olb_pos_8 = 0
        ss_pos_8 = 1

    if official_position_9 == 'CB':
        cb_pos_9 = 1
        de_pos_9 = 0
        dt_pos_9 = 0
        fs_pos_9 = 0
        ilb_pos_9 = 0
        mlb_pos_9 = 0
        nt_pos_9 = 0
        olb_pos_9 = 0
        ss_pos_9 = 0
    elif official_position_9 == 'DE':
        cb_pos_9 = 0
        de_pos_9 = 1
        dt_pos_9 = 0
        fs_pos_9 = 0
        ilb_pos_9 = 0
        mlb_pos_9 = 0
        nt_pos_9 = 0
        olb_pos_9 = 0
        ss_pos_9 = 0
    elif official_position_9 == 'DT':
        cb_pos_9 = 0
        de_pos_9 = 0
        dt_pos_9 = 1
        fs_pos_9 = 0
        ilb_pos_9 = 0
        mlb_pos_9 = 0
        nt_pos_9 = 0
        olb_pos_9 = 0
        ss_pos_9 = 0
    elif official_position_9 == 'FS':
        cb_pos_9 = 0
        de_pos_9 = 0
        dt_pos_9 = 0
        fs_pos_9 = 1
        ilb_pos_9 = 0
        mlb_pos_9 = 0
        nt_pos_9 = 0
        olb_pos_9 = 0
        ss_pos_9 = 0
    elif official_position_9 == 'ILB':
        cb_pos_9 = 0
        de_pos_9 = 0
        dt_pos_9 = 0
        fs_pos_9 = 0
        ilb_pos_9 = 1
        mlb_pos_9 = 0
        nt_pos_9 = 0
        olb_pos_9 = 0
        ss_pos_9 = 0
    elif official_position_9 == 'MLB':
        cb_pos_9 = 0
        de_pos_9 = 0
        dt_pos_9 = 0
        fs_pos_9 = 0
        ilb_pos_9 = 0
        mlb_pos_9 = 1
        nt_pos_9 = 0
        olb_pos_9 = 0
        ss_pos_9 = 0
    elif official_position_9 == 'NT':
        cb_pos_9 = 0
        de_pos_9 = 0
        dt_pos_9 = 0
        fs_pos_9 = 0
        ilb_pos_9 = 0
        mlb_pos_9 = 0
        nt_pos_9 = 1
        olb_pos_9 = 0
        ss_pos_9 = 0
    elif official_position_9 == 'OLB':
        cb_pos_9 = 0
        de_pos_9 = 0
        dt_pos_9 = 0
        fs_pos_9 = 0
        ilb_pos_9 = 0
        mlb_pos_9 = 0
        nt_pos_9 = 0
        olb_pos_9 = 1
        ss_pos_9 = 0
    elif official_position_9 == 'SS':
        cb_pos_9 = 0
        de_pos_9 = 0
        dt_pos_9 = 0
        fs_pos_9 = 0
        ilb_pos_9 = 0
        mlb_pos_9 = 0
        nt_pos_9 = 0
        olb_pos_9 = 0
        ss_pos_9 = 1

    if official_position_10 == 'CB':
        cb_pos_10 = 1
        de_pos_10 = 0
        dt_pos_10 = 0
        fs_pos_10 = 0
        ilb_pos_10 = 0
        mlb_pos_10 = 0
        nt_pos_10 = 0
        olb_pos_10 = 0
        ss_pos_10 = 0
    elif official_position_10 == 'DE':
        cb_pos_10 = 0
        de_pos_10 = 1
        dt_pos_10 = 0
        fs_pos_10 = 0
        ilb_pos_10 = 0
        mlb_pos_10 = 0
        nt_pos_10 = 0
        olb_pos_10 = 0
        ss_pos_10 = 0
    elif official_position_10 == 'DT':
        cb_pos_10 = 0
        de_pos_10 = 0
        dt_pos_10 = 1
        fs_pos_10 = 0
        ilb_pos_10 = 0
        mlb_pos_10 = 0
        nt_pos_10 = 0
        olb_pos_10 = 0
        ss_pos_10 = 0
    elif official_position_10 == 'FS':
        cb_pos_10 = 0
        de_pos_10 = 0
        dt_pos_10 = 0
        fs_pos_10 = 1
        ilb_pos_10 = 0
        mlb_pos_10 = 0
        nt_pos_10 = 0
        olb_pos_10 = 0
        ss_pos_10 = 0
    elif official_position_10 == 'ILB':
        cb_pos_10 = 0
        de_pos_10 = 0
        dt_pos_10 = 0
        fs_pos_10 = 0
        ilb_pos_10 = 1
        mlb_pos_10 = 0
        nt_pos_10 = 0
        olb_pos_10 = 0
        ss_pos_10 = 0
    elif official_position_10 == 'MLB':
        cb_pos_10 = 0
        de_pos_10 = 0
        dt_pos_10 = 0
        fs_pos_10 = 0
        ilb_pos_10 = 0
        mlb_pos_10 = 1
        nt_pos_10 = 0
        olb_pos_10 = 0
        ss_pos_10 = 0
    elif official_position_10 == 'NT':
        cb_pos_10 = 0
        de_pos_10 = 0
        dt_pos_10 = 0
        fs_pos_10 = 0
        ilb_pos_10 = 0
        mlb_pos_10 = 0
        nt_pos_10 = 1
        olb_pos_10 = 0
        ss_pos_10 = 0
    elif official_position_10 == 'OLB':
        cb_pos_10 = 0
        de_pos_10 = 0
        dt_pos_10 = 0
        fs_pos_10 = 0
        ilb_pos_10 = 0
        mlb_pos_10 = 0
        nt_pos_10 = 0
        olb_pos_10 = 1
        ss_pos_10 = 0
    elif official_position_10 == 'SS':
        cb_pos_10 = 0
        de_pos_10 = 0
        dt_pos_10 = 0
        fs_pos_10 = 0
        ilb_pos_10 = 0
        mlb_pos_10 = 0
        nt_pos_10 = 0
        olb_pos_10 = 0
        ss_pos_10 = 1

    if official_position_11 == 'CB':
        cb_pos_11 = 1
        de_pos_11 = 0
        dt_pos_11 = 0
        fs_pos_11 = 0
        ilb_pos_11 = 0
        mlb_pos_11 = 0
        nt_pos_11 = 0
        olb_pos_11 = 0
        ss_pos_11 = 0
    elif official_position_11 == 'DE':
        cb_pos_11 = 0
        de_pos_11 = 1
        dt_pos_11 = 0
        fs_pos_11 = 0
        ilb_pos_11 = 0
        mlb_pos_11 = 0
        nt_pos_11 = 0
        olb_pos_11 = 0
        ss_pos_11 = 0
    elif official_position_11 == 'DT':
        cb_pos_11 = 0
        de_pos_11 = 0
        dt_pos_11 = 1
        fs_pos_11 = 0
        ilb_pos_11 = 0
        mlb_pos_11 = 0
        nt_pos_11 = 0
        olb_pos_11 = 0
        ss_pos_11 = 0
    elif official_position_11 == 'FS':
        cb_pos_11 = 0
        de_pos_11 = 0
        dt_pos_11 = 0
        fs_pos_11 = 1
        ilb_pos_11 = 0
        mlb_pos_11 = 0
        nt_pos_11 = 0
        olb_pos_11 = 0
        ss_pos_11 = 0
    elif official_position_11 == 'ILB':
        cb_pos_11 = 0
        de_pos_11 = 0
        dt_pos_11 = 0
        fs_pos_11 = 0
        ilb_pos_11 = 1
        mlb_pos_11 = 0
        nt_pos_11 = 0
        olb_pos_11 = 0
        ss_pos_11 = 0
    elif official_position_11 == 'MLB':
        cb_pos_11 = 0
        de_pos_11 = 0
        dt_pos_11 = 0
        fs_pos_11 = 0
        ilb_pos_11 = 0
        mlb_pos_11 = 1
        nt_pos_11 = 0
        olb_pos_11 = 0
        ss_pos_11 = 0
    elif official_position_11 == 'NT':
        cb_pos_11 = 0
        de_pos_11 = 0
        dt_pos_11 = 0
        fs_pos_11 = 0
        ilb_pos_11 = 0
        mlb_pos_11 = 0
        nt_pos_11 = 1
        olb_pos_11 = 0
        ss_pos_11 = 0
    elif official_position_11 == 'OLB':
        cb_pos_11 = 0
        de_pos_11 = 0
        dt_pos_11 = 0
        fs_pos_11 = 0
        ilb_pos_11 = 0
        mlb_pos_11 = 0
        nt_pos_11 = 0
        olb_pos_11 = 1
        ss_pos_11 = 0
    elif official_position_11 == 'SS':
        cb_pos_11 = 0
        de_pos_11 = 0
        dt_pos_11 = 0
        fs_pos_11 = 0
        ilb_pos_11 = 0
        mlb_pos_11 = 0
        nt_pos_11 = 0
        olb_pos_11 = 0
        ss_pos_11 = 1
    
    if offenseFormation == "Empty":
        empty_form = 1
        shotgun_form = 0
        iform_form = 0
        jumbo_form = 0
        pistol_form = 0
        singleback_form = 0
        other_form = 0
        qb_dist_from_ball = 5
        qb_rel_x = -5
        qb_rel_y = 0
        dist_from_qb_1 = np.sqrt(np.square(rel_x_1 + 5)  + np.square(rel_y_1 - 0))
        dist_from_qb_2 = np.sqrt(np.square(rel_x_2 + 5)  + np.square(rel_y_2 - 0))
        dist_from_qb_3 = np.sqrt(np.square(rel_x_3 + 5)  + np.square(rel_y_3 - 0))
        dist_from_qb_4 = np.sqrt(np.square(rel_x_4 + 5)  + np.square(rel_y_4 - 0))
        dist_from_qb_5 = np.sqrt(np.square(rel_x_5 + 5)  + np.square(rel_y_5 - 0))
        dist_from_qb_6 = np.sqrt(np.square(rel_x_6 + 5)  + np.square(rel_y_6 - 0))
        dist_from_qb_7 = np.sqrt(np.square(rel_x_7 + 5)  + np.square(rel_y_7 - 0))
        dist_from_qb_8 = np.sqrt(np.square(rel_x_8 + 5)  + np.square(rel_y_8 - 0))
        dist_from_qb_9 = np.sqrt(np.square(rel_x_9 + 5)  + np.square(rel_y_9 - 0))
        dist_from_qb_10 = np.sqrt(np.square(rel_x_10 + 5)  + np.square(rel_y_10 - 0))
        dist_from_qb_11 = np.sqrt(np.square(rel_x_11 + 5)  + np.square(rel_y_11 - 0))
    elif offenseFormation == 'Shotgun':
        empty_form = 0
        shotgun_form = 1
        iform_form = 0
        jumbo_form = 0
        pistol_form = 0
        singleback_form = 0
        other_form = 0
        qb_dist_from_ball = 5 
        qb_rel_x = -5
        qb_rel_y = 0
        dist_from_qb_1 = np.sqrt(np.square(rel_x_1 + 5)  + np.square(rel_y_1 - 0))
        dist_from_qb_2 = np.sqrt(np.square(rel_x_2 + 5)  + np.square(rel_y_2 - 0))
        dist_from_qb_3 = np.sqrt(np.square(rel_x_3 + 5)  + np.square(rel_y_3 - 0))
        dist_from_qb_4 = np.sqrt(np.square(rel_x_4 + 5)  + np.square(rel_y_4 - 0))
        dist_from_qb_5 = np.sqrt(np.square(rel_x_5 + 5)  + np.square(rel_y_5 - 0))
        dist_from_qb_6 = np.sqrt(np.square(rel_x_6 + 5)  + np.square(rel_y_6 - 0))
        dist_from_qb_7 = np.sqrt(np.square(rel_x_7 + 5)  + np.square(rel_y_7 - 0))
        dist_from_qb_8 = np.sqrt(np.square(rel_x_8 + 5)  + np.square(rel_y_8 - 0))
        dist_from_qb_9 = np.sqrt(np.square(rel_x_9 + 5)  + np.square(rel_y_9 - 0))
        dist_from_qb_10 = np.sqrt(np.square(rel_x_10 + 5)  + np.square(rel_y_10 - 0))
        dist_from_qb_11 = np.sqrt(np.square(rel_x_11 + 5)  + np.square(rel_y_11 - 0))
    elif offenseFormation == 'I Formation':
        empty_form = 0
        shotgun_form = 0
        iform_form = 1
        jumbo_form = 0
        pistol_form = 0
        singleback_form = 0
        other_form = 0
        qb_dist_from_ball = 1.5
        qb_rel_x = -1.5
        qb_rel_y = 0
        dist_from_qb_1 = np.sqrt(np.square(rel_x_1 + 1.5)  + np.square(rel_y_1 - 0))
        dist_from_qb_2 = np.sqrt(np.square(rel_x_2 + 1.5)  + np.square(rel_y_2 - 0))
        dist_from_qb_3 = np.sqrt(np.square(rel_x_3 + 1.5)  + np.square(rel_y_3 - 0))
        dist_from_qb_4 = np.sqrt(np.square(rel_x_4 + 1.5)  + np.square(rel_y_4 - 0))
        dist_from_qb_5 = np.sqrt(np.square(rel_x_5 + 1.5)  + np.square(rel_y_5 - 0))
        dist_from_qb_6 = np.sqrt(np.square(rel_x_6 + 1.5)  + np.square(rel_y_6 - 0))
        dist_from_qb_7 = np.sqrt(np.square(rel_x_7 + 1.5)  + np.square(rel_y_7 - 0))
        dist_from_qb_8 = np.sqrt(np.square(rel_x_8 + 1.5)  + np.square(rel_y_8 - 0))
        dist_from_qb_9 = np.sqrt(np.square(rel_x_9 + 1.5)  + np.square(rel_y_9 - 0))
        dist_from_qb_10 = np.sqrt(np.square(rel_x_10 + 1.5)  + np.square(rel_y_10 - 0))
        dist_from_qb_11 = np.sqrt(np.square(rel_x_11 + 1.5)  + np.square(rel_y_11 - 0))
    elif offenseFormation == 'Jumbo':
        empty_form = 0
        shotgun_form = 0
        iform_form = 0
        jumbo_form = 1
        pistol_form = 0
        singleback_form = 0
        other_form = 0
        qb_dist_from_ball = 1.5
        qb_rel_x = -1.5
        qb_rel_y = 0
        dist_from_qb_1 = np.sqrt(np.square(rel_x_1 + 1.5)  + np.square(rel_y_1 - 0))
        dist_from_qb_2 = np.sqrt(np.square(rel_x_2 + 1.5)  + np.square(rel_y_2 - 0))
        dist_from_qb_3 = np.sqrt(np.square(rel_x_3 + 1.5)  + np.square(rel_y_3 - 0))
        dist_from_qb_4 = np.sqrt(np.square(rel_x_4 + 1.5)  + np.square(rel_y_4 - 0))
        dist_from_qb_5 = np.sqrt(np.square(rel_x_5 + 1.5)  + np.square(rel_y_5 - 0))
        dist_from_qb_6 = np.sqrt(np.square(rel_x_6 + 1.5)  + np.square(rel_y_6 - 0))
        dist_from_qb_7 = np.sqrt(np.square(rel_x_7 + 1.5)  + np.square(rel_y_7 - 0))
        dist_from_qb_8 = np.sqrt(np.square(rel_x_8 + 1.5)  + np.square(rel_y_8 - 0))
        dist_from_qb_9 = np.sqrt(np.square(rel_x_9 + 1.5)  + np.square(rel_y_9 - 0))
        dist_from_qb_10 = np.sqrt(np.square(rel_x_10 + 1.5)  + np.square(rel_y_10 - 0))
        dist_from_qb_11 = np.sqrt(np.square(rel_x_11 + 1.5)  + np.square(rel_y_11 - 0))
    elif offenseFormation == 'Pistol':
        empty_form = 0
        shotgun_form = 0
        iform_form = 0
        jumbo_form = 0
        pistol_form = 1
        singleback_form = 0
        other_form = 0
        qb_dist_from_ball = 5
        qb_rel_x = 5
        qb_rel_y = 0
        dist_from_qb_1 = np.sqrt(np.square(rel_x_1 + 5)  + np.square(rel_y_1 - 0))
        dist_from_qb_2 = np.sqrt(np.square(rel_x_2 + 5)  + np.square(rel_y_2 - 0))
        dist_from_qb_3 = np.sqrt(np.square(rel_x_3 + 5)  + np.square(rel_y_3 - 0))
        dist_from_qb_4 = np.sqrt(np.square(rel_x_4 + 5)  + np.square(rel_y_4 - 0))
        dist_from_qb_5 = np.sqrt(np.square(rel_x_5 + 5)  + np.square(rel_y_5 - 0))
        dist_from_qb_6 = np.sqrt(np.square(rel_x_6 + 5)  + np.square(rel_y_6 - 0))
        dist_from_qb_7 = np.sqrt(np.square(rel_x_7 + 5)  + np.square(rel_y_7 - 0))
        dist_from_qb_8 = np.sqrt(np.square(rel_x_8 + 5)  + np.square(rel_y_8 - 0))
        dist_from_qb_9 = np.sqrt(np.square(rel_x_9 + 5)  + np.square(rel_y_9 - 0))
        dist_from_qb_10 = np.sqrt(np.square(rel_x_10 + 5)  + np.square(rel_y_10 - 0))
        dist_from_qb_11 = np.sqrt(np.square(rel_x_11 + 5)  + np.square(rel_y_11 - 0))
    elif offenseFormation == 'Singleback':
        empty_form = 0
        shotgun_form = 0
        iform_form = 0
        jumbo_form = 0
        pistol_form = 0
        singleback_form = 1
        other_form = 0
        qb_dist_from_ball = 1.5
        qb_rel_x = -1.5
        qb_rel_y = 0
        dist_from_qb_1 = np.sqrt(np.square(rel_x_1 + 1.5)  + np.square(rel_y_1 - 0))
        dist_from_qb_2 = np.sqrt(np.square(rel_x_2 + 1.5)  + np.square(rel_y_2 - 0))
        dist_from_qb_3 = np.sqrt(np.square(rel_x_3 + 1.5)  + np.square(rel_y_3 - 0))
        dist_from_qb_4 = np.sqrt(np.square(rel_x_4 + 1.5)  + np.square(rel_y_4 - 0))
        dist_from_qb_5 = np.sqrt(np.square(rel_x_5 + 1.5)  + np.square(rel_y_5 - 0))
        dist_from_qb_6 = np.sqrt(np.square(rel_x_6 + 1.5)  + np.square(rel_y_6 - 0))
        dist_from_qb_7 = np.sqrt(np.square(rel_x_7 + 1.5)  + np.square(rel_y_7 - 0))
        dist_from_qb_8 = np.sqrt(np.square(rel_x_8 + 1.5)  + np.square(rel_y_8 - 0))
        dist_from_qb_9 = np.sqrt(np.square(rel_x_9 + 1.5)  + np.square(rel_y_9 - 0))
        dist_from_qb_10 = np.sqrt(np.square(rel_x_10 + 1.5)  + np.square(rel_y_10 - 0))
        dist_from_qb_11 = np.sqrt(np.square(rel_x_11 + 1.5)  + np.square(rel_y_11 - 0))
    elif offenseFormation == "Wildcat":
        empty_form = 0
        shotgun_form = 0
        iform_form = 0
        jumbo_form = 0
        pistol_form = 0
        singleback_form = 0
        other_form = 1
        qb_dist_from_ball = 5
        qb_rel_x = 5
        qb_rel_y = 0
        dist_from_qb_1 = np.sqrt(np.square(rel_x_1 + 5)  + np.square(rel_y_1 - 0))
        dist_from_qb_2 = np.sqrt(np.square(rel_x_2 + 5)  + np.square(rel_y_2 - 0))
        dist_from_qb_3 = np.sqrt(np.square(rel_x_3 + 5)  + np.square(rel_y_3 - 0))
        dist_from_qb_4 = np.sqrt(np.square(rel_x_4 + 5)  + np.square(rel_y_4 - 0))
        dist_from_qb_5 = np.sqrt(np.square(rel_x_5 + 5)  + np.square(rel_y_5 - 0))
        dist_from_qb_6 = np.sqrt(np.square(rel_x_6 + 5)  + np.square(rel_y_6 - 0))
        dist_from_qb_7 = np.sqrt(np.square(rel_x_7 + 5)  + np.square(rel_y_7 - 0))
        dist_from_qb_8 = np.sqrt(np.square(rel_x_8 + 5)  + np.square(rel_y_8 - 0))
        dist_from_qb_9 = np.sqrt(np.square(rel_x_9 + 5)  + np.square(rel_y_9 - 0))
        dist_from_qb_10 = np.sqrt(np.square(rel_x_10 + 5)  + np.square(rel_y_10 - 0))
        dist_from_qb_11 = np.sqrt(np.square(rel_x_11 + 5)  + np.square(rel_y_11 - 0))

    x_1 = np.array([[float(yardsToGo), float(absoluteYardlineNumber), float(defendersInBox),
    float(num_rb), float(num_te), float(num_wr), float(num_dl), float(num_lb), float(num_db), float(rel_x_1), 
    float(rel_y_1), float(0.96), float(0.90), float(absoluteYardlineNumber), float(ball_y_array),
    float(-2.61), float(3.0), float(5.61), float(qb_dist_from_ball), float(qb_rel_x), float(qb_rel_y),
    float(dist_from_qb_1), float(down_1), float(down_2), float(down_3), float(down_4), float(cb_pos_1), 
    float(de_pos_1), float(dt_pos_1), float(fs_pos_1), float(ilb_pos_1), float(mlb_pos_1), float(nt_pos_1), 
    float(olb_pos_1), float(ss_pos_1), float(empty_form), float(iform_form), float(jumbo_form),
    float(pistol_form), float(shotgun_form), float(singleback_form), float(other_form)]])
    dtrain_1 = DMatrix(x_1)
    prediction_1 = round(100*(mod.predict(dtrain_1)[0]), 3)

    x_2 = np.array([[float(yardsToGo), float(absoluteYardlineNumber), float(defendersInBox),
    float(num_rb), float(num_te), float(num_wr), float(num_dl), float(num_lb), float(num_db), float(rel_x_2),
    float(rel_y_2), float(0.37), float(2.44), float(absoluteYardlineNumber), float(ball_y_array),
    float(-2.61), float(3.0), float(5.61), float(qb_dist_from_ball), float(qb_rel_x), float(qb_rel_y),
    float(dist_from_qb_2), float(down_1), float(down_2), float(down_3), float(down_4), float(cb_pos_2), 
    float(de_pos_2), float(dt_pos_2), float(fs_pos_2), float(ilb_pos_2), float(mlb_pos_2), float(nt_pos_2), 
    float(olb_pos_2), float(ss_pos_2), float(empty_form), float(iform_form), float(jumbo_form),
    float(pistol_form), float(shotgun_form), float(singleback_form), float(other_form)]])
    dtrain_2 = DMatrix(x_2)
    prediction_2 = round(100*(mod.predict(dtrain_2)[0]), 3)

    x_3 = np.array([[float(yardsToGo), float(absoluteYardlineNumber), float(defendersInBox),
    float(num_rb), float(num_te), float(num_wr), float(num_dl), float(num_lb), float(num_db), float(rel_x_3),
    float(rel_y_3), float(0.39), float(0.24), float(absoluteYardlineNumber), float(ball_y_array),
    float(-2.61), float(3.0), float(5.61), float(qb_dist_from_ball), float(qb_rel_x), float(qb_rel_y),
    float(dist_from_qb_3), float(down_1), float(down_2), float(down_3), float(down_4), float(cb_pos_3), 
    float(de_pos_3), float(dt_pos_3), float(fs_pos_3), float(ilb_pos_3), float(mlb_pos_3), float(nt_pos_3), 
    float(olb_pos_3), float(ss_pos_3), float(empty_form), float(iform_form), float(jumbo_form),
    float(pistol_form), float(shotgun_form), float(singleback_form), float(other_form)]])
    dtrain_3 = DMatrix(x_3)
    prediction_3 = round(100*(mod.predict(dtrain_3)[0]), 3)

    x_4 = np.array([[float(yardsToGo), float(absoluteYardlineNumber), float(defendersInBox),
    float(num_rb), float(num_te), float(num_wr), float(num_dl), float(num_lb), float(num_db), float(rel_x_4),
    float(rel_y_4), float(0.17), float(0.14), float(absoluteYardlineNumber), float(ball_y_array),
    float(-2.61), float(3.0), float(5.61), float(qb_dist_from_ball), float(qb_rel_x), float(qb_rel_y),
    float(dist_from_qb_4), float(down_1), float(down_2), float(down_3), float(down_4), float(cb_pos_4), 
    float(de_pos_4), float(dt_pos_4), float(fs_pos_4), float(ilb_pos_4), float(mlb_pos_4), float(nt_pos_4), 
    float(olb_pos_4), float(ss_pos_4), float(empty_form), float(iform_form), float(jumbo_form),
    float(pistol_form), float(shotgun_form), float(singleback_form), float(other_form)]])
    dtrain_4 = DMatrix(x_4)
    prediction_4 = round(100*(mod.predict(dtrain_4)[0]), 3)

    x_5 = np.array([[float(yardsToGo), float(absoluteYardlineNumber), float(defendersInBox),
    float(num_rb), float(num_te), float(num_wr), float(num_dl), float(num_lb), float(num_db), float(rel_x_5),
    float(rel_y_5), float(2.54), float(1.46), float(absoluteYardlineNumber), float(ball_y_array),
    float(-2.61), float(3.0), float(5.61), float(qb_dist_from_ball), float(qb_rel_x), float(qb_rel_y),
    float(dist_from_qb_5), float(down_1), float(down_2), float(down_3), float(down_4), float(cb_pos_5), 
    float(de_pos_5), float(dt_pos_5), float(fs_pos_5), float(ilb_pos_5), float(mlb_pos_5), float(nt_pos_5), 
    float(olb_pos_5), float(ss_pos_5), float(empty_form), float(iform_form), float(jumbo_form),
    float(pistol_form), float(shotgun_form), float(singleback_form), float(other_form)]])
    dtrain_5 = DMatrix(x_5)
    prediction_5 = round(100*(mod.predict(dtrain_5)[0]), 3)

    x_6 = np.array([[float(yardsToGo), float(absoluteYardlineNumber), float(defendersInBox),
    float(num_rb), float(num_te), float(num_wr), float(num_dl), float(num_lb), float(num_db), float(rel_x_6),
    float(rel_y_6), float(0.56), float(2.42), float(absoluteYardlineNumber), float(ball_y_array),
    float(-2.61), float(3.0), float(5.61), float(qb_dist_from_ball), float(qb_rel_x), float(qb_rel_y),
    float(dist_from_qb_6), float(down_1), float(down_2), float(down_3), float(down_4), float(cb_pos_6), 
    float(de_pos_6), float(dt_pos_6), float(fs_pos_6), float(ilb_pos_6), float(mlb_pos_6), float(nt_pos_6), 
    float(olb_pos_6), float(ss_pos_6), float(empty_form), float(iform_form), float(jumbo_form),
    float(pistol_form), float(shotgun_form), float(singleback_form), float(other_form)]])
    dtrain_6 = DMatrix(x_6)
    prediction_6 = round(100*(mod.predict(dtrain_6)[0]), 3)

    x_7 = np.array([[float(yardsToGo), float(absoluteYardlineNumber), float(defendersInBox),
    float(num_rb), float(num_te), float(num_wr), float(num_dl), float(num_lb), float(num_db), float(rel_x_7),
    float(rel_y_7), float(1.17), float(1.21), float(absoluteYardlineNumber), float(ball_y_array),
    float(-2.61), float(3.0), float(5.61), float(qb_dist_from_ball), float(qb_rel_x), float(qb_rel_y),
    float(dist_from_qb_7), float(down_1), float(down_2), float(down_3), float(down_4), float(cb_pos_7), 
    float(de_pos_7), float(dt_pos_7), float(fs_pos_7), float(ilb_pos_7), float(mlb_pos_7), float(nt_pos_7), 
    float(olb_pos_7), float(ss_pos_7), float(empty_form), float(iform_form), float(jumbo_form),
    float(pistol_form), float(shotgun_form), float(singleback_form), float(other_form)]])
    dtrain_7 = DMatrix(x_7)
    prediction_7 = round(100*(mod.predict(dtrain_7)[0]), 3)

    x_8 = np.array([[float(yardsToGo), float(absoluteYardlineNumber), float(defendersInBox),
    float(num_rb), float(num_te), float(num_wr), float(num_dl), float(num_lb), float(num_db), float(rel_x_8),
    float(rel_y_8), float(1.25), float(1.05), float(absoluteYardlineNumber), float(ball_y_array),
    float(-2.61), float(3.0), float(5.61), float(qb_dist_from_ball), float(qb_rel_x), float(qb_rel_y),
    float(dist_from_qb_8), float(down_1), float(down_2), float(down_3), float(down_4), float(cb_pos_8), 
    float(de_pos_8), float(dt_pos_8), float(fs_pos_8), float(ilb_pos_8), float(mlb_pos_8), float(nt_pos_8), 
    float(olb_pos_8), float(ss_pos_8), float(empty_form), float(iform_form), float(jumbo_form),
    float(pistol_form), float(shotgun_form), float(singleback_form), float(other_form)]])
    dtrain_8 = DMatrix(x_8)
    prediction_8 = round(100*(mod.predict(dtrain_8)[0]), 3)

    x_9 = np.array([[float(yardsToGo), float(absoluteYardlineNumber), float(defendersInBox),
    float(num_rb), float(num_te), float(num_wr), float(num_dl), float(num_lb), float(num_db), float(rel_x_9),
    float(rel_y_9), float(0.90), float(0.18), float(absoluteYardlineNumber), float(ball_y_array),
    float(-2.61), float(3.0), float(5.61), float(qb_dist_from_ball), float(qb_rel_x), float(qb_rel_y),
    float(dist_from_qb_9), float(down_1), float(down_2), float(down_3), float(down_4), float(cb_pos_9), 
    float(de_pos_9), float(dt_pos_9), float(fs_pos_9), float(ilb_pos_9), float(mlb_pos_9), float(nt_pos_9), 
    float(olb_pos_9), float(ss_pos_9), float(empty_form), float(iform_form), float(jumbo_form),
    float(pistol_form), float(shotgun_form), float(singleback_form), float(other_form)]])
    dtrain_9 = DMatrix(x_9)
    prediction_9 = round(100*(mod.predict(dtrain_9)[0]), 3)

    x_10 = np.array([[float(yardsToGo), float(absoluteYardlineNumber), float(defendersInBox),
    float(num_rb), float(num_te), float(num_wr), float(num_dl), float(num_lb), float(num_db), float(rel_x_10),
    float(rel_y_10), float(0.36), float(2.85), float(absoluteYardlineNumber), float(ball_y_array),
    float(-2.61), float(3.0), float(5.61), float(qb_dist_from_ball), float(qb_rel_x), float(qb_rel_y),
    float(dist_from_qb_10), float(down_1), float(down_2), float(down_3), float(down_4), float(cb_pos_10), 
    float(de_pos_10), float(dt_pos_10), float(fs_pos_10), float(ilb_pos_10), float(mlb_pos_10), float(nt_pos_10), 
    float(olb_pos_10), float(ss_pos_10), float(empty_form), float(iform_form), float(jumbo_form),
    float(pistol_form), float(shotgun_form), float(singleback_form), float(other_form)]])
    dtrain_10 = DMatrix(x_10)
    prediction_10 = round(100*(mod.predict(dtrain_10)[0]), 3)

    x_11 = np.array([[float(yardsToGo), float(absoluteYardlineNumber), float(defendersInBox),
    float(num_rb), float(num_te), float(num_wr), float(num_dl), float(num_lb), float(num_db), float(rel_x_11),
    float(rel_y_11), float(0.03), float(0.18), float(absoluteYardlineNumber), float(ball_y_array),
    float(-2.61), float(3.0), float(5.61), float(qb_dist_from_ball), float(qb_rel_x), float(qb_rel_y),
    float(dist_from_qb_11), float(down_1), float(down_2), float(down_3), float(down_4), float(cb_pos_11), 
    float(de_pos_11), float(dt_pos_11), float(fs_pos_11), float(ilb_pos_11), float(mlb_pos_11), float(nt_pos_11), 
    float(olb_pos_11), float(ss_pos_11), float(empty_form), float(iform_form), float(jumbo_form),
    float(pistol_form), float(shotgun_form), float(singleback_form), float(other_form)]])
    dtrain_11 = DMatrix(x_11)
    prediction_11 = round(100*(mod.predict(dtrain_11)[0]), 3)

    display_dist_1 = round(dist_from_qb_1, 1)
    display_dist_2 = round(dist_from_qb_2, 1)
    display_dist_3 = round(dist_from_qb_3, 1)
    display_dist_4 = round(dist_from_qb_4, 1)
    display_dist_5 = round(dist_from_qb_5, 1)
    display_dist_6 = round(dist_from_qb_6, 1)
    display_dist_7 = round(dist_from_qb_7, 1)
    display_dist_8 = round(dist_from_qb_8, 1)
    display_dist_9 = round(dist_from_qb_9, 1)
    display_dist_10 = round(dist_from_qb_10, 1)
    display_dist_11 = round(dist_from_qb_11, 1)

    data = [["Player 1", official_position_1, rel_x_1, rel_y_1, display_dist_1, prediction_1], 
            ["Player 2", official_position_2, rel_x_2, rel_y_2, display_dist_2, prediction_2],
            ["Player 3", official_position_3, rel_x_3, rel_y_3, display_dist_3, prediction_3],
            ["Player 4", official_position_4, rel_x_4, rel_y_4, display_dist_4, prediction_4],
            ["Player 5", official_position_5, rel_x_5, rel_y_5, display_dist_5, prediction_5],
            ["Player 6", official_position_6, rel_x_6, rel_y_6, display_dist_6, prediction_6],
            ["Player 7", official_position_7, rel_x_7, rel_y_7, display_dist_7, prediction_7],
            ["Player 8", official_position_8, rel_x_8, rel_y_8, display_dist_8, prediction_8],
            ["Player 9", official_position_9, rel_x_9, rel_y_9, display_dist_9, prediction_9],
            ["Player 10", official_position_10, rel_x_10, rel_y_10, display_dist_10, prediction_10],
            ["Player 11", official_position_11, rel_x_11, rel_y_11, display_dist_11, prediction_11]]
    
    df = pd.DataFrame(data,columns=['Player','Position', 'Rel. x', 'Rel. y', 'Dist. From QB', 'Chance of a Sack (%)'])
    df_graph = df
    df_graph['Off_Def'] = ['D','D','D','D','D','D','D','D','D','D','D']

    off_data = [["C", official_position_1, 0, 0, 0, 0.4, 'O'], 
            ["LG", official_position_2, 0, -1, 0, 0.4, 'O'],
            ["LT", official_position_3, 0, -2, 0, 0.4, 'O'],
            ["RG", official_position_4, 0, 1, 0, 0.4, 'O'],
            ["RT", official_position_4, 0, 2, 0, 0.4, 'O'],
            ["TE-R", official_position_4, -0.75, 3, 0, 0.4, 'O'],
            ["QB", official_position_4, -5, 0, 0, 0.4, 'O'],
            ["RB-L", official_position_4, -5, -1, 0, 0.4, 'O'],
            ["OR-WR", official_position_4, 0, 20, 0, 0.4, 'O'],
            ["OL-WR", official_position_4, 0, -20, 0, 0.4, 'O'],
            ["SL-WR", official_position_4, -1, -10, 0, 0.4, 'O']
    ]

    off_df = pd.DataFrame(off_data,columns=['Player','Position', 'Rel. x', 'Rel. y', 'Dist. From QB', 'Chance of a Sack (%)', 'Off_Def'])

    df_graph = df_graph.append(off_df)
    #diagram = px.scatter(data, x=2, y=3)
    diagram = px.scatter(df_graph, x='Rel. y', y='Rel. x', size = 'Chance of a Sack (%)', color = 'Off_Def')
    diagram.update_xaxes(range=[-26.65, 26.65])
    diagram.update_yaxes(range=[-10,20])

    if n_clicks:                            
        data = df.to_dict('rows')
        columns =  [{"name": i, "id": i,} for i in (df.columns)]
        return (dt.DataTable(data=data, columns=columns, sort_action='native', sort_mode='multi', sort_as_null=['', 'No'],
         sort_by=[{'column_id': 'Chance of a Sack (%)', 'direction': 'desc'}], editable=True), diagram)
    else :
        return(0, diagram)

if __name__ == '__main__':
    app.run_server(debug = True)