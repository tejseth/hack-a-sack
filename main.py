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

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = 'Front Builder'
server = app.server

mod = Booster({'nthread': 8})
mod.load_model('xgb_sack')

app.layout = html.Div([
    # represents the browser address bar and doesn't render anything
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


index_page = html.Div([
    dbc.Row([html.H1(children='Hack-a-Sack Front Builder')]),
    dbc.Row([html.H2(children='BDB 2023 - Tej Seth, Joey DiCresce, Arjun Menon')]),
    dbc.Row([html.H5(children='How to use: input situation and formation variables in the corresponding dropdowns/sliders')]),
    dbc.Row([html.H5(children='There are two options for inputting player positions, the exact coordinate page, and the defensive technique page.')]),
    dbc.Row([html.H5(children='For the exact coordinate dashboard input player x and y coordinates, and defensive positions.')]),
    dbc.Row([html.H5(children='For the defensive technique dashboard input defensive player techniques, distance from LOS, L/R, and position.')]),
    dbc.Row([html.H5(children='For more info on defensive techniques see https://www.viqtorysports.com/understanding-defensive-techniques/')]),
    dbc.Row([html.H5(children='When finished, press submit to see updated graph and table displaying chance of a sack for each player.')]),

    dcc.Link('Go to Page 1 - Exact Coordinate Dashboard', href='/page-1'),
    html.Br(),
    dcc.Link('Go to Page 2 - Defensive Technique Dashboard', href='/page-2'),
])

# @callback(Output('page-content', 'children'),
#               [Input('url', 'pathname')])
# def display_page(pathname):
#     if(pathname == '/home'):
#         return html.Div([
#             html.H3(f'You are on page {pathname}')
#         ])
#     else:
#         return html.Div([
#             html.H3(f'oof')
#         ])



page_1_layout = html.Div([
    dbc.Row([html.H3(children='Front Builder - Exact Coordinate Dashboard')]),
    dcc.Link('Go back to home', href='/'),
    html.Br(),
    dcc.Link('Go to Page 2 - Defensive Technique Dashboard', href='/page-2'),
    dbc.Row([
        dbc.Col(html.Label(children='Down'), width={"order": "first"}),
        dcc.Dropdown(['1', '2', '3', '4'], '3', id='down', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children='Offensive Personnel (RB-TE)'), width={"order": "first"}),
        dcc.Dropdown(['11', '12', '21', '13', '10', '22', '01', '20', '11*', '02', '12*'], '11', id='o_dropdown', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children='Defensive Formation (DL-LB-DB)'), width={"order": "first"}),    
        dcc.Dropdown(['4-2-5', '2-4-5', '3-3-5', '2-3-6', '4-3-4', '3-4-4', '4-1-6', '3-2-6', '1-4-6', '1-5-5'], '4-2-5', id='d_dropdown', style={"width": "85%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children='Ball Spot'), width={"order": "first"}),
        dcc.Dropdown(['Left Hash', 'Middle', 'Right Hash'], 'Middle', id='the_hash', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children='Offense Formation'), width={"order": "first"}),
        dcc.Dropdown(['Empty', 'Shotgun', 'I Formation', 'Jumbo', 'Pistol', 'Singleback', 'Wildcat'], 'Shotgun', id='offenseFormation', style={"width": "80%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children='Defenders in the Box'), width={"order": "first"}),
        dcc.Dropdown(['4', '5', '6', '7'], '6', id='defendersInBox', style={"width": "80%", 'display': 'inline-block'})
    ], style={'columnCount': 3}),
   dbc.Row([
        dbc.Col(html.Label(children='Yards to Go'), width={"order": "first"}),
        dbc.Col(dcc.Slider(min=1, max=15, step = 1, value = 2, id='yardsToGo')),
        html.Br(),
        dbc.Col(html.Label(children='Yards From Endzone'), width={"order": "first"}),    
        dbc.Col(dcc.Slider(min = 1, max = 99, step = 5, value = 59, id = "absoluteYardlineNumber")),
        html.Br()
    ], style={'columnCount': 2}),
    dbc.Row([
        html.Br(),
        dbc.Row([dbc.Button('Submit', id='submit-val', n_clicks=0, color="primary")])
    ]),
    dbc.Row([
        html.Br(),
        dbc.Row([html.Div(id='prediction output-1')])
    ]),
    dcc.Graph(id = 'play-diagram-1', style={'width': '150vh', 'height': '80vh'}),
    dbc.Row([
        dbc.Col(html.Label(children="Player 1's Distance from MoF (-26.66 to 26.66)"), width={"order": "first"}),    
        dcc.Input(id='rel_y_1', value=-5.03, type='number', min = -26.66, max = 26.66),
        dbc.Col(html.Label(children="Player 1's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_1', value=1.74, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 1's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'DE', id='official_position_1', style={"width": "65%", 'display': 'inline-block'})
    ], style={'columnCount': 3}),
    dbc.Row([
        dbc.Col(html.Label(children="Player 2's Distance from MoF (-26.66 to 26.66)"), width={"order": "first"}),    
        dcc.Input(id='rel_y_2', value=5, type='number', min = -26.66, max = 26.66),
        dbc.Col(html.Label(children="Player 2's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_2', value=1.8, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 2's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'DE', id='official_position_2', style={"width": "65%", 'display': 'inline-block'})
    ], style={'columnCount': 3}),
    dbc.Row([
        dbc.Col(html.Label(children="Player 3's Distance from MoF (-26.66 to 26.66)"), width={"order": "first"}),    
        dcc.Input(id='rel_y_3', value=5.43, type='number', min = -26.66, max = 26.66),
        dbc.Col(html.Label(children="Player 3's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_3', value=3.12, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 3's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'OLB', id='official_position_3', style={"width": "65%", 'display': 'inline-block'})
    ], style={'columnCount': 3}),
    dbc.Row([
        dbc.Col(html.Label(children="Player 4's Distance from MoF (-26.66 to 26.66)"), width={"order": "first"}),    
        dcc.Input(id='rel_y_4', value=20, type='number', min = -26.66, max = 26.66),
        dbc.Col(html.Label(children="Player 4's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_4', value=2.05, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 4's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'CB', id='official_position_4', style={"width": "65%", 'display': 'inline-block'})
    ], style={'columnCount': 3}),
    dbc.Row([
        dbc.Col(html.Label(children="Player 5's Distance from MoF (-26.66 to 26.66)"), width={"order": "first"}),    
        dcc.Input(id='rel_y_5', value=-10, type='number', min = -26.66, max = 26.66),
        dbc.Col(html.Label(children="Player 5's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_5', value=3.05, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 5's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'CB', id='official_position_5', style={"width": "65%", 'display': 'inline-block'})
    ], style={'columnCount': 3}),
    dbc.Row([
        dbc.Col(html.Label(children="Player 6's Distance from MoF (-26.66 to 26.66)"), width={"order": "first"}),    
        dcc.Input(id='rel_y_6', value=1.26, type='number', min = -26.66, max = 26.66),
        dbc.Col(html.Label(children="Player 6's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_6', value=1.79, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 6's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'DT', id='official_position_6', style={"width": "65%", 'display': 'inline-block'})
    ], style={'columnCount': 3}),
    dbc.Row([
        dbc.Col(html.Label(children="Player 7's Distance from MoF (-26.66 to 26.66)"), width={"order": "first"}),    
        dcc.Input(id='rel_y_7', value=3.26, type='number', min = -26.66, max = 26.66),
        dbc.Col(html.Label(children="Player 7's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_7', value=17.43, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 7's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'SS', id='official_position_7', style={"width": "65%", 'display': 'inline-block'})
    ], style={'columnCount': 3}),
    dbc.Row([
        dbc.Col(html.Label(children="Player 8's Distance from MoF (-26.66 to 26.66)"), width={"order": "first"}),    
        dcc.Input(id='rel_y_8', value=-1.51, type='number', min = -26.66, max = 26.66),
        dbc.Col(html.Label(children="Player 8's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_8', value=6.51, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 8's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'SS', id='official_position_8', style={"width": "65%", 'display': 'inline-block'})
    ], style={'columnCount': 3}),
    dbc.Row([
        dbc.Col(html.Label(children="Player 9's Distance from MoF (-26.66 to 26.66)"), width={"order": "first"}),    
        dcc.Input(id='rel_y_9', value=-20, type='number', min = -26.66, max = 26.66),
        dbc.Col(html.Label(children="Player 9's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_9', value=1.79, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 9's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'CB', id='official_position_9', style={"width": "65%", 'display': 'inline-block'})
    ], style={'columnCount': 3}),
    dbc.Row([
        dbc.Col(html.Label(children="Player 10's Distance from MoF (-26.66 to 26.66)"), width={"order": "first"}),    
        dcc.Input(id='rel_y_10', value=-0.14, type='number', min = -26.66, max = 26.66),
        dbc.Col(html.Label(children="Player 10's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_10', value=2.75, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 10's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'MLB', id='official_position_10', style={"width": "65%", 'display': 'inline-block'})
    ], style={'columnCount': 3}),
    dbc.Row([
        dbc.Col(html.Label(children="Player 11's Distance from MoF (-26.66 to 26.66)"), width={"order": "first"}),    
        dcc.Input(id='rel_y_11', value=-1.75, type='number', min = -26.66, max = 26.66),
        dbc.Col(html.Label(children="Player 11's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_11', value=1.85, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 11's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'DT', id='official_position_11', style={"width": "65%", 'display': 'inline-block'})
    ], style={'columnCount': 3}),
    html.Br(),
    dcc.Link('Go to Page 2 - Defensive Technique Dashboard', href='/page-2'),
    html.Br(),
    dcc.Link('Go back to home', href='/')
], style = {'padding': '0px 0px 0px 25px', 'width': '90%'})


@app.callback(
    Output('prediction output-1', 'children'),
    Output('play-diagram-1', component_property= 'figure'),
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
    State('offenseFormation', 'value'))
    
 
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
        off_data = [["O", official_position_1, 0, 0, 0, 0.4, 'O'], # C 
            ["O", official_position_2, 0, -1, 0, 0.4, 'O'], # LG
            ["O", official_position_3, 0, -2, 0, 0.4, 'O'], # LT
            ["O", official_position_4, 0, 1, 0, 0.4, 'O'], # RG
            ["O", official_position_4, 0, 2, 0, 0.4, 'O'], # RT
            ["O", official_position_4, -0.75, 3, 0, 0.4, 'O'], #TE-R
            ["O", official_position_4, -5, 0, 0, 0.4, 'O'], # QB
            ["O", official_position_4, -1, -7, 0, 0.4, 'O'], # RB-L / SLiWR
            ["O", official_position_4, 0, 20, 0, 0.4, 'O'], # OR-WR
            ["O", official_position_4, 0, -20, 0, 0.4, 'O'], # OL_WR
            ["O", official_position_4, -1, -10, 0, 0.4, 'O'] # SL-WR
        ]
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
        off_data = [["O", official_position_1, 0, 0, 0, 0.4, 'O'], # C 
            ["O", official_position_2, 0, -1, 0, 0.4, 'O'], # LG
            ["O", official_position_3, 0, -2, 0, 0.4, 'O'], # LT
            ["O", official_position_4, 0, 1, 0, 0.4, 'O'], # RG
            ["O", official_position_4, 0, 2, 0, 0.4, 'O'], # RT
            ["O", official_position_4, -0.75, 3, 0, 0.4, 'O'], #TE-R
            ["O", official_position_4, -5, 0, 0, 0.4, 'O'], # QB
            ["O", official_position_4, -5, -1, 0, 0.4, 'O'], # RB-L
            ["O", official_position_4, 0, 20, 0, 0.4, 'O'], # OR-WR
            ["O", official_position_4, 0, -20, 0, 0.4, 'O'], # OL_WR
            ["O", official_position_4, -1, -10, 0, 0.4, 'O'] # SL-WR
        ]
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
        off_data = [["O", official_position_1, 0, 0, 0, 0.4, 'O'], # C 
            ["O", official_position_2, 0, -1, 0, 0.4, 'O'], # LG
            ["O", official_position_3, 0, -2, 0, 0.4, 'O'], # LT
            ["O", official_position_4, 0, 1, 0, 0.4, 'O'], # RG
            ["O", official_position_4, 0, 2, 0, 0.4, 'O'], # RT
            ["O", official_position_4, -0.75, 3, 0, 0.4, 'O'], #TE-R
            ["O", official_position_4, -1, 0, 0, 0.4, 'O'], # QB
            ["O", official_position_4, -7, 0, 0, 0.4, 'O'], # RB
            ["O", official_position_4, 0, 20, 0, 0.4, 'O'], # OR-WR
            ["O", official_position_4, 0, -20, 0, 0.4, 'O'], # OL_WR
            ["O", official_position_4, -3, 0, 0, 0.4, 'O'] # FB
        ]
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
        off_data = [["O", official_position_1, 0, 0, 0, 0.4, 'O'], # C 
            ["O", official_position_2, 0, -1, 0, 0.4, 'O'], # LG
            ["O", official_position_3, 0, -2, 0, 0.4, 'O'], # LT
            ["O", official_position_4, 0, 1, 0, 0.4, 'O'], # RG
            ["O", official_position_4, 0, 2, 0, 0.4, 'O'], # RT
            ["O", official_position_4, 0, 3, 0, 0.4, 'O'], #TE-iR
            ["O", official_position_4, -1, 0, 0, 0.4, 'O'], # QB
            ["O", official_position_4, -6, 0, 0, 0.4, 'O'], # RB-L
            ["O", official_position_4, -0.75, -20, 0, 0.4, 'O'], # OL-WR
            ["O", official_position_4, 0, -3, 0, 0.4, 'O'], # TE-L
            ["O", official_position_4, -0.75, 4, 0, 0.4, 'O'] # TE-oR
        ]  
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
        off_data = [["O", official_position_1, 0, 0, 0, 0.4, 'O'], # C 
            ["O", official_position_2, 0, -1, 0, 0.4, 'O'], # LG
            ["O", official_position_3, 0, -2, 0, 0.4, 'O'], # LT
            ["O", official_position_4, 0, 1, 0, 0.4, 'O'], # RG
            ["O", official_position_4, 0, 2, 0, 0.4, 'O'], # RT
            ["O", official_position_4, -0.75, 3, 0, 0.4, 'O'], #TE-R
            ["O", official_position_4, -4, 0, 0, 0.4, 'O'], # QB
            ["O", official_position_4, -7, 0, 0, 0.4, 'O'], # RB
            ["O", official_position_4, 0, 20, 0, 0.4, 'O'], # OR-WR
            ["O", official_position_4, 0, -20, 0, 0.4, 'O'], # OL_WR
            ["O", official_position_4, -1, -10, 0, 0.4, 'O'] # SL-WR
        ]
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
        off_data = [["O", official_position_1, 0, 0, 0, 0.4, 'O'], # C 
            ["O", official_position_2, 0, -1, 0, 0.4, 'O'], # LG
            ["O", official_position_3, 0, -2, 0, 0.4, 'O'], # LT
            ["O", official_position_4, 0, 1, 0, 0.4, 'O'], # RG
            ["O", official_position_4, 0, 2, 0, 0.4, 'O'], # RT
            ["O", official_position_4, -0.75, 3, 0, 0.4, 'O'], #TE-R
            ["O", official_position_4, -1, 0, 0, 0.4, 'O'], # QB
            ["O", official_position_4, -6, 0, 0, 0.4, 'O'], # RB-L
            ["O", official_position_4, 0, 20, 0, 0.4, 'O'], # OR-WR
            ["O", official_position_4, 0, -20, 0, 0.4, 'O'], # OL_WR
            ["O", official_position_4, -1, -10, 0, 0.4, 'O'] # SL-WR
        ]
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
        off_data = [["O", official_position_1, 0, 0, 0, 0.4, 'O'], # C 
            ["O", official_position_2, 0, -1, 0, 0.4, 'O'], # LG
            ["O", official_position_3, 0, -2, 0, 0.4, 'O'], # LT
            ["O", official_position_4, 0, 1, 0, 0.4, 'O'], # RG
            ["O", official_position_4, 0, 2, 0, 0.4, 'O'], # RT
            ["O", official_position_4, -0.75, 7, 0, 0.4, 'O'], #TE-R
            ["O", official_position_4, -5, 0, 0, 0.4, 'O'], # QB
            ["O", official_position_4, -5, -1, 0, 0.4, 'O'], # RB-L
            ["O", official_position_4, 0, 20, 0, 0.4, 'O'], # OR-WR
            ["O", official_position_4, 0, -20, 0, 0.4, 'O'], # OL_WR
            ["O", official_position_4, -1, -10, 0, 0.4, 'O'] # SL-WR
        ]

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

    data = [["1", official_position_1, rel_x_1, rel_y_1, display_dist_1, prediction_1], 
            ["2", official_position_2, rel_x_2, rel_y_2, display_dist_2, prediction_2],
            ["3", official_position_3, rel_x_3, rel_y_3, display_dist_3, prediction_3],
            ["4", official_position_4, rel_x_4, rel_y_4, display_dist_4, prediction_4],
            ["5", official_position_5, rel_x_5, rel_y_5, display_dist_5, prediction_5],
            ["6", official_position_6, rel_x_6, rel_y_6, display_dist_6, prediction_6],
            ["7", official_position_7, rel_x_7, rel_y_7, display_dist_7, prediction_7],
            ["8", official_position_8, rel_x_8, rel_y_8, display_dist_8, prediction_8],
            ["9", official_position_9, rel_x_9, rel_y_9, display_dist_9, prediction_9],
            ["10", official_position_10, rel_x_10, rel_y_10, display_dist_10, prediction_10],
            ["11", official_position_11, rel_x_11, rel_y_11, display_dist_11, prediction_11]]
    
    df = pd.DataFrame(data,columns=['Player','Position', 'Rel. x', 'Rel. y', 'Dist. From QB', 'Chance of a Sack (%)'])
    df_graph = df
    df_graph['Off_Def'] = ['D','D','D','D','D','D','D','D','D','D','D']

    

    off_df = pd.DataFrame(off_data,columns=['Player','Position', 'Rel. x', 'Rel. y', 'Dist. From QB', 'Chance of a Sack (%)', 'Off_Def'])

    df_graph = df_graph.append(off_df)
    #diagram = px.scatter(data, x=2, y=3)
    diagram = px.scatter(df_graph, x='Rel. y', y='Rel. x', size = 'Chance of a Sack (%)', color = 'Off_Def', text='Player',
    size_max=20)
    diagram.update_xaxes(range=[-26.65, 26.65])
    diagram.update_yaxes(range=[-10,20])

    if n_clicks:                            
        data = df.to_dict('rows')
        columns =  [{"name": i, "id": i,} for i in (df.columns)]
        return (dt.DataTable(data=data, columns=columns, sort_action='native', sort_mode='multi', sort_as_null=['', 'No'],
         sort_by=[{'column_id': 'Chance of a Sack (%)', 'direction': 'desc'}], style_cell={'textAlign': 'center',
        # all three widths are needed
        'minWidth': '150px', 'width': '150px', 'maxWidth': '150px',
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',
    }, style_header = {'fontWeight': 'bold'}, editable=True), diagram)
    else :
        return('Press submit to view results', diagram)


page_2_layout = html.Div([
    dbc.Row([html.H3(children='Front Builder - Defensive Technique Dashboard')]),
    dcc.Link('Go to Page 1 - Exact Coordinate Dashboard', href='/page-1'),
    html.Br(),
    dcc.Link('Go back to home', href='/'),
    dbc.Row([
        dbc.Col(html.Label(children='Down'), width={"order": "first"}),
        dcc.Dropdown(['1', '2', '3', '4'], '3', id='down', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children='Offensive Personnel (RB-TE)'), width={"order": "first"}),
        dcc.Dropdown(['11', '12', '21', '13', '10', '22', '01', '20', '11*', '02', '12*'], '11', id='o_dropdown', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children='Defensive Formation (DL-LB-DB)'), width={"order": "first"}),    
        dcc.Dropdown(['4-2-5', '2-4-5', '3-3-5', '2-3-6', '4-3-4', '3-4-4', '4-1-6', '3-2-6', '1-4-6', '1-5-5'], '4-2-5', id='d_dropdown', style={"width": "85%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children='Ball Spot'), width={"order": "first"}),
        dcc.Dropdown(['Left Hash', 'Middle', 'Right Hash'], 'Middle', id='the_hash', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children='Offense Formation'), width={"order": "first"}),
        dcc.Dropdown(['Empty', 'Shotgun', 'I Formation', 'Jumbo', 'Pistol', 'Singleback', 'Wildcat'], 'Shotgun', id='offenseFormation', style={"width": "80%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children='Defenders in the Box'), width={"order": "first"}),
        dcc.Dropdown(['4', '5', '6', '7'], '6', id='defendersInBox', style={"width": "80%", 'display': 'inline-block'})
    ], style={'columnCount': 3}),
    dbc.Row([
        dbc.Col(html.Label(children='Yards to Go'), width={"order": "first"}),
        dbc.Col(dcc.Slider(min=1, max=15, step = 1, value = 2, id='yardsToGo')),
        html.Br(),
        dbc.Col(html.Label(children='Yards From Endzone'), width={"order": "first"}),    
        dbc.Col(dcc.Slider(min = 1, max = 99, step = 5, value = 59, id = "absoluteYardlineNumber")),
        html.Br()
    ], style={'columnCount': 2}),
     dbc.Row([
        html.Br(),
        dbc.Row([dbc.Button('Submit', id='submit-val', n_clicks=0, color="primary")]),
        html.Br(),
        dbc.Row([html.Div(id='prediction output')])
    ]),
    dcc.Graph(id = 'play-diagram', style={'width': '150vh', 'height': '80vh'}),
    dbc.Row([
        dbc.Col(html.Label(children="Player 1 Defensive Right or Left"), width={"order": "first"}),
        dcc.Dropdown(['L', 'R'], 'L', id='LR_1', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children="Player 1's Technique/Alignment"), width={"order": "first"}),    
        dcc.Dropdown(['1', '2i', '2', '3', '4i', '4', '5', '6i', '6', '7/9', 'Wide 7/9' ,'Slot', 'Wide'], '7/9', id='tech_1', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children="Player 1's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_1', value=0.5, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 1's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'DE', id='official_position_1', style={"width": "50%", 'display': 'inline-block'})
    ], style={'columnCount': 4}),
    dbc.Row([
        dbc.Col(html.Label(children="Player 2 Defensive Right or Left"), width={"order": "first"}),
        dcc.Dropdown(['L', 'R'], 'R', id='LR_2', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children="Player 2's Technique/Alignment"), width={"order": "first"}),    
        dcc.Dropdown(['1', '2i', '2', '3', '4i', '4', '5', '6i', '6', '7/9', 'Wide 7/9' ,'Slot', 'Wide'], '5', id='tech_2', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children="Player 2's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_2', value=0.5, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 2's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'DE', id='official_position_2', style={"width": "50%", 'display': 'inline-block'})
    ], style={'columnCount': 4}),
    dbc.Row([
        dbc.Col(html.Label(children="Player 3 Defensive Right or Left"), width={"order": "first"}),
        dcc.Dropdown(['L', 'R'], 'L', id='LR_3', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children="Player 3's Technique/Alignment"), width={"order": "first"}),    
        dcc.Dropdown(['1', '2i', '2', '3', '4i', '4', '5', '6i', '6', '7/9', 'Wide 7/9' ,'Slot', 'Wide'], '3', id='tech_3', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children="Player 3's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_3', value=0.5, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 3's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'DT', id='official_position_3', style={"width": "50%", 'display': 'inline-block'})
    ], style={'columnCount': 4}),
    dbc.Row([
        dbc.Col(html.Label(children="Player 4 Defensive Right or Left"), width={"order": "first"}),
        dcc.Dropdown(['L', 'R'], 'R', id='LR_4', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children="Player 4's Technique/Alignment"), width={"order": "first"}),    
        dcc.Dropdown(['1', '2i', '2', '3', '4i', '4', '5', '6i', '6', '7/9', 'Wide 7/9' ,'Slot', 'Wide'], '1', id='tech_4', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children="Player 4's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_4', value=0.5, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 4's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'DT', id='official_position_4', style={"width": "50%", 'display': 'inline-block'})
    ], style={'columnCount': 4}),
    dbc.Row([
        dbc.Col(html.Label(children="Player 5 Defensive Right or Left"), width={"order": "first"}),
        dcc.Dropdown(['L', 'R'], 'L', id='LR_5', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children="Player 5's Technique/Alignment"), width={"order": "first"}),    
        dcc.Dropdown(['1', '2i', '2', '3', '4i', '4', '5', '6i', '6', '7/9', 'Wide 7/9' ,'Slot', 'Wide'], '2i', id='tech_5', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children="Player 5's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_5', value=4, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 5's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'MLB', id='official_position_5', style={"width": "50%", 'display': 'inline-block'})
    ], style={'columnCount': 4}),
    dbc.Row([
        dbc.Col(html.Label(children="Player 6 Defensive Right or Left"), width={"order": "first"}),
        dcc.Dropdown(['L', 'R'], 'R', id='LR_6', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children="Player 6's Technique/Alignment"), width={"order": "first"}),    
        dcc.Dropdown(['1', '2i', '2', '3', '4i', '4', '5', '6i', '6', '7/9', 'Wide 7/9' ,'Slot', 'Wide'], '3', id='tech_6', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children="Player 6's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_6', value=4, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 6's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'ILB', id='official_position_6', style={"width": "50%", 'display': 'inline-block'})
    ], style={'columnCount': 4}),
    dbc.Row([
        dbc.Col(html.Label(children="Player 7 Defensive Right or Left"), width={"order": "first"}),
        dcc.Dropdown(['L', 'R'], 'R', id='LR_7', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children="Player 7's Technique/Alignment"), width={"order": "first"}),    
        dcc.Dropdown(['1', '2i', '2', '3', '4i', '4', '5', '6i', '6', '7/9', 'Wide 7/9' ,'Slot', 'Wide'], 'Slot', id='tech_7', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children="Player 7's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_7', value=1.74, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 7's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'CB', id='official_position_7', style={"width": "50%", 'display': 'inline-block'})
    ], style={'columnCount': 4}),
    dbc.Row([
        dbc.Col(html.Label(children="Player 8 Defensive Right or Left"), width={"order": "first"}),
        dcc.Dropdown(['L', 'R'], 'L', id='LR_8', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children="Player 8's Technique/Alignment"), width={"order": "first"}),    
        dcc.Dropdown(['1', '2i', '2', '3', '4i', '4', '5', '6i', '6', '7/9', 'Wide 7/9' ,'Slot', 'Wide'], 'Wide', id='tech_8', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children="Player 8's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_8', value=1.74, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 8's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'CB', id='official_position_8', style={"width": "50%", 'display': 'inline-block'})
    ], style={'columnCount': 4}),
    dbc.Row([
        dbc.Col(html.Label(children="Player 9 Defensive Right or Left"), width={"order": "first"}),
        dcc.Dropdown(['L', 'R'], 'R', id='LR_9', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children="Player 9's Technique/Alignment"), width={"order": "first"}),    
        dcc.Dropdown(['1', '2i', '2', '3', '4i', '4', '5', '6i', '6', '7/9', 'Wide 7/9' ,'Slot', 'Wide'], 'Wide', id='tech_9', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children="Player 9's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_9', value=1.74, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 9's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'CB', id='official_position_9', style={"width": "50%", 'display': 'inline-block'})
    ], style={'columnCount': 4}),
    dbc.Row([
        dbc.Col(html.Label(children="Player 10 Defensive Right or Left"), width={"order": "first"}),
        dcc.Dropdown(['L', 'R'], 'L', id='LR_10', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children="Player 10's Technique/Alignment"), width={"order": "first"}),    
        dcc.Dropdown(['1', '2i', '2', '3', '4i', '4', '5', '6i', '6', '7/9', 'Wide 7/9' ,'Slot', 'Wide'], 'Slot', id='tech_10', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children="Player 2's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_10', value=8, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 10's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'SS', id='official_position_10', style={"width": "50%", 'display': 'inline-block'})
    ], style={'columnCount': 4}),
    dbc.Row([
        dbc.Col(html.Label(children="Player 11 Defensive Right or Left"), width={"order": "first"}),
        dcc.Dropdown(['L', 'R'], 'R', id='LR_11', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children="Player 11's Technique/Alignment"), width={"order": "first"}),    
        dcc.Dropdown(['1', '2i', '2', '3', '4i', '4', '5', '6i', '6', '7/9', 'Wide 7/9' ,'Slot', 'Wide'], 'Wide 7/9', id='tech_11', style={"width": "65%", 'display': 'inline-block'}),
        dbc.Col(html.Label(children="Player 11's Depth From LOS (0 to 45)"), width={"order": "first"}),    
        dcc.Input(id='rel_x_11', value=11, type='number', min = 0, max = 45),
        dbc.Col(html.Label(children="Player 11's Official Position"), width={"order": "first"}),
        dcc.Dropdown(['CB', 'DE', 'DT', 'FS', 'ILB', 'MLB', 'NT', 'OLB', 'SS'], 'FS', id='official_position_11', style={"width": "50%", 'display': 'inline-block'})
    ], style={'columnCount': 4}),
    dcc.Link('Go to Page 1 - Exact Coordinate Dashboard', href='/page-1'),
    html.Br(),
    dcc.Link('Go back to home', href='/'),
], style = {'padding': '0px 0px 0px 25px', 'width': '90%'})

@app.callback(
    Output('prediction output', 'children'),
    Output('play-diagram', component_property= 'figure'),
    Input('submit-val', 'n_clicks'),
    State('yardsToGo', 'value'),
    State('absoluteYardlineNumber', 'value'),
    State('defendersInBox', 'value'),
    State('o_dropdown', 'value'),
    State('d_dropdown', 'value'),
    State('rel_x_1', 'value'),
    State('tech_1', 'value'),
    State('rel_x_2', 'value'),
    State('tech_2', 'value'),
    State('rel_x_3', 'value'),
    State('tech_3', 'value'),
    State('rel_x_4', 'value'),
    State('tech_4', 'value'),
    State('rel_x_5', 'value'),
    State('tech_5', 'value'),
    State('rel_x_6', 'value'),
    State('tech_6', 'value'),
    State('rel_x_7', 'value'),
    State('tech_7', 'value'),
    State('rel_x_8', 'value'),
    State('tech_8', 'value'),
    State('rel_x_9', 'value'),
    State('tech_9', 'value'),
    State('rel_x_10', 'value'),
    State('tech_10', 'value'),
    State('rel_x_11', 'value'),
    State('tech_11', 'value'),
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
    State('offenseFormation', 'value'),
    State('LR_1', 'value'),
    State('LR_2', 'value'),
    State('LR_3', 'value'),
    State('LR_4', 'value'),
    State('LR_5', 'value'),
    State('LR_6', 'value'),
    State('LR_7', 'value'),
    State('LR_8', 'value'),
    State('LR_9', 'value'),
    State('LR_10', 'value'),
    State('LR_11', 'value'))


def update_output(n_clicks, yardsToGo, absoluteYardlineNumber, defendersInBox, o_dropdown, 
                  d_dropdown, rel_x_1, tech_1, rel_x_2, tech_2, rel_x_3, 
                  tech_3, rel_x_4, tech_4, rel_x_5, tech_5, rel_x_6, tech_6, 
                  rel_x_7, tech_7, rel_x_8, tech_8, rel_x_9, tech_9, rel_x_10, 
                  tech_10, rel_x_11, tech_11, the_hash, down, official_position_1, 
                  official_position_2, official_position_3, official_position_4, 
                  official_position_5, official_position_6, official_position_7, 
                  official_position_8, official_position_9, official_position_10,
                  official_position_11, offenseFormation,
                  LR_1, LR_2, LR_3, LR_4, LR_5, LR_6, LR_7, LR_8, LR_9, LR_10, LR_11):
    
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
    

# DEFENSIVE TECHNIQUE to REL Y CONVERSION 
# '1', '2i', '2', '3', '4i', '4', '5', '6i', '6', '7/9', 'Wide 7/9' ,'Slot', 'Wide'

    if tech_1 == '0':
        rel_y_1 = 0
    elif tech_1 == '1':
        if LR_1 == 'L':
            rel_y_1 = 0.2
        else:
            rel_y_1 = -0.2
    elif tech_1 == '2i':
        if LR_1 == 'L':
            rel_y_1 = 0.8
        else:
            rel_y_1 = -0.8
    elif tech_1 == '2':
        if LR_1 == 'L':
            rel_y_1 = 1
        else:
            rel_y_1 = -1
    elif tech_1 == '3':
        if LR_1 == 'L':
            rel_y_1 = 1.2
        else:
            rel_y_1 = -1.2
    elif tech_1 == '4i':
        if LR_1 == 'L':
            rel_y_1 = 1.8
        else:
            rel_y_1 = -1.8
    elif tech_1 == '4':
        if LR_1 == 'L':
            rel_y_1 = 2
        else:
            rel_y_1 = -2
    elif tech_1 == '5':
        if LR_1 == 'L':
            rel_y_1 = 2.2
        else:
            rel_y_1 = -2.2
    elif tech_1 == '6i':
        if LR_1 == 'L':
            rel_y_1 = 2.8
        else:
            rel_y_1 = -2.8
    elif tech_1 == '6':
        if LR_1 == 'L':
            rel_y_1 = 3
        else:
            rel_y_1 = -3
    elif tech_1 == '7/9':
        if LR_1 == 'L':
            rel_y_1 = 3.2
        else:
            rel_y_1 = -3.2
    elif tech_1 == 'Wide 7/9':
        if LR_1 == 'L':
            rel_y_1 = 4
        else:
            rel_y_1 = -4
    elif tech_1 == 'Slot':
        if LR_1 == 'L':
            rel_y_1 = 10
        else:
            rel_y_1 = -10
    elif tech_1 == 'Wide':
        if LR_1 == 'L':
            rel_y_1 = 20
        else:
            rel_y_1 = -20
    else:
        rel_y_1 = 26

# Player 2    
    if tech_2 == '0':
        rel_y_2 = 0
    elif tech_2 == '1':
        if LR_2 == 'L':
            rel_y_2 = 0.2
        else:
            rel_y_2 = -0.2
    elif tech_2 == '2i':
        if LR_2 == 'L':
            rel_y_2 = 0.8
        else:
            rel_y_2 = -0.8
    elif tech_2 == '2':
        if LR_2 == 'L':
            rel_y_2 = 1
        else:
            rel_y_2 = -1
    elif tech_2 == '3':
        if LR_2 == 'L':
            rel_y_2 = 1.2
        else:
            rel_y_2 = -1.2
    elif tech_2 == '4i':
        if LR_2 == 'L':
            rel_y_2 = 1.8
        else:
            rel_y_2 = -1.8
    elif tech_2 == '4':
        if LR_2 == 'L':
            rel_y_2 = 2
        else:
            rel_y_2 = -2
    elif tech_2 == '5':
        if LR_2 == 'L':
            rel_y_2 = 2.2
        else:
            rel_y_2 = -2.2
    elif tech_2 == '6i':
        if LR_2 == 'L':
            rel_y_2 = 2.8
        else:
            rel_y_2 = -2.8
    elif tech_2 == '6':
        if LR_2 == 'L':
            rel_y_2 = 3
        else:
            rel_y_2 = -3
    elif tech_2 == '7/9':
        if LR_2 == 'L':
            rel_y_2 = 3.2
        else:
            rel_y_2 = -3.2
    elif tech_2 == 'Wide 7/9':
        if LR_2 == 'L':
            rel_y_2 = 4
        else:
            rel_y_2 = -4
    elif tech_2 == 'Slot':
        if LR_2 == 'L':
            rel_y_2 = 10
        else:
            rel_y_2 = -10
    elif tech_2 == 'Wide':
        if LR_2 == 'L':
            rel_y_2 = 20
        else:
            rel_y_2 = -20
    else:
        rel_y_2 = 26

# Player 3    
    if tech_3 == '0':
        rel_y_3 = 0
    elif tech_3 == '1':
        if LR_3 == 'L':
            rel_y_3 = 0.2
        else:
            rel_y_3 = -0.2
    elif tech_3 == '2i':
        if LR_3 == 'L':
            rel_y_3 = 0.8
        else:
            rel_y_3 = -0.8
    elif tech_3 == '2':
        if LR_3 == 'L':
            rel_y_3 = 1
        else:
            rel_y_3 = -1
    elif tech_3 == '3':
        if LR_3 == 'L':
            rel_y_3 = 1.2
        else:
            rel_y_3 = -1.2
    elif tech_3 == '4i':
        if LR_3 == 'L':
            rel_y_3 = 1.8
        else:
            rel_y_3 = -1.8
    elif tech_3 == '4':
        if LR_3 == 'L':
            rel_y_3 = 2
        else:
            rel_y_3 = -2
    elif tech_3 == '5':
        if LR_3 == 'L':
            rel_y_3 = 2.2
        else:
            rel_y_3 = -2.2
    elif tech_3 == '6i':
        if LR_3 == 'L':
            rel_y_3 = 2.8
        else:
            rel_y_3 = -2.8
    elif tech_3 == '6':
        if LR_3 == 'L':
            rel_y_3 = 3
        else:
            rel_y_3 = -3
    elif tech_3 == '7/9':
        if LR_3 == 'L':
            rel_y_3 = 3.2
        else:
            rel_y_3 = -3.2
    elif tech_3 == 'Wide 7/9':
        if LR_3 == 'L':
            rel_y_3 = 4
        else:
            rel_y_3 = -4
    elif tech_3 == 'Slot':
        if LR_3 == 'L':
            rel_y_3 = 10
        else:
            rel_y_3 = -10
    elif tech_3 == 'Wide':
        if LR_3 == 'L':
            rel_y_3 = 20
        else:
            rel_y_3 = -20
    else:
        rel_y_3 = 26


# Player 4    
    if tech_4 == '0':
        rel_y_4 = 0
    elif tech_4 == '1':
        if LR_4 == 'L':
            rel_y_4 = 0.2
        else:
            rel_y_4 = -0.2
    elif tech_4 == '2i':
        if LR_4 == 'L':
            rel_y_4 = 0.8
        else:
            rel_y_4 = -0.8
    elif tech_4 == '2':
        if LR_4 == 'L':
            rel_y_4 = 1
        else:
            rel_y_4 = -1
    elif tech_4 == '3':
        if LR_4 == 'L':
            rel_y_4 = 1.2
        else:
            rel_y_4 = -1.2
    elif tech_4 == '4i':
        if LR_4 == 'L':
            rel_y_4 = 1.8
        else:
            rel_y_4 = -1.8
    elif tech_4 == '4':
        if LR_4 == 'L':
            rel_y_4 = 2
        else:
            rel_y_4 = -2
    elif tech_4 == '5':
        if LR_4 == 'L':
            rel_y_4 = 2.2
        else:
            rel_y_4 = -2.2
    elif tech_4 == '6i':
        if LR_4 == 'L':
            rel_y_4 = 2.8
        else:
            rel_y_4 = -2.8
    elif tech_4 == '6':
        if LR_4 == 'L':
            rel_y_4 = 3
        else:
            rel_y_4 = -3
    elif tech_4 == '7/9':
        if LR_4 == 'L':
            rel_y_4 = 3.2
        else:
            rel_y_4 = -3.2
    elif tech_4 == 'Wide 7/9':
        if LR_4 == 'L':
            rel_y_4 = 4
        else:
            rel_y_4 = -4
    elif tech_4 == 'Slot':
        if LR_4 == 'L':
            rel_y_4 = 10
        else:
            rel_y_4 = -10
    elif tech_4 == 'Wide':
        if LR_4 == 'L':
            rel_y_4 = 20
        else:
            rel_y_4 = -20
    else:
        rel_y_4 = 26


# Player 5    
    if tech_5 == '0':
        rel_y_5 = 0
    elif tech_5 == '1':
        if LR_5 == 'L':
            rel_y_5 = 0.2
        else:
            rel_y_5 = -0.2
    elif tech_5 == '2i':
        if LR_5 == 'L':
            rel_y_5 = 0.8
        else:
            rel_y_5 = -0.8
    elif tech_5 == '2':
        if LR_5 == 'L':
            rel_y_5 = 1
        else:
            rel_y_5 = -1
    elif tech_5 == '3':
        if LR_5 == 'L':
            rel_y_5 = 1.2
        else:
            rel_y_5 = -1.2
    elif tech_5 == '4i':
        if LR_5 == 'L':
            rel_y_5 = 1.8
        else:
            rel_y_5 = -1.8
    elif tech_5 == '4':
        if LR_5 == 'L':
            rel_y_5 = 2
        else:
            rel_y_5 = -2
    elif tech_5 == '5':
        if LR_5 == 'L':
            rel_y_5 = 2.2
        else:
            rel_y_5 = -2.2
    elif tech_5 == '6i':
        if LR_5 == 'L':
            rel_y_5 = 2.8
        else:
            rel_y_5 = -2.8
    elif tech_5 == '6':
        if LR_5 == 'L':
            rel_y_5 = 3
        else:
            rel_y_5 = -3
    elif tech_5 == '7/9':
        if LR_5 == 'L':
            rel_y_5 = 3.2
        else:
            rel_y_5 = -3.2
    elif tech_5 == 'Wide 7/9':
        if LR_5 == 'L':
            rel_y_5 = 4
        else:
            rel_y_5 = -4
    elif tech_5 == 'Slot':
        if LR_5 == 'L':
            rel_y_5 = 10
        else:
            rel_y_5 = -10
    elif tech_5 == 'Wide':
        if LR_5 == 'L':
            rel_y_5 = 20
        else:
            rel_y_5 = -20
    else:
        rel_y_5 = 26


# Player 6    
    if tech_6 == '0':
        rel_y_6 = 0
    elif tech_6 == '1':
        if LR_6 == 'L':
            rel_y_6 = 0.2
        else:
            rel_y_6 = -0.2
    elif tech_6 == '2i':
        if LR_6 == 'L':
            rel_y_6 = 0.8
        else:
            rel_y_6 = -0.8
    elif tech_6 == '2':
        if LR_6 == 'L':
            rel_y_6 = 1
        else:
            rel_y_6 = -1
    elif tech_6 == '3':
        if LR_6 == 'L':
            rel_y_6 = 1.2
        else:
            rel_y_6 = -1.2
    elif tech_6 == '4i':
        if LR_6 == 'L':
            rel_y_6 = 1.8
        else:
            rel_y_6 = -1.8
    elif tech_6 == '4':
        if LR_6 == 'L':
            rel_y_6 = 2
        else:
            rel_y_6 = -2
    elif tech_6 == '5':
        if LR_6 == 'L':
            rel_y_6 = 2.2
        else:
            rel_y_6 = -2.2
    elif tech_6 == '6i':
        if LR_6 == 'L':
            rel_y_6 = 2.8
        else:
            rel_y_6 = -2.8
    elif tech_6 == '6':
        if LR_6 == 'L':
            rel_y_6 = 3
        else:
            rel_y_6 = -3
    elif tech_6 == '7/9':
        if LR_6 == 'L':
            rel_y_6 = 3.2
        else:
            rel_y_6 = -3.2
    elif tech_6 == 'Wide 7/9':
        if LR_6 == 'L':
            rel_y_6 = 4
        else:
            rel_y_6 = -4
    elif tech_6 == 'Slot':
        if LR_6 == 'L':
            rel_y_6 = 10
        else:
            rel_y_6 = -10
    elif tech_6 == 'Wide':
        if LR_6 == 'L':
            rel_y_6 = 20
        else:
            rel_y_6 = -20
    else:
        rel_y_6 = 26

# Player 7    
    if tech_7 == '0':
        rel_y_7 = 0
    elif tech_7 == '1':
        if LR_7 == 'L':
            rel_y_7 = 0.2
        else:
            rel_y_7 = -0.2
    elif tech_7 == '2i':
        if LR_7 == 'L':
            rel_y_7 = 0.8
        else:
            rel_y_7 = -0.8
    elif tech_7 == '2':
        if LR_7 == 'L':
            rel_y_7 = 1
        else:
            rel_y_7 = -1
    elif tech_7 == '3':
        if LR_7 == 'L':
            rel_y_7 = 1.2
        else:
            rel_y_7 = -1.2
    elif tech_7 == '4i':
        if LR_7 == 'L':
            rel_y_7 = 1.8
        else:
            rel_y_7 = -1.8
    elif tech_7 == '4':
        if LR_7 == 'L':
            rel_y_7 = 2
        else:
            rel_y_7 = -2
    elif tech_7 == '5':
        if LR_7 == 'L':
            rel_y_7 = 2.2
        else:
            rel_y_7 = -2.2
    elif tech_7 == '6i':
        if LR_7 == 'L':
            rel_y_7 = 2.8
        else:
            rel_y_7 = -2.8
    elif tech_7 == '6':
        if LR_7 == 'L':
            rel_y_7 = 3
        else:
            rel_y_7 = -3
    elif tech_7 == '7/9':
        if LR_7 == 'L':
            rel_y_7 = 3.2
        else:
            rel_y_7 = -3.2
    elif tech_7 == 'Wide 7/9':
        if LR_7 == 'L':
            rel_y_7 = 4
        else:
            rel_y_7 = -4
    elif tech_7 == 'Slot':
        if LR_7 == 'L':
            rel_y_7 = 10
        else:
            rel_y_7 = -10
    elif tech_7 == 'Wide':
        if LR_7 == 'L':
            rel_y_7 = 20
        else:
            rel_y_7 = -20
    else:
        rel_y_7 = 26

# Player 8    
    if tech_8 == '0':
        rel_y_8 = 0
    elif tech_8 == '1':
        if LR_8 == 'L':
            rel_y_8 = 0.2
        else:
            rel_y_8 = -0.2
    elif tech_8 == '2i':
        if LR_8 == 'L':
            rel_y_8 = 0.8
        else:
            rel_y_8 = -0.8
    elif tech_8 == '2':
        if LR_8 == 'L':
            rel_y_8 = 1
        else:
            rel_y_8 = -1
    elif tech_8 == '3':
        if LR_8 == 'L':
            rel_y_8 = 1.2
        else:
            rel_y_8 = -1.2
    elif tech_8 == '4i':
        if LR_8 == 'L':
            rel_y_8 = 1.8
        else:
            rel_y_8 = -1.8
    elif tech_8 == '4':
        if LR_8 == 'L':
            rel_y_8 = 2
        else:
            rel_y_8 = -2
    elif tech_8 == '5':
        if LR_8 == 'L':
            rel_y_8 = 2.2
        else:
            rel_y_8 = -2.2
    elif tech_8 == '6i':
        if LR_8 == 'L':
            rel_y_8 = 2.8
        else:
            rel_y_8 = -2.8
    elif tech_8 == '6':
        if LR_8 == 'L':
            rel_y_8 = 3
        else:
            rel_y_8 = -3
    elif tech_8 == '7/9':
        if LR_8 == 'L':
            rel_y_8 = 3.2
        else:
            rel_y_8 = -3.2
    elif tech_8 == 'Wide 7/9':
        if LR_8 == 'L':
            rel_y_8 = 4
        else:
            rel_y_8 = -4
    elif tech_8 == 'Slot':
        if LR_8 == 'L':
            rel_y_8 = 10
        else:
            rel_y_8 = -10
    elif tech_8 == 'Wide':
        if LR_8 == 'L':
            rel_y_8 = 20
        else:
            rel_y_8 = -20
    else:
        rel_y_8 = 26

# Player 9    
    if tech_9 == '0':
        rel_y_9 = 0
    elif tech_9 == '1':
        if LR_9 == 'L':
            rel_y_9 = 0.2
        else:
            rel_y_9 = -0.2
    elif tech_9 == '2i':
        if LR_9 == 'L':
            rel_y_9 = 0.8
        else:
            rel_y_9 = -0.8
    elif tech_9 == '2':
        if LR_9 == 'L':
            rel_y_9 = 1
        else:
            rel_y_9 = -1
    elif tech_9 == '3':
        if LR_9 == 'L':
            rel_y_9 = 1.2
        else:
            rel_y_9 = -1.2
    elif tech_9 == '4i':
        if LR_9 == 'L':
            rel_y_9 = 1.8
        else:
            rel_y_9 = -1.8
    elif tech_9 == '4':
        if LR_9 == 'L':
            rel_y_9 = 2
        else:
            rel_y_9 = -2
    elif tech_9 == '5':
        if LR_9 == 'L':
            rel_y_9 = 2.2
        else:
            rel_y_9 = -2.2
    elif tech_9 == '6i':
        if LR_9 == 'L':
            rel_y_9 = 2.8
        else:
            rel_y_9 = -2.8
    elif tech_9 == '6':
        if LR_9 == 'L':
            rel_y_9 = 3
        else:
            rel_y_9 = -3
    elif tech_9 == '7/9':
        if LR_9 == 'L':
            rel_y_9 = 3.2
        else:
            rel_y_9 = -3.2
    elif tech_9 == 'Wide 7/9':
        if LR_9 == 'L':
            rel_y_9 = 4
        else:
            rel_y_9 = -4
    elif tech_9 == 'Slot':
        if LR_9 == 'L':
            rel_y_9 = 10
        else:
            rel_y_9 = -10
    elif tech_9 == 'Wide':
        if LR_9 == 'L':
            rel_y_9 = 20
        else:
            rel_y_9 = -20
    else:
        rel_y_9 = 26

# Player 10    
    if tech_10 == '0':
        rel_y_10 = 0
    elif tech_10 == '1':
        if LR_10 == 'L':
            rel_y_10 = 0.2
        else:
            rel_y_10 = -0.2
    elif tech_10 == '2i':
        if LR_10 == 'L':
            rel_y_10 = 0.8
        else:
            rel_y_10 = -0.8
    elif tech_10 == '2':
        if LR_10 == 'L':
            rel_y_10 = 1
        else:
            rel_y_10 = -1
    elif tech_10 == '3':
        if LR_10 == 'L':
            rel_y_10 = 1.2
        else:
            rel_y_10 = -1.2
    elif tech_10 == '4i':
        if LR_10 == 'L':
            rel_y_10 = 1.8
        else:
            rel_y_10 = -1.8
    elif tech_10 == '4':
        if LR_10 == 'L':
            rel_y_10 = 2
        else:
            rel_y_10 = -2
    elif tech_10 == '5':
        if LR_10 == 'L':
            rel_y_10 = 2.2
        else:
            rel_y_10 = -2.2
    elif tech_10 == '6i':
        if LR_10 == 'L':
            rel_y_10 = 2.8
        else:
            rel_y_10 = -2.8
    elif tech_10 == '6':
        if LR_10 == 'L':
            rel_y_10 = 3
        else:
            rel_y_10 = -3
    elif tech_10 == '7/9':
        if LR_10 == 'L':
            rel_y_10 = 3.2
        else:
            rel_y_10 = -3.2
    elif tech_10 == 'Wide 7/9':
        if LR_10 == 'L':
            rel_y_10 = 4
        else:
            rel_y_10 = -4
    elif tech_10 == 'Slot':
        if LR_10 == 'L':
            rel_y_10 = 10
        else:
            rel_y_10 = -10
    elif tech_10 == 'Wide':
        if LR_10 == 'L':
            rel_y_10 = 20
        else:
            rel_y_10 = -20
    else:
        rel_y_10 = 26

# Player 11    
    if tech_11 == '0':
        rel_y_11 = 0
    elif tech_11 == '1':
        if LR_11 == 'L':
            rel_y_11 = 0.2
        else:
            rel_y_11 = -0.2
    elif tech_11 == '2i':
        if LR_11 == 'L':
            rel_y_11 = 0.8
        else:
            rel_y_11 = -0.8
    elif tech_11 == '2':
        if LR_11 == 'L':
            rel_y_11 = 1
        else:
            rel_y_11 = -1
    elif tech_11 == '3':
        if LR_11 == 'L':
            rel_y_11 = 1.2
        else:
            rel_y_11 = -1.2
    elif tech_11 == '4i':
        if LR_11 == 'L':
            rel_y_11 = 1.8
        else:
            rel_y_11 = -1.8
    elif tech_11 == '4':
        if LR_11 == 'L':
            rel_y_11 = 2
        else:
            rel_y_11 = -2
    elif tech_11 == '5':
        if LR_11 == 'L':
            rel_y_11 = 2.2
        else:
            rel_y_11 = -2.2
    elif tech_11 == '6i':
        if LR_11 == 'L':
            rel_y_11 = 2.8
        else:
            rel_y_11 = -2.8
    elif tech_11 == '6':
        if LR_11 == 'L':
            rel_y_11 = 3
        else:
            rel_y_11 = -3
    elif tech_11 == '7/9':
        if LR_11 == 'L':
            rel_y_11 = 3.2
        else:
            rel_y_11 = -3.2
    elif tech_11 == 'Wide 7/9':
        if LR_11 == 'L':
            rel_y_11 = 4
        else:
            rel_y_11 = -4
    elif tech_11 == 'Slot':
        if LR_11 == 'L':
            rel_y_11 = 10
        else:
            rel_y_11 = -10
    elif tech_11 == 'Wide':
        if LR_11 == 'L':
            rel_y_11 = 20
        else:
            rel_y_11 = -20
    else:
        rel_y_11 = 26

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
        off_data = [["O", official_position_1, 0, 0, 0, 0.4, 'O'], # C 
            ["O", official_position_2, 0, -1, 0, 0.4, 'O'], # LG
            ["O", official_position_3, 0, -2, 0, 0.4, 'O'], # LT
            ["O", official_position_4, 0, 1, 0, 0.4, 'O'], # RG
            ["O", official_position_4, 0, 2, 0, 0.4, 'O'], # RT
            ["O", official_position_4, -0.75, 3, 0, 0.4, 'O'], #TE-R
            ["O", official_position_4, -5, 0, 0, 0.4, 'O'], # QB
            ["O", official_position_4, -1, -7, 0, 0.4, 'O'], # RB-L / SLiWR
            ["O", official_position_4, 0, 20, 0, 0.4, 'O'], # OR-WR
            ["O", official_position_4, 0, -20, 0, 0.4, 'O'], # OL_WR
            ["O", official_position_4, -1, -10, 0, 0.4, 'O'] # SL-WR
        ]
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
        off_data = [["O", official_position_1, 0, 0, 0, 0.4, 'O'], # C 
            ["O", official_position_2, 0, -1, 0, 0.4, 'O'], # LG
            ["O", official_position_3, 0, -2, 0, 0.4, 'O'], # LT
            ["O", official_position_4, 0, 1, 0, 0.4, 'O'], # RG
            ["O", official_position_4, 0, 2, 0, 0.4, 'O'], # RT
            ["O", official_position_4, -0.75, 3, 0, 0.4, 'O'], #TE-R
            ["O", official_position_4, -5, 0, 0, 0.4, 'O'], # QB
            ["O", official_position_4, -5, -1, 0, 0.4, 'O'], # RB-L
            ["O", official_position_4, 0, 20, 0, 0.4, 'O'], # OR-WR
            ["O", official_position_4, 0, -20, 0, 0.4, 'O'], # OL_WR
            ["O", official_position_4, -1, -10, 0, 0.4, 'O'] # SL-WR
        ]
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
        off_data = [["O", official_position_1, 0, 0, 0, 0.4, 'O'], # C 
            ["O", official_position_2, 0, -1, 0, 0.4, 'O'], # LG
            ["O", official_position_3, 0, -2, 0, 0.4, 'O'], # LT
            ["O", official_position_4, 0, 1, 0, 0.4, 'O'], # RG
            ["O", official_position_4, 0, 2, 0, 0.4, 'O'], # RT
            ["O", official_position_4, -0.75, 3, 0, 0.4, 'O'], #TE-R
            ["O", official_position_4, -1, 0, 0, 0.4, 'O'], # QB
            ["O", official_position_4, -7, 0, 0, 0.4, 'O'], # RB
            ["O", official_position_4, 0, 20, 0, 0.4, 'O'], # OR-WR
            ["O", official_position_4, 0, -20, 0, 0.4, 'O'], # OL_WR
            ["O", official_position_4, -3, 0, 0, 0.4, 'O'] # FB
        ]
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
        off_data = [["O", official_position_1, 0, 0, 0, 0.4, 'O'], # C 
            ["O", official_position_2, 0, -1, 0, 0.4, 'O'], # LG
            ["O", official_position_3, 0, -2, 0, 0.4, 'O'], # LT
            ["O", official_position_4, 0, 1, 0, 0.4, 'O'], # RG
            ["O", official_position_4, 0, 2, 0, 0.4, 'O'], # RT
            ["O", official_position_4, 0, 3, 0, 0.4, 'O'], #TE-iR
            ["O", official_position_4, -1, 0, 0, 0.4, 'O'], # QB
            ["O", official_position_4, -6, 0, 0, 0.4, 'O'], # RB-L
            ["O", official_position_4, -0.75, -20, 0, 0.4, 'O'], # OL-WR
            ["O", official_position_4, 0, -3, 0, 0.4, 'O'], # TE-L
            ["O", official_position_4, -0.75, 4, 0, 0.4, 'O'] # TE-oR
        ]  
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
        off_data = [["O", official_position_1, 0, 0, 0, 0.4, 'O'], # C 
            ["O", official_position_2, 0, -1, 0, 0.4, 'O'], # LG
            ["O", official_position_3, 0, -2, 0, 0.4, 'O'], # LT
            ["O", official_position_4, 0, 1, 0, 0.4, 'O'], # RG
            ["O", official_position_4, 0, 2, 0, 0.4, 'O'], # RT
            ["O", official_position_4, -0.75, 3, 0, 0.4, 'O'], #TE-R
            ["O", official_position_4, -4, 0, 0, 0.4, 'O'], # QB
            ["O", official_position_4, -7, 0, 0, 0.4, 'O'], # RB
            ["O", official_position_4, 0, 20, 0, 0.4, 'O'], # OR-WR
            ["O", official_position_4, 0, -20, 0, 0.4, 'O'], # OL_WR
            ["O", official_position_4, -1, -10, 0, 0.4, 'O'] # SL-WR
        ]
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
        off_data = [["O", official_position_1, 0, 0, 0, 0.4, 'O'], # C 
            ["O", official_position_2, 0, -1, 0, 0.4, 'O'], # LG
            ["O", official_position_3, 0, -2, 0, 0.4, 'O'], # LT
            ["O", official_position_4, 0, 1, 0, 0.4, 'O'], # RG
            ["O", official_position_4, 0, 2, 0, 0.4, 'O'], # RT
            ["O", official_position_4, -0.75, 3, 0, 0.4, 'O'], #TE-R
            ["O", official_position_4, -1, 0, 0, 0.4, 'O'], # QB
            ["O", official_position_4, -6, 0, 0, 0.4, 'O'], # RB-L
            ["O", official_position_4, 0, 20, 0, 0.4, 'O'], # OR-WR
            ["O", official_position_4, 0, -20, 0, 0.4, 'O'], # OL_WR
            ["O", official_position_4, -1, -10, 0, 0.4, 'O'] # SL-WR
        ]
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
        off_data = [["O", official_position_1, 0, 0, 0, 0.4, 'O'], # C 
            ["O", official_position_2, 0, -1, 0, 0.4, 'O'], # LG
            ["O", official_position_3, 0, -2, 0, 0.4, 'O'], # LT
            ["O", official_position_4, 0, 1, 0, 0.4, 'O'], # RG
            ["O", official_position_4, 0, 2, 0, 0.4, 'O'], # RT
            ["O", official_position_4, -0.75, 7, 0, 0.4, 'O'], #TE-R
            ["O", official_position_4, -5, 0, 0, 0.4, 'O'], # QB
            ["O", official_position_4, -5, -1, 0, 0.4, 'O'], # RB-L
            ["O", official_position_4, 0, 20, 0, 0.4, 'O'], # OR-WR
            ["O", official_position_4, 0, -20, 0, 0.4, 'O'], # OL_WR
            ["O", official_position_4, -1, -10, 0, 0.4, 'O'] # SL-WR
        ]
    
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

    data = [["1", official_position_1, rel_x_1, rel_y_1, display_dist_1, prediction_1], 
            ["2", official_position_2, rel_x_2, rel_y_2, display_dist_2, prediction_2],
            ["3", official_position_3, rel_x_3, rel_y_3, display_dist_3, prediction_3],
            ["4", official_position_4, rel_x_4, rel_y_4, display_dist_4, prediction_4],
            ["5", official_position_5, rel_x_5, rel_y_5, display_dist_5, prediction_5],
            ["6", official_position_6, rel_x_6, rel_y_6, display_dist_6, prediction_6],
            ["7", official_position_7, rel_x_7, rel_y_7, display_dist_7, prediction_7],
            ["8", official_position_8, rel_x_8, rel_y_8, display_dist_8, prediction_8],
            ["9", official_position_9, rel_x_9, rel_y_9, display_dist_9, prediction_9],
            ["10", official_position_10, rel_x_10, rel_y_10, display_dist_10, prediction_10],
            ["11", official_position_11, rel_x_11, rel_y_11, display_dist_11, prediction_11]]
    
    df = pd.DataFrame(data,columns=['Player','Position', 'Rel. x', 'Rel. y', 'Dist. From QB', 'Chance of a Sack (%)'])
    df_graph = df
    df_graph['Off_Def'] = ['D','D','D','D','D','D','D','D','D','D','D']


    off_df = pd.DataFrame(off_data,columns=['Player','Position', 'Rel. x', 'Rel. y', 'Dist. From QB', 'Chance of a Sack (%)', 'Off_Def'])

    df_graph = df_graph.append(off_df)
    #diagram = px.scatter(data, x=2, y=3)
    diagram = px.scatter(df_graph, x='Rel. y', y='Rel. x', size = 'Chance of a Sack (%)', color = 'Off_Def', text='Player',
    size_max=14)
    diagram.update_xaxes(range=[-26.65, 26.65])
    diagram.update_yaxes(range=[-10,20])

    if n_clicks:                            
        data = df.to_dict('rows')
        columns =  [{"name": i, "id": i,} for i in (df.columns)]
        return (dt.DataTable(data=data, columns=columns, sort_action='native', sort_mode='multi', sort_as_null=['', 'No'],
         sort_by=[{'column_id': 'Chance of a Sack (%)', 'direction': 'desc'}], style_cell={'textAlign': 'center',
         'minWidth': '150px', 'width': '150px', 'maxWidth': '150px',
         'overflow': 'hidden',
         'textOverflow': 'ellipsis',
         }, style_header = {'fontWeight': 'bold'}, editable=True), diagram)
    else :
        return('Press submit to view results', diagram)


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here

if __name__ == '__main__':
    app.run_server(debug=False)