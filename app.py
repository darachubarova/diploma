from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from mathmodel import find_patterns
import urllib
import os
import base64
import io
import matplotlib.pyplot as plt
import zipfile
import matplotlib
matplotlib.use('agg')

external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?"
        "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "math model visualisation"

prev_calculate = 0
prev_patternlist = 0
prev_draw = 0
prev_draw_save = 0
data = pd.DataFrame()
index = list()
indicators = list()
num_patterns = np.array(list())
groups_calc = list()
eps = 0.1
h = 0.0
type_of_clasterisation = "abs"
tube_type = 'fixed'
charts = list()
load_data_prev = None
load_patterns_prev = None
draw_style = {'display': 'none'}
draw_save_style = {'display': 'none'}

app = Dash(prevent_initial_callbacks=True)
app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.H1(children="Tunnel Clustering in the Pattern Analysis", className="header-title"),
                html.P(
                    children="Man-Machine Procedure",
                    className="header-description",
                ),
            ],
            className="header",
        ),
        dcc.Upload(
            id='load-data',
            children=html.Div([
                html.A('Load data')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        ),
        dcc.Upload(
            id='load-patterns',
            children=html.Div([
                html.A('Load numbers of patterns')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        ),
        html.Div(
            id="params-div",
            style={'display': 'none'},
            children=[
                'enter numbers of groups of patterns to union: ',
                dcc.Input(
                id="input1",
                type="text",
                placeholder="no patterns to union",
                pattern=r'[0-9\s;]*',  # только цифры, пробелы и точки с запятой
                debounce=True
                ),
                'enter numbers of groups of patterns to recalculate: ',
                dcc.Input(
                    id="input2",
                    type="text",
                    placeholder="no patterns to recalculate",
                    pattern=r'[0-9\s]*',  # только цифры и пробелы
                    debounce=True,
                ),
                'eps = ',
                dcc.Input(
                    id="input3",
                    type="text",
                    placeholder="",
                    value="0.1",
                    debounce=True,
                    style={'width': '30px'},
                    pattern=r'[0-9.]*',  # только цифры и точки
                ),
                'h = ',
                dcc.Input(
                    id="input4",
                    type="text",
                    placeholder="",
                    value="0.0",
                    debounce=True,
                    style={'width': '30px'},
                    pattern=r'[0-9.]*',  # только цифры и точки
                )
            ],
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Div(children=[
                                    "Type of values for clustering",
                                    dcc.Dropdown(
                                        id="input5",
                                        options=[
                                            {"label": "absolute values", "value": "abs"},
                                            {"label": "tangent of values", "value": "tan"}
                                        ],
                                        value="abs",
                                        clearable=False,
                                        className="dropdown"
                                    )
                                    ],
                                    style={'display': 'none'},
                                    id="input5-div",
                                    className="menu-title"),
                                
                            ]
                        ),
                        html.Div(
                            children=[
                                html.Div(children=[
                                    "Type of E-tube",
                                    dcc.Dropdown(
                                        id="input6",
                                        options=[
                                        {"label": "fixed tube", "value": "fixed"},
                                        {"label": "adaptive tube", "value": "adaptive"},
                                        {"label": "combined tube", "value": "combined"}
                                        ],
                                        value="fixed",
                                        clearable=False,
                                        className="dropdown"
                                    )
                                ],
                                style={'display': 'none'},
                                id="input6-div",
                                className="menu-title"),
                                
                            ]
                        ),
                    ]
                ),
            ],
            className="menu",
        ),
        html.Div(
            id="wrapper",
            className="wrapper",
            style={'display': 'none'}
        ),
        html.Button(
            'Calculate patterns',
            id='calculate',
            n_clicks=0,
            style={'display': 'none'}
        ),
        html.Button(
            'Show list of patterns',
            id='patternlist-button',
            n_clicks=0,
            style={'display': 'none'}
        ),
        html.Button(
            'Draw interactive charts',
            id='draw',
            n_clicks=0,
            style={'display': 'none'}
        ),
        html.Div([
            html.Button(
                'Download charts as zip',
                id='draw-save',
                style={'display': 'none'}
            ),
            dcc.Download(id="download-charts"),
            html.Br(),
            ]
        ),
        html.Div(
            id="wrapper-list",
            className="wrapper-list",
            style={'display': 'none'}
        ),
        html.Div(
            id="wrapper-charts",
            className="wrapper-charts",
        ),
    ]
)

def list_of_patterns(num_patterns):
    data_charts = list()
    if num_patterns.shape[0] != 0:
        for i in range(num_patterns.max() + 1):
            data_charts.append(html.Div(u'Patern number {}: number of paterns {}'.format(i, (num_patterns == i).sum())))
    return data_charts

@app.callback(
    Output("download-charts", "data"),
    [Input('draw-save', 'n_clicks')]
)
def func(n_clicks):
    global num_patterns, data, index, indicators
    def write_archive(bytes_io):
        with zipfile.ZipFile(bytes_io, mode="w") as zf:
            for i in range(num_patterns.max() + 1):
                name = 'pattern ' + str(i)
                plt.title(name)
                mask = (num_patterns == i).reshape(-1)
                vecs = data[mask]
                plt.plot(indicators, vecs.T, c='b')
                pattern = data[mask].mean(axis=0)
                plt.plot(pattern, linestyle = '-', c='r')
                plt.ylim([0, 1]) 
                s = str(np.sum(num_patterns == i)) 
                plt.text(1, 0.9, s)
                buf = io.BytesIO()
                plt.savefig(buf)
                plt.close()
                img_name = "pattern_{:02d}.png".format(i)
                zf.writestr(img_name, buf.getvalue())
    return dcc.send_bytes(write_archive, "charts.zip")

def draw_func(num_patterns, data, index, indicators):
    
    data_charts = list()
    
    if num_patterns.shape[0] != 0:
        vec_nums = np.arange(data.shape[0])
        for i in range(num_patterns.max() + 1):
            fig = go.Figure()
            name = 'pattern ' + str(i)
            fig.update_layout(title=name, yaxis_range=[0,1])
    
            mask = (num_patterns == i).reshape(-1)
            indexes = np.array(data.shape[0])
            for count, vec in enumerate(data[mask]):
                name = str(index[vec_nums[mask][count]])
                fig_i = go.Scatter(x=indicators, y=vec, name=name,  marker_color='LightSkyBlue')
                fig.add_trace(fig_i)
            name = 'pattern'
            pattern = data[mask].mean(axis=0)
            fig_i = go.Scatter(x=indicators, y=pattern, name=name, marker_color='MediumPurple')
            fig.add_trace(fig_i)
            data_charts.append(html.Div(dcc.Graph(figure=fig), className='card',))
    
    return data_charts
    
@app.callback(
    Output("wrapper-list", 'style'),
    Output('patternlist-button', 'children'),
    [Input('patternlist-button', 'n_clicks')]
)
def show_list_func(n_clicks):
    if n_clicks % 2 == 1:
        return {'display': 'block'}, 'Hide list of patterns'
    return {'display': 'none'}, 'Show list of patterns'

@app.callback(
    Output("params-div", 'style'),
    Output("input5-div", 'style'),
    Output("input6-div", 'style'),
    Output('calculate', 'style'),
    Output('patternlist-button', 'style'),
    Output('draw', 'style'),
    Output('draw-save', 'style'),
    Output('wrapper', 'style'),
    Output("wrapper", "children"),
    Output("wrapper-list", "children"),
    Output("wrapper-charts", "children"),
    Output("input1", "value"),
    Output("input2", "value"),
    [
        Input('load-data', 'contents'),
        Input('load-patterns', 'contents'),
        Input("input1", "value"),
        Input("input2", "value"),
        Input("input3", "value"),
        Input("input4", "value"),
        Input("input5", "value"),
        Input("input6", "value"),
        Input('calculate', 'n_clicks'),
        Input('draw', 'n_clicks'),
    ],
)
def pattern_func(load_data, load_patterns, input1, input2, input3, input4, input5, input6, calculate, draw):
    global num_patterns, groups_calc, type_of_clasterisation, eps, h, tube_type, prev_calculate, prev_draw, charts, data, index, indicators, load_data_prev, load_patterns_prev, draw_style, draw_save_style
   
    if load_data !=  load_data_prev:
        load_data_prev = load_data
        content_type, content_string = load_data.split(',')
        decoded = base64.b64decode(content_string)
        data = pd.read_csv(io.StringIO(decoded.decode('utf-8')), index_col=0).astype(float)

        index = data.index.tolist()
        indicators = data.columns
        data = data.to_numpy()
        num_patterns = np.zeros(data.shape[0], dtype=int)
        
        charts = list()
        draw_style = {'display': 'block'}
        draw_save_style = {'display': 'block'}
        
    if load_patterns != load_patterns_prev:
        load_patterns_prev = load_patterns
        content_type, content_string = load_patterns.split(',')
        decoded = base64.b64decode(content_string)
        num_patterns = pd.read_csv(io.StringIO(decoded.decode('utf-8')), index_col=0).astype(int)
        if not index:
            index = num_patterns.index.tolist()
        num_patterns = num_patterns.to_numpy()
        charts = list()
    
    data_charts = list_of_patterns(num_patterns)
 
    if input1:
        
        flag1=True
        for elem in input1:
            if elem not in '1234567890 ;':
                flag1=False
                break
                
        flag2=True
        if flag1:
            for elem in map(int, ' '.join(input1.split(';')).split()):
                if elem > num_patterns.max():
                    flag2=False
                    break
        
        if flag1 and flag2:
            groups_union = input1.split(";")
            for i in range(len(groups_union)):
                groups_union[i] = list(map(int, groups_union[i].split()))
                
            if groups_union:
                print('groups_union')
                num_pattern_mask = np.ones(num_patterns.shape[0], dtype = bool)
                num_patterns_delta = np.zeros(num_patterns.shape[0], dtype=int).reshape((-1, 1))
                num_patterns_old = np.copy(num_patterns)
                for n, groups in enumerate(groups_union):
                    for i, group_elem in enumerate(groups):
                        mask = (num_patterns_old == group_elem).reshape(-1)
                        num_pattern_mask[mask] = False
                        num_patterns[mask] = n
                        num_patterns_delta[num_patterns_old > group_elem] -= 1

                num_patterns[num_pattern_mask] += len(groups_union) + num_patterns_delta[num_pattern_mask]
                
            input1=''

    if input2:
        flag1=True
        for elem in input2:
            if elem not in '1234567890 ':
                flag1=False
                break
                
        flag2=True
        if flag1:
            for elem in map(int, input2.split()):
                if elem > num_patterns.max():
                    flag2=False
                    break
        
        if flag1 and flag2:
                groups_calc = list(map(int, input2.split()))
                input2=''

    if input3:
        flag=True
        for elem in input3:
            if elem not in '1234567890.':
                flag=False
                break
        if flag:
            eps = float(input3)

    if input4:
        flag=True
        for elem in input4:
            if elem not in '1234567890.':
                flag=False
                break
        if flag:
            h = float(input4)

    if input5:
        type_of_clasterisation = input5

    if input6:
        tube_type = input6

    if calculate > prev_calculate:
        prev_calculate = calculate 

        if groups_calc:
            N_patterns = num_patterns.max()
            N_groups = len(groups_calc)
            data_mask = np.zeros(num_patterns.shape[0], dtype=bool)
            num_patterns_new = np.copy(num_patterns)
            for i, group in enumerate(groups_calc):
                data_mask += (num_patterns == group).reshape(-1)
                num_patterns_new[num_patterns > group] -= 1
            num_patterns_g = find_patterns(data[data_mask], type_of_clasterisation=type_of_clasterisation, eps=eps, h=h, tube_type=tube_type)
            num_patterns = num_patterns_new + num_patterns_g.max() + 1
            num_patterns[data_mask] = num_patterns_g
            
            input2 = ""

        else:
            num_patterns = find_patterns(data, type_of_clasterisation=type_of_clasterisation, eps=eps, h=h, tube_type=tube_type)
        
        charts = list()
    
    if draw > prev_draw:
        prev_draw = draw
        charts = draw_func(num_patterns, data, index, indicators)
        
        
    df = pd.DataFrame(index=index, data=num_patterns)
    csv_string = df.to_csv(encoding='utf-8', index=True)
    csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)

    wrapper = [
        html.A(
        "Download numbers of patterns",
        id="download-link",
        download="number_of_patterns.csv",
        href=csv_string,
        target="_blank",
        ),
        html.Br(),
        
    ]
    
    if groups_calc:
        wrapper += 'Сlustering will be applied to groups of patterns with numbers ' + ' '.join(list(map(str, groups_calc)))
    else:
        wrapper += 'Clustering will be applied to the entire sample'
        
    type_of_clasterisations = {'abs': 'absolute values', 'tan': 'tangent of values'}
        
    wrapper += u'with epsilon = {}, h = {}, type of values for classtering: {}, type of tube = {}'.format(eps, h, type_of_clasterisations[type_of_clasterisation], tube_type)
    
    return {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, draw_style, draw_save_style, {'display': 'block'}, wrapper, list_of_patterns(num_patterns), charts, input1, input2


if __name__ == "__main__":
    app.run_server(debug=True,
                   host = '127.0.0.1')
