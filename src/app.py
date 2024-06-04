"""
Dash app entry point

To launch the app, run

> python app.py

Dash documentation: https://dash.plot.ly/
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash.exceptions import PreventUpdate
from dash import callback_context
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, no_update, callback
import dash_player as dp
from dash import Dash, html, dash_table, dcc
import pandas as pd
import plotly.express as px
import os

# Check if the DEBUG environment variable is set
if 'DEBUG' in os.environ:
    debug = os.environ['DEBUG'] == 'True'
    print(f"DEBUG environment variable present, DEBUG set to {debug}")
else:
    print("No DEBUG environment variable: defaulting to debug mode")
    debug = True

# Load data
plotting_df = pd.read_csv('/storyboard-visualization/data/plotting_df.csv')
# group the plotting_df by clip and add a column of indices. This will be used to calculate the peak indices later.
plotting_df['index'] = plotting_df.groupby('clip').cumcount()
# rename the indices column to peak_indices
plotting_df.rename(columns={'index': 'peak_indices'}, inplace=True)
plotting_df['frame_number'] = plotting_df.apply(lambda x: int((x['peak_indices']/x['pdf_len'])*x['num_frames']), axis=1)
plotting_df['image_path'] = plotting_df.apply(lambda x: 'https://adibuoy23.github.io/event_representations/video_frames/'+x['Movie+clip_number']+'/frames'+str(x['frame_number']).zfill(4)+'.jpg', axis=1)

# Extract the storyboard and event boundary peaks from the plotting_df
df = plotting_df[(plotting_df['sb_peaks']==1) | (plotting_df['eb_peaks']==1)].reset_index()
df['type'] = df.apply(lambda x: 'storyboards' if x['sb_peaks']==1 else 'eventboundaries', axis=1)

# Write functions to make figures
def make_scatter_fig(df, type='Map max to 1'):
    if type == 'Map max to 1':
        x = 'sum_dist_scaled'
        y = 'diff_dist_scaled'
        x_title = '(SB + EB distribution)'
        y_title = '(SB - EB) distribution'
    elif type == 'zscore':
        x = 'sum_dist_z'
        y = 'diff_dist_z'
        x_title = '(SB + EB distribution)'
        y_title = '(SB - EB) distribution'
    satter_fig = go.Figure()
    for i, clip in enumerate(df['clip'].unique()):
        clip_df = df[df['clip']==clip]
        sb_clip_df = clip_df[clip_df['sb_peaks']==1]
        eb_clip_df = clip_df[clip_df['eb_peaks']==1]
        if i == 0:
            satter_fig.add_trace(go.Scatter(x=sb_clip_df[x], y=sb_clip_df[y], mode='markers', name='Storyboards', showlegend=True, marker=dict(size=10, opacity = 0.35, color='orange')))
            satter_fig.add_trace(go.Scatter(x=eb_clip_df[x], y=eb_clip_df[y], mode='markers', name='Eventboundaries', showlegend=True, marker=dict(size=10, opacity = 0.35, color='red')))
        else:
            satter_fig.add_trace(go.Scatter(x=sb_clip_df[x], y=sb_clip_df[y], mode='markers', name=str(clip), showlegend=False, marker=dict(size=10, opacity = 0.35, color='orange')))
            satter_fig.add_trace(go.Scatter(x=eb_clip_df[x], y=eb_clip_df[y], mode='markers', name=str(clip), showlegend=False, marker=dict(size=10, opacity = 0.35, color='red')))
    satter_fig.add_annotation(x=1.15, y=0.5, xref='paper', yref='paper', text='Both SB <br> and EB', showarrow=False, font=dict(size=14, color='black'))
    # satter_fig.add_annotation(x=0.005, y=0.55, xref='paper', yref='paper', text='Neither', showarrow=False, font=dict(size=14, color='black'))
    satter_fig.add_annotation(x=0.15, y=0.95, xref='paper', yref='paper', text='More SB like', showarrow=True, ax=0, ay=30, arrowhead=2, font=dict(size=14, color='black'))
    satter_fig.add_annotation(x=0.15, y=0.05, xref='paper', yref='paper', text='More EB like', showarrow=True, ax=0, ay=-30, arrowhead=2, font=dict(size=14, color='black'))

    satter_fig.update_layout(title_text='Scatter plot', paper_bgcolor = 'rgba(255,255,255,0.1)',
                        legend_title = '', width = 500, height = 450,
                        xaxis = dict(title = x_title, nticks = 10, autorange=True, gridcolor = 'rgba(0,0,0,0.1)', linecolor='gray', showline=False, zeroline=True, zerolinecolor='rgba(0,0,0,0.5)'),
                        yaxis = dict(title = y_title, nticks = 10, gridcolor = 'rgba(0,0,0,0.1)', linecolor='gray', showline = False, zeroline=True, zerolinecolor='rgba(0,0,0,0.5)'),
                        plot_bgcolor = 'rgba(255,255,255,.1)',
                        font =dict(size = 14, color = "black", family = "Arial"))
    return satter_fig

def plot_distributions(plotting_df, clip_number=0, type='Map max to 1'):
    if type == 'Map max to 1':
        x1 = 'sb_cont_dist_norm'
        x2 = 'eb_cont_dist_norm'
        y_title = 'Scaled density'
    elif type == 'zscore':
        x1 = 'sb_dist_z'
        x2 = 'eb_dist_z'
        y_title = 'Scaled density'
    # plot the sb and eb distributions
    dist_fig = go.Figure()
    #   get the clip data
    clip_data = plotting_df[plotting_df['clip']==clip_number]
    # create a filled area plot for the sb and eb distributions
    dist_fig.add_trace(go.Scatter(x=np.arange(len(clip_data)), y=clip_data[x1], name='Storyboard', mode='lines', fill='tozeroy',line=dict(color='orange', width=2)))
    dist_fig.add_trace(go.Scatter(x=np.arange(len(clip_data)), y=clip_data[x2], name='Event boundary', mode='lines', fill='tozeroy', line=dict(color='rgba(231,41,138, 0.35)', width=2)))
    # add the peaks to the plot
    # add vertical lines wherever the sb_peaks are 1
    max_y = max(clip_data[x1].max(), clip_data[x2].max())
    min_y = min(clip_data[x1].min(), clip_data[x2].min())
    for ix,val in enumerate(clip_data['sb_peaks'].values):
        if val:
            dist_fig.add_shape(type="line", x0=ix, y0=0, x1=ix, y1=max_y, line=dict(color="orange", width=2, dash="dash"))
    # add vertical lines wherever the eb_peaks are 1
    for ix,val in enumerate(clip_data['eb_peaks'].values):
        if val:
            dist_fig.add_shape(type="line", x0=ix, y0=0, x1=ix, y1=max_y, line=dict(color="red", width=2, dash="dash"))

    dist_fig.update_layout(title_text='Distributions', paper_bgcolor = 'rgba(255,255,255,0.1)',
                        legend_title = '', width = 600, height = 450,
                        xaxis = dict(title = "Time (ms)", nticks = 10, gridcolor = 'rgba(0,0,0,0.1)', linecolor='gray', linewidth=2),
                        yaxis = dict(title = y_title, range = [min_y*1.05, max_y*1.05], nticks = 0, gridcolor = 'rgba(0,0,0,0.1)', linecolor='gray', linewidth = 2),
                        plot_bgcolor = 'rgba(255,255,255,.1)',
                        font =dict(size = 14, color = "black", family = "Arial"))
    return dist_fig
    
# make a subplot grid of dimension 1 x n where n is the length of the sb_img_sources
# import make_subplots
def show_images(df, clip_number=0):
    clip_df = df[df['clip']==clip_number].reset_index()
    sb_df = clip_df[clip_df['sb_peaks']==1]
    eb_df = clip_df[clip_df['eb_peaks']==1]
    # create a subplot grid
    max_len = max(len(sb_df), len(eb_df))
    frames_fig = make_subplots(rows=2, cols=max_len)
    # create empty subplots
    # add images
    for col, src in enumerate(sb_df.iloc):
        frames_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="markers",
                marker_opacity=0,
                showlegend=False
            ), row=1,
            col = col+1,
        )
    for col, src in enumerate(eb_df.iloc):
        frames_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="markers",
                marker_opacity=0,
                showlegend=False
            ), row=2,
            col = col+1,
        )
    for col, rec in enumerate(sb_df.iloc):
        frames_fig.add_layout_image(
            row=1,
            col=col + 1,
            source=rec['image_path'],
            xref="x domain",
            yref="y domain",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            sizex=1,
            sizey=1,
        )
    for col, rec in enumerate(eb_df.iloc):
        frames_fig.add_layout_image(
            row=2,
            col=col + 1,
            source=rec['image_path'],
            xref="x domain",
            yref="y domain",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            sizex=1,
            sizey=1,
        )

    # turn off the grid
    frames_fig.update_xaxes(showgrid=False, showticklabels=False, showline=True, linewidth=2, linecolor='rgba(0,0,0,0.1)', mirror=True)
    frames_fig.update_yaxes(showgrid=False, showticklabels=False, showline=True, linewidth=2, linecolor='rgba(0,0,0,0.1)', mirror=True)

    # update the layout
    height=400
    frames_fig.update_layout(width=height/2*max_len, height=height)
    # create transparent background
    frames_fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return frames_fig

def show_images_in_timeline(df, clip_number=0):
    clip_df = df[df['clip']==clip_number].reset_index()
    frames_fig = make_subplots(rows=2, cols=len(clip_df), shared_yaxes=True)
    # create empty subplots
    # add images
    for col, rec in enumerate(clip_df.iloc):
        frames_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="markers",
                marker_opacity=0,
                showlegend=False
            ), row=1,
            col = col+1,
        )
    for col, rec in enumerate(clip_df.iloc):
        frames_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="markers",
                marker_opacity=0,
                showlegend=False
            ), row=2,
            col = col+1,
        )
    for col, rec in enumerate(clip_df.iloc):
        if rec['sb_peaks']==1:
            frames_fig.add_layout_image(
                row=1,
                col=col + 1,
                source=rec['image_path'],
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="middle",
                sizex=1,
                sizey=1,
            )
        else:
            frames_fig.add_layout_image(
                row=2,
                col=col + 1,
                source=rec['image_path'],
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="middle",
                sizex=1,
                sizey=1,
            )
    # turn off the grid
    frames_fig.update_xaxes(showgrid=False, showticklabels=False, showline=True, linewidth=2, linecolor='rgba(0,0,0,0.1)', mirror=True)
    frames_fig.update_yaxes(showgrid=False, showticklabels=False, showline=True, linewidth=2, linecolor='rgba(0,0,0,0.1)', mirror=True)

    # update the layout
    height=400
    frames_fig.update_layout(width=height/2*len(clip_df), height=height)
    # create transparent background
    frames_fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return frames_fig


# Initialize the app
app = dash.Dash(__name__,
    external_stylesheets = [
        {
            'href': 'https://unpkg.com/purecss@1.0.1/build/pure-min.css',
            'rel': 'stylesheet',
            'integrity': 'sha384-oAOxQR6DkCoMliIh8yFnu25d7Eq/PHS21PClpwjOTeU2jRSq11vu66rf90/cZr47',
            'crossorigin': 'anonymous'
        },
        'https://unpkg.com/purecss@1.0.1/build/grids-responsive-min.css',
        'https://unpkg.com/purecss@1.0.1/build/base-min.css',
    ],
)
app.title = 'Visualizing event boundaries and storyboards'
server = app.server

# App layout
app.layout = html.Div([
    html.Div(className='row', children='Visualizing Storyboards and Event boundaries', style={'textAlign': 'center', 'color': 'gray', 'fontSize': 30}),
    
    html.Div(className='row', 
             children=['Manipulation type', ' ',
                       dcc.RadioItems([
                            {'label': 'Map max to 1', 'value': 'Map max to 1'},
                            {'label': 'zscore', 'value': 'zscore'},
                            ], id='scaling-type', value='Map max to 1', labelStyle={'display': 'inline-block'}),           
                        dcc.Dropdown(plotting_df['clip'].unique(),
                                     placeholder="Select a clip number",
                                     id='clip_number-dropdown',
                                     ),
                        ], style={'width': '25%', 'display': 'inline-block'}),

    html.Div(className='row', 
             children=[dp.DashPlayer(id="player", url="",controls=True,
                                     style={'width': '30%', 'display': 'inline-block', 'padding-left':'0.5%'}), 
                       dcc.Graph(id='distribution-plot', style={'width': '30%', 'display': 'inline-block', 'padding-left':'2.25%'}, clear_on_unhover=True), 
                       dcc.Tooltip(id="graph-tooltip"), 
                       dcc.Graph(id='scatter-plot', style={'width': '30%', 'display': 'inline-block', 'padding-left':'2.25%'}, clear_on_unhover=True)]),
    html.Div(className='row',
                children=[html.H2('Frame visualization', style={'textAlign': 'center', 'color': 'gray', 'fontSize': 30}),
                          html.P('Storyboards (top) and Event boundaries (bottom)', style={'textAlign': 'center', 'color': 'gray', 'fontSize': 20})]),
    html.Div(className='row', 
             children=[
                       dcc.RadioItems([
                            {'label': 'Tile-view', 'value': 'Tile'},
                            {'label': 'Temporal-view', 'value': 'Timeline'},
                            ], id='viewing-type', value='Tile', labelStyle={'display': 'inline-block'}),
                       dcc.Graph(id='frame-visualization', style={'display': 'inline-block', 'width': '50vh'})])
])
@callback(
    Output('player', 'url'),
    Input('clip_number-dropdown', 'value'))
def update_player(clip_number, df=plotting_df):
    if clip_number is not None:
        clip = plotting_df[plotting_df['clip']==clip_number]['Movie+clip_number'].values[0]
        return 'https://adibuoy23.github.io/event_representations/videos/'+str(clip)+'.mp4'
    else:
        raise PreventUpdate

# Update the scatter plot
@callback(
    Output('scatter-plot', 'figure', allow_duplicate=True),
    Input('scaling-type', 'value'),
    Input('clip_number-dropdown', 'value'), prevent_initial_call=True)
def update_graph(scale_type, clip_number,df=df):
    fig = make_scatter_fig(df, type=scale_type)
    namesList = ['Storyboards', 'Eventboundaries'] + [str(i) for i in df['clip'].unique() if i!=0]
    if clip_number is not None:
        ddf = df[df['clip']==clip_number]
        if clip_number==0:
            fig.update_traces(marker=dict(size=10, opacity = 0.85, line=dict(width=2, color='black')), selector=dict(name='Storyboards'))
            fig.update_traces(marker=dict(size=10, opacity = 0.85, line=dict(width=2, color='black')), selector=dict(name='Eventboundaries'))
            for name in set(namesList) - set(['Storyboards', 'Eventboundaries']):
                fig.update_traces(marker=dict(size=10, opacity = 0.35), selector=dict(name=str(name)))
        else:
            fig.update_traces(marker=dict(size=10, opacity = 0.85, line=dict(width=2, color='black')), selector=dict(name=str(clip_number)))
            for name in set(namesList) - set([str(clip_number)]):
                fig.update_traces(marker=dict(size=10, opacity = 0.35), selector=dict(name=str(name)))

    return fig

# Update the distribution plot
@callback(
    Output('distribution-plot', 'figure'),
    Input('clip_number-dropdown', 'value'),
    Input('scaling-type', 'value')
)
def update_distribution_plot(clip_number, scale_type, df=plotting_df):
    if clip_number is not None:
        fig = plot_distributions(df, clip_number, type=scale_type)
        return fig
    else:
        raise PreventUpdate

# Update the frame visualization
@callback(
    Output('frame-visualization', 'figure'),
    Input('clip_number-dropdown', 'value'),
    Input('viewing-type', 'value'))
def update_frame_visualization(clip_number, viewing_type="Tile", df=df):
    if clip_number is not None:
        if viewing_type == 'Tile':
            fig = show_images(df, clip_number)
        elif viewing_type == 'Timeline':
            fig = show_images_in_timeline(df, clip_number)
        return fig
    else:
        raise PreventUpdate

# Update the hover display on the distribution plot
@callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("distribution-plot", "hoverData"),
    Input('clip_number-dropdown', 'value'),
)
def display_hover(hoverData, clip_number=0, df=plotting_df):
    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]
    df = df[df['clip']==clip_number]
    df_row = df.iloc[num]
    img_src = df_row['image_path']
    name = df_row['frame_number']

    children = [
        html.Div([
            html.Img(src=img_src, style={"width": "100%"}),
            html.H2(f"{name}", style={"color": "darkblue", "overflow-wrap": "break-word"}),
        ], style={'width': '200px', 'white-space': 'normal'})
    ]

    return True, bbox, children

# Highlight the peak on the distribution plot when hovering over the scatter plot
@callback(
    Output("distribution-plot", "figure", allow_duplicate=True),
    Input("scatter-plot", "hoverData"),
    Input('clip_number-dropdown', 'value'),
    Input('scaling-type', 'value'),
    prevent_initial_call=True
)
def highlight_hover(hoverData, clip_number, scale_type, df=df):
    if hoverData is None:
        fig = update_distribution_plot(clip_number, scale_type)
        return no_update
    # get the corresponding point's x and y coordinates on the scatter plot
    if scale_type == 'Map max to 1':
        col1 = 'sum_dist_scaled'
        col2 = 'diff_dist_scaled'
        sb_col = 'sb_cont_dist_norm'
        eb_col = 'eb_cont_dist_norm'
    elif scale_type == 'zscore':
        col1 = 'sum_dist_z'
        col2 = 'diff_dist_z'
        sb_col = 'sb_dist_z'
        eb_col = 'eb_dist_z'
    elif scale_type == 'regression':
        col1 = 'SB variance captured by EB'
        col2 = 'SB variance left out by EB'
        sb_col = 'sb_cont_dist'
        eb_col = 'eb_cont_dist'
    x = hoverData.get('points')[0]['x']
    y = hoverData.get('points')[0]['y']
    clip_df = df[df['clip']==clip_number]
    # find the closest point in the df to the hovered point
    closest_index = clip_df[[col1, col2]].sub([x, y]).pow(2).sum(1).idxmin()
    closest_peak_index = df.iloc[closest_index]['peak_indices']
    max_y = max(clip_df[sb_col].max(), clip_df[eb_col].max())
    min_y = min(clip_df[sb_col].min(), clip_df[eb_col].min())
    fig = update_distribution_plot(clip_number, scale_type)
    if closest_peak_index <= df['peak_indices'].max():
        # add a rectangle around the coordinate
        fig.add_shape(type="rect", x0=max(closest_peak_index-1000, 0), y0=min(0,min_y*1.05), x1=min(closest_peak_index+1000, clip_df['pdf_len'].unique()[0]), y1=max_y*1.05, line=dict(color="black", width=0.5), fillcolor="rgba(0,0,0,0.35)")
    else:
        # do nothing
        pass
    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=debug)