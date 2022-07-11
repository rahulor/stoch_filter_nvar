import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 11,
})

fontdict = dict(
        family='Computer Modern Roman',
        size=12,
        color='black')

figext = '.pdf'

def weights():
    path = 'data/weight.csv'
    df_W = pd.read_csv(path)
    W = df_W['Wout']
    maxW = max(abs(W))*1.1
    fig, (ax) = plt.subplots(1, sharex=True, figsize=(9, 2.5))
    #fig.suptitle(r'Output weights')
    ax.bar( np.arange(len(W)), W)
    ax.set_ylim([-maxW, maxW])
    ax.set_xlabel(r'index', labelpad=5, size=12)
    ax.set_ylabel(r'$W$', rotation=0, labelpad=10, size=12)
    path = 'fig/weight' + figext
    plt.savefig(path,bbox_inches='tight', dpi=300)
    plt.show()


from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import plot as offline
import plotly.io as pio
def dataset():
    outputpath = 'data/testdata.csv'
    df = pd.read_csv(outputpath)
    
    title = 'Testing performance'
    fig = make_subplots(rows=2, cols=1)
    
    input_trace = go.Scatter( {'x': df['time'], 'y': df['u'] }, showlegend=False, 
                             line=dict( color='#23CE78', width=1) 
                             )
    fig.append_trace( input_trace, row=1, col=1)
    fig.update_yaxes(title_text='input', row=1, col=1)
    
    output_trace1 = go.Scatter( {'x': df['time'], 'y': df['true'] }, name='true', showlegend=True,
                               line=dict( color='#FA3535', width=1.5) 
                               )
    output_trace2 = go.Scatter( {'x': df['time'], 'y': df['pred'] }, name='pred', showlegend=True,
                               line=dict( color='#3A87E9', width=1.2) 
                               )
    fig.append_trace( output_trace1, row=2, col=1)
    fig.append_trace( output_trace2, row=2, col=1)
    fig.update_xaxes(title_text=r'time [s]', row=2, col=1)
    fig.update_yaxes(title_text=r'output', row=2, col=1)
    
    
    fig.update_layout(height=500, width=800, title_text=title)
    fig.update_layout(legend=dict(yanchor="bottom", y=0.1, xanchor="left", x=1.01))

    #fig.show()
    #fig.write_image("fig/dataset"+ figext, scale=1.5)
    pio.write_image(fig, 'fig/dataset'+ figext, width=800, height=500)
    offline(fig, filename = 'view_plot.html', auto_open=True)
    

if __name__ == '__main__':
    weights()
    dataset()
