from bokeh.plotting import figure
from bokeh.models import Range1d, HoverTool, Label, CustomJS
from bokeh.models.formatters import PrintfTickFormatter
import panel as pn
import pandas as pd
import numpy as np
import os
import evaluateann
import datetime as dt
from panel.widgets import FloatSlider as fs
import itertools
from bokeh.palettes import Set2_5 as palette
import yaml

dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir,'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

input_dict = config['input_dict']
name_map = config['name_mapping']
name_map_swap = {v: k for k, v in name_map.items()}
variables = config['output_vars']
inp_template = os.path.join(dir,'ann_inp.csv')
dfobs = pd.read_csv('obs_hist_ec.csv',index_col=0, parse_dates = ['Time'])
dfobs.replace(-2, np.nan, inplace=True)
dfinps = pd.read_csv(inp_template,index_col=0, parse_dates = ['Time'])
dfinps_global = dfinps.copy()
dfouts = pd.read_csv('dsm2_hist_ec_output.csv',index_col=0, parse_dates = ['Time'])
wateryear = 2014
start_date = dt.datetime(wateryear-1, 10, 1)
end_date = dt.datetime(wateryear, 9, 30)

scale_df1 =pd.read_csv(os.path.join(dir,'input_scale.csv'),
                       index_col=0, parse_dates = ['month'])
scale_df = scale_df1.copy()

class SliderGroup:
    def __init__(self,input_loc):
        sp = dict(start=0.80,  end=1.20, step=0.05, value=1.00,
                  orientation = 'vertical',direction ='rtl',
                  margin=3, height=100,
                  format=PrintfTickFormatter(format='%.2f'))
        self.input_loc = input_loc
        self.fs1 = fs(name='Jan', **sp)
        self.fs2 = fs(name='Feb', **sp)
        self.fs3 = fs(name='Mar', **sp)
        self.fs4 = fs(name='Apr', **sp)
        self.fs5 = fs(name='May', **sp)
        self.fs6 = fs(name='Jun', **sp)
        self.fs7 = fs(name='Jul', **sp)
        self.fs8 = fs(name='Aug', **sp)
        self.fs9 = fs(name='Sep', **sp)
        self.fs10 = fs(name='Oct', **sp)
        self.fs11 = fs(name='Nov', **sp)
        self.fs12 = fs(name='Dec', **sp)

        self.fs_set=[self.fs10,self.fs11,self.fs12,self.fs1,
                     self.fs2,self.fs3,self.fs4,self.fs5,self.fs6,
                     self.fs7,self.fs8,self.fs9]

        self.kwargs = dict(fs10=self.fs10,fs11=self.fs11,fs12=self.fs12,
                     fs1=self.fs1,fs2=self.fs2,fs3=self.fs3,fs4=self.fs4,
                     fs5=self.fs5,fs6=self.fs6,fs7=self.fs7,fs8=self.fs8,
                     fs9=self.fs9,)

def scale_inputs(inp_template,input_loc,scale_df,fs1,fs2,fs3,
                 fs4,fs5,fs6,fs7,fs8,fs9,fs10,fs11,fs12,
                 sd = None, ed = None):
                 
    global dfinps_global
    dfinps = pd.read_csv(inp_template,index_col=0, parse_dates = ['Time'])

    scale_df.loc[1,input_loc] = fs1
    scale_df.loc[2,input_loc] = fs2
    scale_df.loc[3,input_loc] = fs3
    scale_df.loc[4,input_loc] = fs4
    scale_df.loc[5,input_loc] = fs5
    scale_df.loc[6,input_loc] = fs6
    scale_df.loc[7,input_loc] = fs7
    scale_df.loc[8,input_loc] = fs8
    scale_df.loc[9,input_loc] = fs9
    scale_df.loc[10,input_loc] = fs10
    scale_df.loc[11,input_loc] = fs11
    scale_df.loc[12,input_loc] = fs12

    for mon in scale_df.index:
        dfmod = dfinps.loc[(dfinps.index.month == mon) &
                           (dfinps.index>sd) &
                           (dfinps.index<ed),
                           input_loc]*scale_df.loc[mon,input_loc]

        dfinps_global.update(dfmod, overwrite=True)

    inputdf = dfinps_global.loc[(dfinps_global.index > sd) &
                                (dfinps_global.index <= ed)]
    inputdf.to_csv('ann_inputs.csv')
    return dfinps_global

def make_sd(wateryear):
    start_date = dt.datetime(int(wateryear)-1, 10, 1)
    return start_date
def make_ed(wateryear):
    end_date = dt.datetime(int(wateryear), 9, 30)
    return end_date

def make_input_plot(inp_template,dfinp,input_loc,start_date,end_date,refresh):
    refresh = refresh
    # Determine the min and max y-axis limits from given start and end dates
    dfinp_window = dfinp.loc[(dfinp.index>start_date) &
                             (dfinp.index<end_date)]
    hist_window = inp_template.loc[(inp_template.index>start_date) &
                             (inp_template.index<end_date)]

    y_min = dfinp_window[input_loc].min()
    y_max = dfinp_window[input_loc].max()*1.5
    date_list = pd.date_range(start = start_date, periods=12, freq='MS')
    dfinp_window_avg = dfinp_window.groupby(dfinp_window.index.month).mean()
    hist_window_avg = hist_window.groupby(hist_window.index.month).mean()

    p = figure(title = "",x_axis_type='datetime')
    p.line(source = dfinp,x='Time',y=str(input_loc), line_color = 'blue',
           line_dash = 'solid', line_width=1.5, legend_label=f'{input_loc} (scaled)')
    p.line(source = inp_template,x='Time',y=str(input_loc), line_color = 'silver',
           line_dash = 'solid', line_width=1, line_alpha = 0.5,
           legend_label=f'{input_loc} (historical)')
    
    # Styling attributes.
    p.plot_height = 415
    p.plot_width = 700
    p.x_range = Range1d(start=start_date, end=end_date)
    p.xaxis.ticker.desired_num_ticks = 12
    p.y_range = Range1d(y_min,y_max)
    p.yaxis.axis_label = input_dict[input_loc]

    # Add data labels.
    for d in date_list:
        #print(d)
        lbl_scaled = Label(x=d, y=290, x_units='data', y_units='screen',
                        text=str(round(dfinp_window_avg[input_loc][d.month])),
                        text_font_size='8pt', text_color='blue', x_offset=10)
        lbl_hist = Label(x=d, y=275, x_units='data', y_units='screen',
                        text=str(round(hist_window_avg[input_loc][d.month])),
                        text_font_size='8pt', text_color='silver', x_offset=10)
        p.add_layout(lbl_scaled)
        p.add_layout(lbl_hist)
    
    annot_1 = Label(x=start_date, y=305, x_units='data', y_units='screen',
                    text='Monthly Average (cfs):',
                    text_font_size='10pt', text_color='black', x_offset=10)
    p.add_layout(annot_1)

    # Tools and tooltips.
    tt = [
    ("Value:", "$y{0,0.0}"),
    ("Date:", "$x{%F}"),
    ]

    p.add_tools(HoverTool(
        tooltips = tt,
        formatters = {'$x':'datetime'}
    ))
    p.toolbar.active_drag = None
    if input_upload.value is not None:
        input_upload.save('ann_inp.csv')
    return p

def make_ts_plot_ANN(selected_key_stations,dfinp,start_date,end_date,
                     refresh,listener,model_kind,overlay_obs=False):
    
    colors = itertools.cycle(palette)

    refresh = refresh
    listener = listener
    p = figure(title = f'{name_map[selected_key_stations]} ({selected_key_stations})',
               x_axis_type='datetime')
    outputdf = pd.DataFrame()
    for m in model_kind:
        targ_df,pred_df = evaluateann.run_ann(selected_key_stations,dfinp,dfouts,m)
        p.line(source = targ_df,x='Time',y=str(selected_key_stations),
            line_color = 'black', line_width=1, legend_label='Historical (DSM2 simulated)')
        p.line(source = pred_df, x='Time', y=str(selected_key_stations),
            line_color = next(colors), line_width=1, legend_label=m)
        outputdf[f'{selected_key_stations}_{m}'] = pred_df
    outputdf = outputdf.loc[(outputdf.index > start_date) & (outputdf.index <= end_date)]
    outputdf.to_csv('ann_outputs.csv')

    # Overlay CDEC observed historical data.
    if overlay_obs:
        p.line(source = dfobs,x='Time',y=str(selected_key_stations),
        line_color = 'red', line_width=1,
        line_alpha=0.75,
        line_dash = 'dashed',
        legend_label='Historical (Observed)')

    # Styling attributes.
    p.plot_height = 500
    p.plot_width = 900
    p.legend.location = 'top_left'
    p.yaxis.axis_label = 'EC (uS/cm)'
    p.xaxis.axis_label = 'Date'

    p.x_range = Range1d(start=start_date, end=end_date)

    # Tools and tooltips.
    tt = [
    ("Value:", "$y{0,0.0}"),
    ("Date:", "$x{%F}"),
    ]

    p.add_tools(HoverTool(
        tooltips = tt,
        formatters = {'$x':'datetime'}
    ))
    p.toolbar.active_drag = None
    p.legend.click_policy="hide"
    
    return p

def listener(e1,e2,e3,e4,e5,e6):
    e1 = e1
    e2 = e2
    e3 = e3
    e4 = e4
    e5 = e5
    e6 = e6
    return None

# Widgets
variables_w = pn.widgets.Select(name='Output Location', options = name_map_swap)
model_kind_w = pn.widgets.CheckBoxGroup(
                    name='ML Model Selection', value = ['Res-LSTM'],
                    options = ['Res-LSTM','Res-GRU','LSTM', 'GRU', 'ResNet'],
                    inline=True)

overlay_obs_w =  pn.widgets.Checkbox(name='Overlay Observed Data', value = True)

yearselect_w = pn.widgets.RadioButtonGroup(name='WY Selector',
                options=['1991','1992','1993','1994',
                         '1995','1996','1997','1998','1999','2000',
                         '2001','2002','2003','2004','2005','2006',
                         '2007','2008','2009','2010','2011','2012',
                         '2013','2014', '2015','2016','2017','2018',
                         '2019','2020','2021'], 
                value = '2014',
                button_type='primary')

run_btn = pn.widgets.Button(name='Run ANN', button_type='primary')
train_btn = pn.widgets.Button(name='Train ANN', button_type='primary')
refresh_btn = pn.widgets.Button(name='Refresh Plot', button_type='default',width=50)
output_download = pn.widgets.FileDownload(file='ann_outputs.csv',
                                        filename='ann_outputs.csv',
                                        label = 'Download ANN Output Data')
input_download = pn.widgets.FileDownload(file='ann_inputs.csv',
                                        filename='ann_inputs.csv',
                                        label = 'Download ANN Input Data')
input_upload = pn.widgets.FileInput(accept='.csv')

title_pane = pn.pane.Markdown('''
## DSM2 Emulator Dashboard
A browser-based Delta Salinity Dashboard which serves 
as the front-end user interface for the DSM2 salinity emulation machine learning models 
co-developed by the California Department of Water Resources and University of California, Davis.​

''',background='white')

assumptions_pane = pn.pane.Markdown('''
#### References  
Qi, S.; He M.; Bai Z.; Ding Z.; Sandhu, P.; Chung, F.; Namadi, P.; 
Zhou, Y.; Hoang, R.; Tom, B.; Anderson, J.; Roh, D.M. 
Novel Salinity Modeling Using Deep Learning for the Sacramento—San
Joaquin Delta of California. Water 2022, 14, 3628. 
[https://doi.org/10.3390/w14223628](https://doi.org/10.3390/w14223628)  
Qi, S.; He, M.; Bai, Z.; Ding, Z.; Sandhu, P.; Zhou, Y.; Namadi, P.; 
Tom, B.; Hoang, R.; Anderson, J.
 Multi-Location Emulation of a Process-Based Salinity Model Using Machine Learning. Water 2022, 14, 2030. 
[https://doi.org/10.3390/w14132030](https://doi.org/10.3390/w14132030)  
Qi, S.; He, M.; Hoang, R.; Zhou, Y.; Namadi, P.; Tom, B.;
Sandhu, P.; Bai, Z.; Chung, F.; Ding, Z.; et al. 
Salinity Modeling Using Deep Learning with Data Augmentation and Transfer Learning. Water 2023, 15, 2482. 
[https://doi.org/10.3390/w15132482](https://doi.org/10.3390/w15132482)
''')

feedback_pane = pn.pane.Markdown('''
#### Disclaimer: this dashboard is still in beta.  
Thank you for evaluating the DSM2 Emulator Dashboard. Your feedback and suggestions are welcome. 
[Leave Feeback](https://forms.gle/C6ysGxvxwqK1XY54A)  
If you have questions, please contact Kevin He (Kevin.He@Water.ca.gov)
''',background='white')

# Bindings.
sd_bnd = pn.bind(make_sd,wateryear = yearselect_w)
ed_bnd = pn.bind(make_ed,wateryear = yearselect_w)

northern_flow = SliderGroup('northern_flow')
scale_northern_flow = pn.bind(scale_inputs,scale_df = scale_df,
                           input_loc = northern_flow.input_loc,inp_template = inp_template,
                           sd = sd_bnd, ed = ed_bnd,
                           **northern_flow.kwargs)

exports = SliderGroup('exports')
scale_exp = pn.bind(scale_inputs,scale_df = scale_df,
                           input_loc = exports.input_loc,inp_template = inp_template,
                           sd = sd_bnd, ed = ed_bnd,
                           **exports.kwargs)

sjr_flow = SliderGroup('sjr_flow')
scale_sjr_flow = pn.bind(scale_inputs,scale_df = scale_df,
                           input_loc = sjr_flow.input_loc,inp_template = inp_template,
                           sd = sd_bnd, ed = ed_bnd,
                           **sjr_flow.kwargs)

#net_delta_cu = SliderGroup('net_delta_cu')
#scale_net_delta_cu = pn.bind(scale_inputs,scale_df = scale_df,
#                           input_loc = net_delta_cu.input_loc,inp_template = inp_template,
#                           **net_delta_cu.kwargs)

sjr_vernalis_ec = SliderGroup('sjr_vernalis_ec')
scale_sjr_vernalis_ec = pn.bind(scale_inputs,scale_df = scale_df,
                           input_loc = sjr_vernalis_ec.input_loc,inp_template = inp_template,
                           sd = sd_bnd, ed = ed_bnd,
                           **sjr_vernalis_ec.kwargs)

sac_greens_ec = SliderGroup('sac_greens_ec')
scale_sac_greens_ec = pn.bind(scale_inputs,scale_df = scale_df,
                           input_loc = sac_greens_ec.input_loc,inp_template = inp_template,
                           sd = sd_bnd, ed = ed_bnd,
                           **sac_greens_ec.kwargs)


listener_bnd = pn.bind(listener,
                       e1 = scale_northern_flow,
                       e2 = scale_exp,
                       e3 = scale_sjr_flow,
                       #e4 = scale_net_delta_cu,
                       e4 = None,
                       e5 = scale_sjr_vernalis_ec,
                       e6 = scale_sac_greens_ec)

# Dashboard Layout
pn.extension(loading_spinner='dots', loading_color='silver')
pn.param.ParamMethod.loading_indicator = True

dash = pn.Column(title_pane,
                 pn.pane.Markdown('### Simulation Period (WY)'),
                 yearselect_w,pn.Row(

    pn.Column(pn.pane.Markdown('### ANN Inputs - Input Scaler'),
            
            pn.Tabs(
                ("Northern Flow",
                pn.Column(
                pn.Row(*northern_flow.fs_set),
                pn.bind(make_input_plot,inp_template = dfinps,
                    dfinp=scale_northern_flow,input_loc='northern_flow',
                    start_date=sd_bnd,end_date=ed_bnd,refresh=refresh_btn))),

                ("Pumping",
                pn.Column(
                pn.Row(*exports.fs_set),
                pn.bind(make_input_plot,inp_template = dfinps,
                    dfinp=scale_exp,input_loc='exports',
                    start_date=sd_bnd,end_date=ed_bnd,refresh=refresh_btn))),

                ("SJR flow",
                pn.Column(
                pn.Row(*sjr_flow.fs_set),
                pn.bind(make_input_plot,inp_template = dfinps,
                    dfinp=scale_sjr_flow,input_loc='sjr_flow',
                    start_date=sd_bnd,end_date=ed_bnd,refresh=refresh_btn))),

#                ("Net Delta Consumptive Use",
#                pn.Column(
#                pn.Row(*net_delta_cu.fs_set),
#                pn.bind(make_input_plot,inp_template = dfinps,
#                        dfinp=scale_net_delta_cu,input_loc='net_delta_cu',
#                    start_date=sd_bnd,end_date=ed_bnd))),

                ("SJR Vernalis EC",
                pn.Column(
                pn.Row(*sjr_vernalis_ec.fs_set),
                pn.bind(make_input_plot,inp_template = dfinps,
                    dfinp=scale_sjr_vernalis_ec,input_loc='sjr_vernalis_ec',
                    start_date=sd_bnd,end_date=ed_bnd,refresh=refresh_btn))),

                ("Sac Greens EC",
                pn.Column(
                pn.Row(*sac_greens_ec.fs_set),
                pn.bind(make_input_plot,inp_template = dfinps,
                    dfinp=scale_sac_greens_ec,input_loc='sac_greens_ec',
                    start_date=sd_bnd,end_date=ed_bnd,refresh=refresh_btn))),

                #("DXC",
                #pn.Column()),   

            ),
        #pn.Row(input_upload)      
    ),

    pn.Column(pn.pane.Markdown('### ANN Outputs'),
    pn.Tabs(
        ('Plots',
        pn.Column(
            variables_w,
            
            pn.bind(make_ts_plot_ANN,
                selected_key_stations=variables_w,
                dfinp = dfinps_global,
                start_date = sd_bnd,
                end_date = ed_bnd,
                refresh=refresh_btn, 
                listener = listener_bnd,
                model_kind = model_kind_w,
                overlay_obs = overlay_obs_w,
            ),
            model_kind_w,overlay_obs_w,
            pn.Row(input_download,output_download,refresh_btn)
            
        )),
    )
    )
),
assumptions_pane,
feedback_pane,
)

dash.servable(title = "DSM2 ANN Emulator Dashboard")


if __name__ == '__main__':
    dash.show(title = "DSM2 ANN Emulator Dashboard")