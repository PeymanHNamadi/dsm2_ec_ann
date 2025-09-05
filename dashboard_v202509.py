from bokeh.plotting import figure
from bokeh.models import Range1d, HoverTool, Label, ColumnDataSource
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
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

# dir = os.path.dirname(os.path.realpath(__file__))
dir = os.getcwd() 

with open(os.path.join(dir,'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

input_dict = config['input_dict']
name_map = config['name_mapping']
name_map_swap = {v: k for k, v in name_map.items()}
variables = config['output_vars']
inp_template = os.path.join(dir,'ann_inp.csv')

# Load data once at startup
print("Loading initial data...")
dfobs = pd.read_csv('obs_hist_ec.csv',index_col=0, parse_dates = ['Time'])
dfobs.replace(-2, np.nan, inplace=True)
dfinps = pd.read_csv(inp_template,index_col=0, parse_dates = ['Time'])
dfinps_global = dfinps.copy()
dfinps_original = dfinps.copy()  # Keep original for reference
dfouts = pd.read_csv('dsm2_hist_ec_output.csv',index_col=0, parse_dates = ['Time'])

# Default values
wateryear = 2014
start_date = dt.datetime(wateryear-1, 10, 1)
end_date = dt.datetime(wateryear, 9, 30)

scale_df1 = pd.read_csv(os.path.join(dir,'input_scale.csv'),
                        index_col=0, parse_dates = ['month'])
scale_df = scale_df1.copy()

# Global cache for pre-computed ANN results
PRECOMPUTED_RESULTS = {}
CACHE_DIR = os.path.join(dir, 'ann_cache')
COMPUTING_LOCK = threading.Lock()
COMPUTATION_STATUS = {}
ALL_MODELS = ['Res-LSTM', 'Res-GRU', 'LSTM', 'GRU', 'ResNet']
CURRENT_INPUT_HASH = None
AUTO_RECOMPUTE = True  # Flag to control automatic recomputation

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_input_hash(dfinp, start_date, end_date):
    """Generate hash for current input configuration"""
    df_subset = dfinp.loc[(dfinp.index > start_date) & (dfinp.index <= end_date)]
    return hashlib.md5(pd.util.hash_pandas_object(df_subset, index=True).values).hexdigest()

def get_cache_filename(station, wateryear, input_hash):
    """Generate cache filename including input hash"""
    return os.path.join(CACHE_DIR, f"{station}_{wateryear}_{input_hash[:8]}_results.pkl")

def load_cached_results(station, wateryear, input_hash):
    """Load pre-computed results from disk if available"""
    cache_file = get_cache_filename(station, wateryear, input_hash)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                if all(model in data for model in ALL_MODELS):
                    print(f"Loaded cached results for {station} WY{wateryear}")
                    return data
        except Exception as e:
            print(f"Error loading cache: {e}")
    return None

def save_cached_results(station, wateryear, input_hash, results):
    """Save pre-computed results to disk"""
    cache_file = get_cache_filename(station, wateryear, input_hash)
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved cache for {station} WY{wateryear}")
    except Exception as e:
        print(f"Error saving cache: {e}")

def precompute_single_model(station, dfinp, dfouts, model, start_date, end_date):
    """Compute results for a single model"""
    try:
        print(f"  Computing {model}...")
        targ_df, pred_df = evaluateann.run_ann(station, dfinp, dfouts, model)
        # Filter to the water year
        pred_filtered = pred_df.loc[(pred_df.index > start_date) & 
                                    (pred_df.index <= end_date)]
        targ_filtered = targ_df.loc[(targ_df.index > start_date) & 
                                    (targ_df.index <= end_date)]
        return model, (targ_filtered, pred_filtered)
    except Exception as e:
        print(f"  Error computing {model}: {e}")
        return model, (pd.DataFrame(), pd.DataFrame())

def precompute_all_models(station, dfinp, dfouts, wateryear, progress_bar=None, force=False):
    """Pre-compute results for all models for a given station and water year"""
    global CURRENT_INPUT_HASH
    
    start_date = dt.datetime(int(wateryear)-1, 10, 1)
    end_date = dt.datetime(int(wateryear), 9, 30)
    
    # Generate input hash
    input_hash = get_input_hash(dfinp, start_date, end_date)
    cache_key = f"{station}_{wateryear}_{input_hash}"
    
    # Update current input hash
    CURRENT_INPUT_HASH = input_hash
    
    # Check if already computed in memory with same inputs
    if not force and cache_key in PRECOMPUTED_RESULTS:
        print(f"Using cached results for {station} WY{wateryear}")
        return PRECOMPUTED_RESULTS[cache_key]
    
    # Try to load from disk
    if not force:
        cached = load_cached_results(station, wateryear, input_hash)
        if cached:
            PRECOMPUTED_RESULTS[cache_key] = cached
            return cached
    
    print(f"Pre-computing all models for {station} WY{wateryear}...")
    results = {}
    
    # Update status
    COMPUTATION_STATUS[cache_key] = "Computing..."
    
    # Use parallel processing for faster computation
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for model in ALL_MODELS:
            future = executor.submit(precompute_single_model, 
                                   station, dfinp, dfouts, model, 
                                   start_date, end_date)
            futures.append(future)
        
        # Collect results
        for i, future in enumerate(as_completed(futures)):
            model, result = future.result()
            results[model] = result
            if progress_bar:
                progress_bar.value = int((i + 1) / len(ALL_MODELS) * 100)
    
    # Store in memory and disk
    PRECOMPUTED_RESULTS[cache_key] = results
    save_cached_results(station, wateryear, input_hash, results)
    
    # Clear old cached results with different input hashes
    old_keys = [k for k in PRECOMPUTED_RESULTS.keys() 
                if k.startswith(f"{station}_{wateryear}") and k != cache_key]
    for k in old_keys:
        del PRECOMPUTED_RESULTS[k]
    
    COMPUTATION_STATUS[cache_key] = "Ready"
    print(f"Completed pre-computation for {station} WY{wateryear}")
    
    return results

def get_model_results(station, wateryear, model, input_hash):
    """Get pre-computed results for a specific model"""
    cache_key = f"{station}_{wateryear}_{input_hash}"
    
    if cache_key in PRECOMPUTED_RESULTS:
        if model in PRECOMPUTED_RESULTS[cache_key]:
            return PRECOMPUTED_RESULTS[cache_key][model]
    
    return pd.DataFrame(), pd.DataFrame()

class SliderGroup:
    def __init__(self, input_loc):
        sp = dict(start=0.80, end=1.20, step=0.05, value=1.00,
                  orientation='vertical', direction='rtl',
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

        self.fs_set = [self.fs10, self.fs11, self.fs12, self.fs1,
                       self.fs2, self.fs3, self.fs4, self.fs5, self.fs6,
                       self.fs7, self.fs8, self.fs9]

        self.kwargs = dict(fs10=self.fs10, fs11=self.fs11, fs12=self.fs12,
                          fs1=self.fs1, fs2=self.fs2, fs3=self.fs3, fs4=self.fs4,
                          fs5=self.fs5, fs6=self.fs6, fs7=self.fs7, fs8=self.fs8,
                          fs9=self.fs9)

def scale_inputs(inp_template, input_loc, scale_df, fs1, fs2, fs3,
                 fs4, fs5, fs6, fs7, fs8, fs9, fs10, fs11, fs12,
                 sd=None, ed=None):
    
    global dfinps_global, CURRENT_INPUT_HASH
    
    # Read original template (unscaled baseline)
    dfinps = pd.read_csv(inp_template, index_col=0, parse_dates=['Time'])
    
    # Update scale_df for this specific input
    scale_values = [fs1, fs2, fs3, fs4, fs5, fs6, fs7, fs8, fs9, fs10, fs11, fs12]
    scale_df.loc[1:12, input_loc] = scale_values
    
    # IMPORTANT FIX: Apply ALL scalings from scale_df to create complete scaled dataset
    dfinps_global = dfinps.copy()  # Start fresh from original
    
    mask = (dfinps_global.index > sd) & (dfinps_global.index < ed)
    
    # Apply scaling for ALL input locations based on current scale_df values
    for col in scale_df.columns:
        for mon in range(1, 13):
            mon_mask = mask & (dfinps_global.index.month == mon)
            if mon_mask.any() and col in dfinps_global.columns:
                dfinps_global.loc[mon_mask, col] = dfinps.loc[mon_mask, col] * scale_df.loc[mon, col]
    
    # Save scaled inputs
    inputdf = dfinps_global.loc[(dfinps_global.index > sd) & 
                                (dfinps_global.index <= ed)]
    inputdf.to_csv('ann_inputs.csv')
    
    # Check if inputs have changed
    new_hash = get_input_hash(dfinps_global, sd, ed)
    if new_hash != CURRENT_INPUT_HASH:
        print(f"Inputs changed for {input_loc}, triggering recomputation...")
        CURRENT_INPUT_HASH = new_hash
        
        # Trigger recomputation if auto-recompute is enabled
        if AUTO_RECOMPUTE and hasattr(scale_inputs, 'current_station') and hasattr(scale_inputs, 'current_wy'):
            trigger_recomputation(scale_inputs.current_station, scale_inputs.current_wy)
    
    return dfinps_global

def make_sd(wateryear):
    return dt.datetime(int(wateryear)-1, 10, 1)

def make_ed(wateryear):
    return dt.datetime(int(wateryear), 9, 30)

def make_input_plot(inp_template, dfinp, input_loc, start_date, end_date, refresh):
    dfinp_window = dfinp.loc[(dfinp.index > start_date) & 
                             (dfinp.index < end_date)]
    hist_window = dfinps_original.loc[(dfinps_original.index > start_date) & 
                                      (dfinps_original.index < end_date)]
    
    y_min = min(dfinp_window[input_loc].min(), hist_window[input_loc].min())
    y_max = max(dfinp_window[input_loc].max(), hist_window[input_loc].max()) * 1.1
    
    date_list = pd.date_range(start=start_date, periods=12, freq='MS')
    dfinp_window_avg = dfinp_window.groupby(dfinp_window.index.month).mean()
    hist_window_avg = hist_window.groupby(hist_window.index.month).mean()
    
    p = figure(title="", x_axis_type='datetime')
    p.line(source=dfinp, x='Time', y=str(input_loc), line_color='blue',
           line_dash='solid', line_width=1.5, legend_label=f'{input_loc} (scaled)')
    p.line(source=dfinps_original, x='Time', y=str(input_loc), line_color='silver',
           line_dash='solid', line_width=1, line_alpha=0.5,
           legend_label=f'{input_loc} (historical)')
    
    # Styling
    p.plot_height = 415
    p.plot_width = 700
    p.x_range = Range1d(start=start_date, end=end_date)
    p.xaxis.ticker.desired_num_ticks = 12
    p.y_range = Range1d(y_min, y_max)
    p.yaxis.axis_label = input_dict[input_loc]
    
    # Add labels
    for d in date_list:
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
    
    # Tools
    tt = [("Value:", "$y{0,0.0}"), ("Date:", "$x{%F}")]
    p.add_tools(HoverTool(tooltips=tt, formatters={'$x': 'datetime'}))
    p.toolbar.active_drag = None
    
    if input_upload.value is not None:
        input_upload.save('ann_inp.csv')
    
    return p

def make_ts_plot_ANN_fast(selected_key_stations, wateryear, model_kind, 
                         overlay_obs=False, computation_status="", _refresh=0):
    """Fast plotting using pre-computed results"""
    
    colors = itertools.cycle(palette)
    start_date = dt.datetime(int(wateryear)-1, 10, 1)
    end_date = dt.datetime(int(wateryear), 9, 30)
    
    # Get current input hash
    input_hash = get_input_hash(dfinps_global, start_date, end_date)
    cache_key = f"{selected_key_stations}_{wateryear}_{input_hash}"
    
    title_text = f'{name_map[selected_key_stations]} ({selected_key_stations})'
    if computation_status:
        title_text += f' - {computation_status}'
    
    p = figure(title=title_text, x_axis_type='datetime')
    
    outputdf = pd.DataFrame()
    historical_plotted = False
    
    # Check if results exist for current inputs
    if cache_key in PRECOMPUTED_RESULTS:
        results = PRECOMPUTED_RESULTS[cache_key]
        
        for m in model_kind:
            if m in results:
                targ_df, pred_df = results[m]
                
                # Plot historical only once
                if not historical_plotted and not targ_df.empty:
                    p.line(targ_df.index, targ_df[selected_key_stations].values,
                           line_color='black', line_width=1,
                           legend_label='Historical (DSM2 simulated)')
                    historical_plotted = True
                
                if not pred_df.empty:
                    p.line(pred_df.index, pred_df[selected_key_stations].values,
                           line_color=next(colors), line_width=1, legend_label=m)
                    outputdf[f'{selected_key_stations}_{m}'] = pred_df[selected_key_stations]
    else:
        # Show message if not ready
        p.add_layout(Label(x=start_date, y=0, 
                          text="Models need recomputation. Click 'Compute' or wait for auto-computation.",
                          text_font_size='12pt', text_color='red'))
    
    if not outputdf.empty:
        outputdf.to_csv('ann_outputs.csv')
    
    # Overlay observed data
    if overlay_obs:
        obs_filtered = dfobs.loc[(dfobs.index > start_date) & 
                                 (dfobs.index <= end_date), selected_key_stations]
        if not obs_filtered.empty:
            p.line(obs_filtered.index, obs_filtered.values,
                   line_color='red', line_width=1, line_alpha=0.75,
                   line_dash='dashed', legend_label='Historical (Observed)')
    
    # Styling
    p.plot_height = 500
    p.plot_width = 900
    p.legend.location = 'top_left'
    p.yaxis.axis_label = 'EC (uS/cm)'
    p.xaxis.axis_label = 'Date'
    p.x_range = Range1d(start=start_date, end=end_date)
    
    # Tools
    tt = [("Value:", "$y{0,0.0}"), ("Date:", "$x{%F}")]
    p.add_tools(HoverTool(tooltips=tt, formatters={'$x': 'datetime'}))
    p.toolbar.active_drag = None
    p.legend.click_policy = "hide"
    
    return p

def trigger_recomputation(station, wateryear):
    """Trigger background pre-computation with current inputs"""
    def compute():
        precompute_all_models(station, dfinps_global, dfouts, wateryear, force=True)
        # Update the plot after computation
        if hasattr(trigger_recomputation, 'refresh_callback'):
            trigger_recomputation.refresh_callback()
    
    thread = threading.Thread(target=compute)
    thread.daemon = True
    thread.start()

def listener(e1, e2, e3, e4, e5, e6):
    """Listener to detect when inputs change"""
    # This will be called when any input changes
    return None

# Initialize first station and year for pre-computation
initial_station = list(name_map.keys())[0]
initial_wy = '2014'

# Widgets
variables_w = pn.widgets.Select(name='Output Location', value=initial_station, 
                                options=name_map_swap)
model_kind_w = pn.widgets.CheckBoxGroup(
    name='ML Model Selection', value=['Res-LSTM'],
    options=ALL_MODELS,
    inline=True)

overlay_obs_w = pn.widgets.Checkbox(name='Overlay Observed Data', value=True)

yearselect_w = pn.widgets.RadioButtonGroup(
    name='WY Selector',
    options=['1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998',
             '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006',
             '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014',
             '2015', '2016', '2017', '2018', '2019', '2020', '2021'],
    value=initial_wy,
    button_type='primary')

# Progress and status indicators
progress_bar = pn.indicators.Progress(
    name='Computing models...', value=0, max=100, width=300, visible=False)
status_text = pn.pane.Markdown("**Ready**", width=300)

# Auto-recompute toggle
auto_recompute_w = pn.widgets.Checkbox(
    name='Auto-recompute on input change', value=True)

def update_auto_recompute(event):
    global AUTO_RECOMPUTE
    AUTO_RECOMPUTE = event.new

auto_recompute_w.param.watch(update_auto_recompute, 'value')

# Buttons
compute_btn = pn.widgets.Button(name='Compute', button_type='success')
clear_cache_btn = pn.widgets.Button(name='Clear Cache', button_type='warning')
refresh_btn = pn.widgets.Button(name='Refresh Plot', button_type='default', width=50)

def compute_current(event):
    """Compute for current selection and inputs"""
    progress_bar.visible = True
    progress_bar.value = 0
    status_text.object = "**Computing all models with current inputs...**"
    
    station = variables_w.value
    wy = yearselect_w.value
    
    # Store current settings
    scale_inputs.current_station = station
    scale_inputs.current_wy = wy
    
    # Run computation with current scaled inputs
    precompute_all_models(station, dfinps_global, dfouts, wy, progress_bar, force=True)
    
    progress_bar.visible = False
    status_text.object = "**✓ Computation complete!**"
    
    # FIXED: Force the plot to refresh by incrementing refresh button clicks
    refresh_btn.clicks += 1

def clear_cache_callback(event):
    """Clear all cached results"""
    global PRECOMPUTED_RESULTS, CURRENT_INPUT_HASH
    PRECOMPUTED_RESULTS.clear()
    CURRENT_INPUT_HASH = None
    
    # Clear disk cache
    if os.path.exists(CACHE_DIR):
        for file in os.listdir(CACHE_DIR):
            if file.endswith('.pkl'):
                os.remove(os.path.join(CACHE_DIR, file))
    
    status_text.object = "**Cache cleared! All saved results have been deleted.**"

compute_btn.on_click(compute_current)
clear_cache_btn.on_click(clear_cache_callback)

# Set up refresh callback for background computation
def refresh_callback():
    refresh_btn.clicks += 1
    status_text.object = "**✓ Auto-computation complete!**"

trigger_recomputation.refresh_callback = refresh_callback

# File download/upload widgets
output_download = pn.widgets.FileDownload(file='ann_outputs.csv',
                                          filename='ann_outputs.csv',
                                          label='Download ANN Output Data')
input_download = pn.widgets.FileDownload(file='ann_inputs.csv',
                                         filename='ann_inputs.csv',
                                         label='Download ANN Input Data')
input_upload = pn.widgets.FileInput(accept='.csv')

# Title and info panes
title_pane = pn.pane.Markdown('''
## DSM2 Emulator Dashboard (Optimized)
A browser-based Delta Salinity Dashboard which serves 
as the front-end user interface for the DSM2 salinity emulation machine learning models 
co-developed by the California Department of Water Resources and University of California, Davis.

**Usage:** 
1. Adjust input scalers on the left
2. Click "Compute" or wait for auto-computation
3. Select models to display on the right

**Cache System:** Results are saved to speed up repeated calculations. Use "Clear Cache" to remove all saved results.
''', background='white')

assumptions_pane = pn.pane.Markdown('''
#### References  
Qi, S.; He M.; Bai Z.; Ding Z.; Sandhu, P.; Chung, F.; Namadi, P.; 
Zhou, Y.; Hoang, R.; Tom, B.; Anderson, J.; Roh, D.M. 
Novel Salinity Modeling Using Deep Learning for the Sacramento—San
Joaquin Delta of California. Water 2022, 14, 3628. 
[https://doi.org/10.3390/w14223628](https://doi.org/10.3390/w14223628)  
''')

feedback_pane = pn.pane.Markdown('''
#### Disclaimer: this dashboard is still in beta.  
Thank you for evaluating the DSM2 Emulator Dashboard. Your feedback and suggestions are welcome. 
[Leave Feedback](https://forms.gle/C6ysGxvxwqK1XY54A)  
If you have questions, please contact Kevin He (Kevin.He@Water.ca.gov)
''', background='white')

# Bindings
sd_bnd = pn.bind(make_sd, wateryear=yearselect_w)
ed_bnd = pn.bind(make_ed, wateryear=yearselect_w)

# Store current station and year for scale_inputs
scale_inputs.current_station = initial_station
scale_inputs.current_wy = initial_wy

# When station or year changes, update stored values and trigger computation
def on_station_change(event):
    scale_inputs.current_station = event.new
    scale_inputs.current_wy = yearselect_w.value
    if AUTO_RECOMPUTE:
        status_text.object = "**Auto-computing for new station...**"
        trigger_recomputation(event.new, yearselect_w.value)

def on_year_change(event):
    scale_inputs.current_station = variables_w.value
    scale_inputs.current_wy = event.new
    if AUTO_RECOMPUTE:
        status_text.object = "**Auto-computing for new year...**"
        trigger_recomputation(variables_w.value, event.new)

variables_w.param.watch(on_station_change, 'value')
yearselect_w.param.watch(on_year_change, 'value')

# Initialize slider groups
northern_flow = SliderGroup('northern_flow')
scale_northern_flow = pn.bind(scale_inputs, scale_df=scale_df,
                              input_loc=northern_flow.input_loc, inp_template=inp_template,
                              sd=sd_bnd, ed=ed_bnd,
                              **northern_flow.kwargs)

exports = SliderGroup('exports')
scale_exp = pn.bind(scale_inputs, scale_df=scale_df,
                    input_loc=exports.input_loc, inp_template=inp_template,
                    sd=sd_bnd, ed=ed_bnd,
                    **exports.kwargs)

sjr_flow = SliderGroup('sjr_flow')
scale_sjr_flow = pn.bind(scale_inputs, scale_df=scale_df,
                         input_loc=sjr_flow.input_loc, inp_template=inp_template,
                         sd=sd_bnd, ed=ed_bnd,
                         **sjr_flow.kwargs)

sjr_vernalis_ec = SliderGroup('sjr_vernalis_ec')
scale_sjr_vernalis_ec = pn.bind(scale_inputs, scale_df=scale_df,
                                input_loc=sjr_vernalis_ec.input_loc, inp_template=inp_template,
                                sd=sd_bnd, ed=ed_bnd,
                                **sjr_vernalis_ec.kwargs)

sac_greens_ec = SliderGroup('sac_greens_ec')
scale_sac_greens_ec = pn.bind(scale_inputs, scale_df=scale_df,
                              input_loc=sac_greens_ec.input_loc, inp_template=inp_template,
                              sd=sd_bnd, ed=ed_bnd,
                              **sac_greens_ec.kwargs)

listener_bnd = pn.bind(listener,
                       e1=scale_northern_flow,
                       e2=scale_exp,
                       e3=scale_sjr_flow,
                       e4=None,
                       e5=scale_sjr_vernalis_ec,
                       e6=scale_sac_greens_ec)

# Get computation status
def get_status():
    start_date = dt.datetime(int(yearselect_w.value)-1, 10, 1)
    end_date = dt.datetime(int(yearselect_w.value), 9, 30)
    input_hash = get_input_hash(dfinps_global, start_date, end_date)
    cache_key = f"{variables_w.value}_{yearselect_w.value}_{input_hash}"
    
    if cache_key in PRECOMPUTED_RESULTS:
        return "Ready"
    return COMPUTATION_STATUS.get(cache_key, "Needs computation")

# Dashboard Layout
pn.extension(loading_spinner='dots', loading_color='silver', throttled=True)
pn.param.ParamMethod.loading_indicator = True

# Main dashboard
dash = pn.Column(
    title_pane,
    pn.Row(
        pn.Column(
            pn.pane.Markdown('### Model Controls'),
            compute_btn,
            auto_recompute_w,
            clear_cache_btn,
            progress_bar,
            status_text,
        ),
        pn.Column(
            pn.pane.Markdown('### Simulation Period (WY)'),
            yearselect_w,
        )
    ),
    pn.Row(
        pn.Column(
            pn.pane.Markdown('### ANN Inputs - Input Scaler'),
            pn.Tabs(
                ("Northern Flow",
                 pn.Column(
                     pn.Row(*northern_flow.fs_set),
                     pn.bind(make_input_plot, inp_template=dfinps,
                            dfinp=scale_northern_flow, input_loc='northern_flow',
                            start_date=sd_bnd, end_date=ed_bnd, refresh=refresh_btn))),
                
                ("Pumping",
                 pn.Column(
                     pn.Row(*exports.fs_set),
                     pn.bind(make_input_plot, inp_template=dfinps,
                            dfinp=scale_exp, input_loc='exports',
                            start_date=sd_bnd, end_date=ed_bnd, refresh=refresh_btn))),
                
                ("SJR flow",
                 pn.Column(
                     pn.Row(*sjr_flow.fs_set),
                     pn.bind(make_input_plot, inp_template=dfinps,
                            dfinp=scale_sjr_flow, input_loc='sjr_flow',
                            start_date=sd_bnd, end_date=ed_bnd, refresh=refresh_btn))),
                
                ("SJR Vernalis EC",
                 pn.Column(
                     pn.Row(*sjr_vernalis_ec.fs_set),
                     pn.bind(make_input_plot, inp_template=dfinps,
                            dfinp=scale_sjr_vernalis_ec, input_loc='sjr_vernalis_ec',
                            start_date=sd_bnd, end_date=ed_bnd, refresh=refresh_btn))),
                
                ("Sac Greens EC",
                 pn.Column(
                     pn.Row(*sac_greens_ec.fs_set),
                     pn.bind(make_input_plot, inp_template=dfinps,
                            dfinp=scale_sac_greens_ec, input_loc='sac_greens_ec',
                            start_date=sd_bnd, end_date=ed_bnd, refresh=refresh_btn))),
            ),
        ),
        
        pn.Column(
            pn.pane.Markdown('### ANN Outputs'),
            pn.Tabs(
                ('Plots',
                 pn.Column(
                     variables_w,
                     pn.bind(make_ts_plot_ANN_fast,
                            selected_key_stations=variables_w,
                            wateryear=yearselect_w,
                            model_kind=model_kind_w,
                            overlay_obs=overlay_obs_w,
                            computation_status=get_status,
                            _refresh=refresh_btn.param.clicks),  # FIXED: Added refresh trigger
                     model_kind_w,
                     overlay_obs_w,
                     pn.Row(input_download, output_download, refresh_btn)
                 )),
            )
        )
    ),
    assumptions_pane,
    feedback_pane,
)

# Start initial pre-computation
print("Starting initial pre-computation...")
trigger_recomputation(initial_station, initial_wy)

dash.servable(title="DSM2 ANN Emulator Dashboard")

if __name__ == '__main__':
    # Pre-compute for initial selection on startup
    print("Pre-computing initial models...")
    precompute_all_models(initial_station, dfinps_global, dfouts, initial_wy)
    dash.show(title="DSM2 ANN Emulator Dashboard")