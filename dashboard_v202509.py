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
# Ensure all keys in name_map are valid
name_map_swap = {v: k for k, v in name_map.items() if k and v}
# If name_map_swap is empty, create a default mapping
if not name_map_swap:
    name_map_swap = {k: k for k in name_map.keys()}
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

# Default values - Use EMMATON if available, otherwise first station
DEFAULT_STATION = 'EMMATON' if 'EMMATON' in name_map else list(name_map.keys())[0]
DEFAULT_WATERYEAR = 2014
start_date = dt.datetime(DEFAULT_WATERYEAR-1, 10, 1)
end_date = dt.datetime(DEFAULT_WATERYEAR, 9, 30)

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
IS_COMPUTING = False
COMPUTATION_THREADS = {}  # Track running threads

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
    global CURRENT_INPUT_HASH, IS_COMPUTING
    
    start_date = dt.datetime(int(wateryear)-1, 10, 1)
    end_date = dt.datetime(int(wateryear), 9, 30)
    
    # Generate input hash
    input_hash = get_input_hash(dfinp, start_date, end_date)
    cache_key = f"{station}_{wateryear}_{input_hash}"
    
    # Update current input hash
    CURRENT_INPUT_HASH = input_hash
    
    # Check if already computed in memory with same inputs - FAST PATH
    if not force and cache_key in PRECOMPUTED_RESULTS:
        print(f"Using in-memory cached results for {station} WY{wateryear}")
        IS_COMPUTING = False
        return PRECOMPUTED_RESULTS[cache_key]
    
    # Try to load from disk - SECOND FAST PATH
    if not force:
        cached = load_cached_results(station, wateryear, input_hash)
        if cached:
            PRECOMPUTED_RESULTS[cache_key] = cached
            print(f"Loaded from disk cache for {station} WY{wateryear}")
            IS_COMPUTING = False
            return cached
    
    # Only compute if necessary
    IS_COMPUTING = True
    print(f"Pre-computing all models for {station} WY{wateryear}...")
    results = {}
    
    # Update status
    COMPUTATION_STATUS[cache_key] = "Computing..."
    
    # Use parallel processing with optimal worker count
    max_workers = min(5, len(ALL_MODELS))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for model in ALL_MODELS:
            future = executor.submit(precompute_single_model, 
                                   station, dfinp, dfouts, model, 
                                   start_date, end_date)
            futures.append(future)
        
        # Collect results and update display immediately
        for i, future in enumerate(as_completed(futures)):
            model, result = future.result()
            results[model] = result
            
            # Store partial results immediately for display
            PRECOMPUTED_RESULTS[cache_key] = results
            
            if progress_bar:
                progress_bar.value = int((i + 1) / len(ALL_MODELS) * 100)
    
    # Save to disk after all models complete (non-blocking)
    def save_async():
        save_cached_results(station, wateryear, input_hash, results)
    
    save_thread = threading.Thread(target=save_async)
    save_thread.daemon = True
    save_thread.start()
    
    # Keep memory cache size reasonable
    if len(PRECOMPUTED_RESULTS) > 10:
        oldest_key = list(PRECOMPUTED_RESULTS.keys())[0]
        del PRECOMPUTED_RESULTS[oldest_key]
    
    COMPUTATION_STATUS[cache_key] = "Ready"
    IS_COMPUTING = False
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
    
    # Apply ALL scalings from scale_df to create complete scaled dataset
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
    # FIXED: Changed historical line to black with full opacity
    p.line(source=dfinps_original, x='Time', y=str(input_loc), line_color='#000000',  # Pure black
           line_dash='solid', line_width=1.2, line_alpha=1.0,  # Full opacity, slightly thicker
           legend_label=f'{input_loc} (historical)')
    
    # Styling - larger for better display with increased font sizes
    p.plot_height = 480  # Increased to accommodate rotated labels
    p.plot_width = 650   # Increased width
    p.x_range = Range1d(start=start_date, end=end_date)
    p.xaxis.ticker.desired_num_ticks = 12
    p.y_range = Range1d(y_min, y_max)
    
    # Increase axis label font sizes
    p.yaxis.axis_label = input_dict[input_loc]
    p.yaxis.axis_label_text_font_size = "16pt"  # Doubled from default (~8pt)
    p.xaxis.axis_label = "Date"
    p.xaxis.axis_label_text_font_size = "16pt"  # Doubled from default
    
    # Increase tick label font sizes
    p.yaxis.major_label_text_font_size = "14pt"  # Doubled from default (~7pt)
    p.xaxis.major_label_text_font_size = "14pt"  # Doubled from default
    p.xaxis.major_label_orientation = 45  # Rotate x-axis labels 45 degrees
    
    # Increase legend font size
    p.legend.label_text_font_size = "12pt"
    
    # Add labels - CHANGED COLOR FROM SILVER TO BLACK FOR HISTORICAL VALUES
    for d in date_list:
        lbl_scaled = Label(x=d, y=240, x_units='data', y_units='screen',
                          text=str(round(dfinp_window_avg[input_loc][d.month])),
                          text_font_size='8pt', text_color='blue', x_offset=10)
        lbl_hist = Label(x=d, y=225, x_units='data', y_units='screen',
                        text=str(round(hist_window_avg[input_loc][d.month])),
                        text_font_size='8pt', text_color='black', x_offset=10)  # CHANGED FROM 'silver' TO 'black'
        p.add_layout(lbl_scaled)
        p.add_layout(lbl_hist)
    
    annot_1 = Label(x=start_date, y=255, x_units='data', y_units='screen',
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
                         overlay_obs=False, _refresh=0):
    """Fast plotting using pre-computed results"""
    
    global IS_COMPUTING
    
    colors = itertools.cycle(palette)
    start_date = dt.datetime(int(wateryear)-1, 10, 1)
    end_date = dt.datetime(int(wateryear), 9, 30)
    
    # Get current input hash
    input_hash = get_input_hash(dfinps_global, start_date, end_date)
    cache_key = f"{selected_key_stations}_{wateryear}_{input_hash}"
    
    # Build title - check if station is in name_map
    if selected_key_stations in name_map:
        title_text = f'{name_map[selected_key_stations]} ({selected_key_stations})'
    else:
        title_text = f'{selected_key_stations}'
    
    p = figure(title=title_text, x_axis_type='datetime')
    p.title.text_font_size = "16pt"  # Increase title font size
    
    # Simple computing indicator without animation
    if IS_COMPUTING:
        computing_label = Label(
            x=start_date + (end_date - start_date) / 2,
            y=500,
            x_units='data',
            y_units='data',
            text='Computing...',
            text_font_size='20pt',
            text_color='#3498db',
            text_align='center',
            text_baseline='middle'
        )
        p.add_layout(computing_label)
    
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
                           line_color='black', line_width=1.5,
                           legend_label='Historical (DSM2 simulated)')
                    historical_plotted = True
                
                if not pred_df.empty:
                    p.line(pred_df.index, pred_df[selected_key_stations].values,
                           line_color=next(colors), line_width=1.5, legend_label=m)
                    outputdf[f'{selected_key_stations}_{m}'] = pred_df[selected_key_stations]
    elif not IS_COMPUTING:
        # Show message if not ready and not computing
        message_label = Label(
            x=start_date + (end_date - start_date) / 3,
            y=500,
            x_units='data',
            y_units='data',
            text="Click 'Compute' to generate results",
            text_font_size='14pt',
            text_color='#e74c3c',
            text_align='center'
        )
        p.add_layout(message_label)
    
    if not outputdf.empty:
        outputdf.to_csv('ann_outputs.csv')
    
    # Overlay observed data
    if overlay_obs:
        obs_filtered = dfobs.loc[(dfobs.index > start_date) & 
                                 (dfobs.index <= end_date), selected_key_stations]
        if not obs_filtered.empty:
            p.line(obs_filtered.index, obs_filtered.values,
                   line_color='red', line_width=1.5, line_alpha=0.75,
                   line_dash='dashed', legend_label='Historical (Observed)')
    
    # Styling - larger for better display with increased font sizes
    p.plot_height = 450  # Increased to accommodate rotated labels
    p.plot_width = 1200   # Increased width for full display
    p.legend.location = 'top_right'  # CHANGED from top_left to top_right
    
    # Increase axis label font sizes
    p.yaxis.axis_label = 'EC (uS/cm)'
    p.yaxis.axis_label_text_font_size = "16pt"  # Doubled from default
    p.xaxis.axis_label = 'Date'
    p.xaxis.axis_label_text_font_size = "16pt"  # Doubled from default
    
    # Increase tick label font sizes
    p.yaxis.major_label_text_font_size = "14pt"  # Doubled from default
    p.xaxis.major_label_text_font_size = "14pt"  # Doubled from default
    p.xaxis.major_label_orientation = 45  # Rotate x-axis labels 45 degrees
    
    # Increase legend font size for model names
    p.legend.label_text_font_size = "14pt"  # Increased for better visibility
    p.legend.background_fill_alpha = 0.8
    
    p.x_range = Range1d(start=start_date, end=end_date)
    
    # Tools
    tt = [("Value:", "$y{0,0.0}"), ("Date:", "$x{%F}")]
    p.add_tools(HoverTool(tooltips=tt, formatters={'$x': 'datetime'}))
    p.toolbar.active_drag = None
    p.legend.click_policy = "hide"
    
    return p

def compute_current_thread(station, wy, progress_bar, status_text, refresh_btn):
    """Computation function to run in thread"""
    global IS_COMPUTING
    
    try:
        IS_COMPUTING = True
        progress_bar.visible = True
        progress_bar.value = 0
        
        # Check if already cached first
        start_date = dt.datetime(int(wy)-1, 10, 1)
        end_date = dt.datetime(int(wy), 9, 30)
        input_hash = get_input_hash(dfinps_global, start_date, end_date)
        cache_key = f"{station}_{wy}_{input_hash}"
        
        if cache_key in PRECOMPUTED_RESULTS:
            status_text.object = "**✓ Results already cached!**"
            IS_COMPUTING = False
            progress_bar.visible = False
            refresh_btn.clicks += 1
            return
        
        status_text.object = "**Computing models...**"
        
        # Run computation with immediate updates
        precompute_all_models(station, dfinps_global, dfouts, wy, progress_bar, force=True)
        
        # Refresh display immediately after computation
        refresh_btn.clicks += 1
        
        IS_COMPUTING = False
        progress_bar.visible = False
        status_text.object = "**✓ Computation complete!**"
        
    except Exception as e:
        IS_COMPUTING = False
        progress_bar.visible = False
        status_text.object = f"**❌ Error: {str(e)}**"
        refresh_btn.clicks += 1

# Initialize widgets early
refresh_btn = pn.widgets.Button(name='Refresh', button_type='default', width=50, visible=False)

# Initialize first station and year
initial_station = DEFAULT_STATION
initial_wy = str(DEFAULT_WATERYEAR)

# Widgets with new order - INCREASED FONT SIZE FOR YEAR BUTTONS
yearselect_w = pn.widgets.RadioButtonGroup(
    name='',  # Removed label since we have title above
    options=['1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998',
             '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006',
             '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014',
             '2015', '2016', '2017', '2018', '2019', '2020', '2021'],
    value=initial_wy,
    button_type='primary',
    stylesheets=["""
    .bk-btn-group .bk-btn {
        font-size: 16px !important;
        padding: 8px 12px !important;
    }
    """]
)

variables_w = pn.widgets.Select(
    name='',  # Removed label since we have title above
    value=initial_station, 
    options=name_map_swap,
    width=200,
    margin=(10, 10, 10, 10)
)

# MODEL SELECTION WIDGET WITH INCREASED FONT SIZE
model_kind_w = pn.widgets.CheckBoxGroup(
    name='ML Model Selection', 
    value=ALL_MODELS,  # All models selected by default
    options=ALL_MODELS,
    inline=True,
    stylesheets=["""
    :host {
        --bokeh-base-font-size: 16px !important;
    }
    .bk-input-group label {
        font-size: 16px !important;
    }
    """]
)

overlay_obs_w = pn.widgets.Checkbox(
    name='Overlay Observed Data', 
    value=True,
    stylesheets=["""
    :host {
        --bokeh-base-font-size: 16px !important;
    }
    """]
)

# Progress and status indicators
progress_bar = pn.indicators.Progress(
    name='Computing models...', value=0, max=100, width=180, visible=False
)
status_text = pn.pane.Markdown("**Ready**", width=180)

# Compute button with LARGER FONT SIZE
compute_btn = pn.widgets.Button(
    name='Compute', 
    button_type='success', 
    width=180, 
    height=50,  # Increased height
    margin=(10, 10, 10, 10),
    stylesheets=["""
    .bk-btn {
        font-size: 24px !important;
        font-weight: bold !important;
    }
    """]
)

def compute_current(event):
    """Compute for current selection and inputs"""
    global IS_COMPUTING, COMPUTATION_THREADS
    
    if IS_COMPUTING:
        status_text.object = "**⚠ Already computing, please wait...**"
        return
    
    station = variables_w.value
    wy = yearselect_w.value
    
    # Cancel any existing computation thread
    thread_key = f"{station}_{wy}"
    if thread_key in COMPUTATION_THREADS and COMPUTATION_THREADS[thread_key].is_alive():
        status_text.object = "**⚠ Previous computation still running...**"
        return
    
    # Start computation in thread
    thread = threading.Thread(
        target=compute_current_thread,
        args=(station, wy, progress_bar, status_text, refresh_btn)
    )
    thread.daemon = True
    thread.start()
    COMPUTATION_THREADS[thread_key] = thread

compute_btn.on_click(compute_current)

# File download/upload widgets with LARGER FONT SIZE
output_download = pn.widgets.FileDownload(
    file='ann_outputs.csv',
    filename='ann_outputs.csv',
    label='Download Output Data',
    button_type='primary',
    height=50,  # Increased height
    stylesheets=["""
    .bk-btn {
        font-size: 24px !important;
        font-weight: bold !important;
    }
    """]
)
input_download = pn.widgets.FileDownload(
    file='ann_inputs.csv',
    filename='ann_inputs.csv',
    label='Download Input Data',
    button_type='primary',
    height=50,  # Increased height
    stylesheets=["""
    .bk-btn {
        font-size: 24px !important;
        font-weight: bold !important;
    }
    """]
)
input_upload = pn.widgets.FileInput(accept='.csv')

# Title and info panes
title_pane = pn.pane.Markdown('''
## DSM2 Salinity Emulator Dashboard
A browser-based Delta Salinity Dashboard 
as the front-end user interface for the DSM2 salinity emulation machine learning models.

**Usage:** 
1. Select Water Year
2. Modify input scales (Optional)
3. Select Location
4. Click "Compute" to generate results

**Note:** Results are cached automatically for faster access.
''', background='white')

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
If you have questions, please contact Kevin He (Kevin.He@Water.ca.gov)
''', background='white')

# Bindings
sd_bnd = pn.bind(make_sd, wateryear=yearselect_w)
ed_bnd = pn.bind(make_ed, wateryear=yearselect_w)

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

# Dashboard Layout
pn.extension(loading_spinner='dots', loading_color='silver', throttled=True)
pn.param.ParamMethod.loading_indicator = True

# Main dashboard - Single page layout with new workflow order
dash = pn.Column(
    title_pane,
    # Top row - Year selection with title - INCREASED FONT SIZE
    pn.Row(
        pn.Column(
            pn.pane.Markdown('## 1. Select Water Year', styles={'font-size': '20px'}),  # Increased font size
            yearselect_w,
            width=1400
        )
    ),
    # Second section - Input scales and controls
    pn.Row(
        # Left column - Input scales - REMOVED "Optional" and INCREASED FONT SIZE
        pn.Column(
            pn.pane.Markdown('## 2. Input Scaler', styles={'font-size': '20px'}),  # Removed Optional, increased font
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
            width=700  # Increased width to accommodate larger plot with larger fonts
        ),
        # Middle column - Location and Compute - INCREASED FONT SIZES
        pn.Column(
            pn.pane.Markdown('## 3. Select Location', styles={'font-size': '20px'}),  # Increased font size
            variables_w,
            pn.Spacer(height=20),
            pn.pane.Markdown('## 4. Click "Compute" to generate results', styles={'font-size': '20px'}),  # Increased font size
            compute_btn,
            progress_bar,
            status_text,
            width=300,
            margin=(0, 0, 0, 120)  # Adjusted margin
        ),
        # Right column - empty space
        pn.Spacer(width=280)  # Reduced from 330
    ),
    # Reduced vertical space before outputs for closer graphs
    pn.Spacer(height=40),  # Further reduced from 80px to 40px for minimal gap
    # Third section - Output graphs (moved much further down) - INCREASED FONT SIZE FOR "Outputs"
    pn.Row(
        pn.Spacer(width=50),  # Left margin
        pn.Column(
            pn.pane.Markdown('## Outputs', styles={'font-size': '24px'}),  # Increased font size for Outputs title
            pn.Tabs(
                ('Plots',
                 pn.Column(
                     pn.bind(make_ts_plot_ANN_fast,
                            selected_key_stations=variables_w,
                            wateryear=yearselect_w,
                            model_kind=model_kind_w,
                            overlay_obs=overlay_obs_w,
                            _refresh=refresh_btn.param.clicks),
                     model_kind_w,
                     overlay_obs_w,
                     pn.Row(input_download, output_download)
                 )),
            ),
            width=1300
        )
    ),
    assumptions_pane,
    feedback_pane
)

# Pre-load default station results on startup
print(f"Pre-loading {DEFAULT_STATION} {DEFAULT_WATERYEAR} results...")

# Try to load from cache first
initial_results = load_cached_results(DEFAULT_STATION, DEFAULT_WATERYEAR, 
                                     get_input_hash(dfinps_global, 
                                                   dt.datetime(DEFAULT_WATERYEAR-1, 10, 1),
                                                   dt.datetime(DEFAULT_WATERYEAR, 9, 30)))
if initial_results:
    cache_key = f"{DEFAULT_STATION}_{DEFAULT_WATERYEAR}_{get_input_hash(dfinps_global, dt.datetime(DEFAULT_WATERYEAR-1, 10, 1), dt.datetime(DEFAULT_WATERYEAR, 9, 30))}"
    PRECOMPUTED_RESULTS[cache_key] = initial_results
    print(f"✓ {DEFAULT_STATION} {DEFAULT_WATERYEAR} loaded from cache - ready for instant display!")
else:
    print(f"Computing {DEFAULT_STATION} {DEFAULT_WATERYEAR} for first time...")
    # Compute in background to not block startup
    def initial_compute():
        precompute_all_models(DEFAULT_STATION, dfinps_global, dfouts, str(DEFAULT_WATERYEAR))
        print(f"✓ {DEFAULT_STATION} {DEFAULT_WATERYEAR} initial computation complete!")
    
    init_thread = threading.Thread(target=initial_compute)
    init_thread.daemon = True
    init_thread.start()

dash.servable(title="DSM2 Salinity Emulator Dashboard")

if __name__ == '__main__':
    dash.show(title="DSM2 Salinity Emulator Dashboard")