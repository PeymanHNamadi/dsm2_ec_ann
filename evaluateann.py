import os
import sys
import pandas as pd
import tensorflow as tf
import holoviews as hv
hv.extension('bokeh')
from annutils import load_model, predict

cwd = os.getcwd()
sys.path.append(cwd)
gdrive_root_path = cwd

ndays=118
window_size=0
nwindows=0

output_stations = ['CHDMC006', 'CHSWP003', 'CHVCT000', 'OLD_MID', 'ROLD024', 'ROLD059',
                  'RSAC064', 'RSAC075', 'RSAC081', 'RSAC092', 'RSAN007', 'RSAN018',
                  'RSAN032', 'RSAN037', 'RSAN058', 'RSAN072', 'RSMKL008', 'SLCBN002',
                  'SLDUT007', 'SLMZU011', 'SLMZU025', 'SLSUS012', 'SLTRM004']

def mse_loss_masked(y_true, y_pred):
    squared_diff = tf.reduce_sum(tf.math.squared_difference(y_pred[y_true>0],y_true[y_true>0]))
    return squared_diff/(tf.reduce_sum(tf.cast(y_true>0, tf.float32))+0.01)

def run_ann(selected_key_station,dfinps,dfouts,model_kind):
    selected_output_variables = output_stations

    model_path_dict = {'Res-LSTM':'mtl_i118_residual_lstm_8_2_Tune_RSAC_RSAN_n_observed_first70',
                       'Res-GRU':'mtl_i118_residual_gru_8_2_Tune_RSAC_RSAN_n_observed_first70',
                       'ResNet':'mtl_i118_resnet_8_2_Tune_RSAC_RSAN_n_observed_first70',
                       'LSTM':'mtl_i118_lstm8_f_o1_Tune_RSAC_RSAN_n_observed_first70',
                       'MLP':'mtl_i18_d8_d2_o1_Tune_RSAC_RSAN_n_observed_first70',
                       'GRU':'mtl_i118_g8_f_o1_Tune_RSAC_RSAN_n_observed_first70'}

    model_path_prefix = model_path_dict[model_kind]

    print('Testing MTL ANN for %d stations: ' % len(selected_output_variables),end='')

    print([station.replace('target/','').replace('target','') 
            for station in selected_output_variables],end='\n')

    print('Load Model')
    annmodel = load_model(os.path.join(gdrive_root_path,'models', model_path_prefix),
                                    custom_objects={"mse_loss_masked": mse_loss_masked})
    #print(annmodel.xscaler.min_val)
    print('Predict')
    dfp=predict(annmodel.model, dfinps, annmodel.xscaler,
                            annmodel.yscaler,columns=selected_output_variables,
                            ndays=ndays,window_size=window_size,nwindows=nwindows)

    y = dfouts.loc[:,selected_key_station].copy()
    y[y<0] = float('nan')
    targ_df = pd.DataFrame(y.iloc[(ndays+nwindows*window_size-1):])
    pred_df = pd.DataFrame(dfp.loc[:,selected_key_station])
    print()
    return targ_df,pred_df


#selected_key_stations = 'RSAC092'
#dfinps = pd.read_csv('ann_inp.csv',index_col=0, parse_dates = ['Time'])
#test_df_targ,test_df_pred = run_ann(selected_key_stations,dfinps,dfouts,'Res-LSTM')
#test_df_targ.to_csv('eval_ann_targ.csv')
#test_df_pred.to_csv('eval_ann_pred.csv')


#ax = test_df_pred.plot(label = "pred")
#test_df_targ.plot(ax=ax)
#plt.show()
#print(test_df)