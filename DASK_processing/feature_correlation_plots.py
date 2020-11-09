import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import plotly as pl

data_path = '/media/max/HDD3/DNS_Data/Planar/NX512/UPRIME5/postProcess_DNN/ALL/filter_width_16_DNN_UPRIME5_train.parquet'

TARGET='omega_DNS_filtered'
FEATURES = [ 'c_bar', 'omega_model_planar',
       'U_bar', 'V_bar', 'W_bar', 'grad_c_x_LES', 'grad_c_y_LES',
       'grad_c_z_LES', 'grad_U_x_LES', 'grad_V_x_LES', 'grad_W_x_LES',
       'grad_U_y_LES', 'grad_V_y_LES', 'grad_W_y_LES', 'grad_U_z_LES',
       'grad_V_z_LES', 'grad_W_z_LES', 'UP_delta', 'SGS_flux', 'Delta_LES',
       'mag_grad_c', 'mag_U', 'sum_c', 'sum_U', 'sum_grad_U', 'mag_grad_U',
       'lambda_1', 'lambda_3',]

ALL_Q = FEATURES.copy()
ALL_Q.insert(0,TARGET)

# load data
data_df = pd.read_parquet(data_path).sample(frac=0.05)

#%%
print(data_df.columns)

#%%
# heat map
plt.figure(figsize=(20,20))

sb.heatmap(data_df[ALL_Q].corr(), annot = False, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
plt.title('Heat map of Quantities of interest')
plt.savefig('corr_plots/heatmap_FEATURES.png')

#%%
for f in FEATURES:

    print(f)
    plt.figure(figsize=(10,8))
    plt.title(f+' vs. omega_DNS_filtered')
    #plt.tight_layout()
    plt.scatter(data_df[f].values,data_df[TARGET].values,s=0.2,c=data_df['c_bar'].values,cmap='jet')
    plt.ylabel(TARGET)
    plt.xlabel(f)
    cbar= plt.colorbar()
    cbar.set_label('c_bar')

    plt.savefig('corr_plots/%s_corr.png' % f)
    plt.close()


#plt.show()

