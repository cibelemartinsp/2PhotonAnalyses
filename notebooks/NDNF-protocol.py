# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Dataset description

# %%
# general python modules for scientific analysis
import sys, pathlib, os, itertools
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# add the physion path:
sys.path.append('/home/user/work/physion/src')
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.process_NWB import EpisodeData
from physion.analysis.behavior import population_analysis as behavior_population_analysis
from physion.dataviz.raw import plot as plot_raw
from physion.dataviz.episodes.trial_average import plot as plot_trial_average
from physion.utils import plot_tools as pt

# %%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA' , 'NDNF-WT-Dec-2022', 'NWBs') 
  
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]


# %%
#behavior_population_analysis(SESSIONS['files'])

# %% [markdown]
# ## Plotting the full time course and the average visually-evoked activity

# %% [markdown]
# ### 1) Plotting the full time course

# %%
def plot_full_time_course(data):
    fig, _ = plot_raw(data, data.tlim, 
                      settings={'Locomotion':dict(fig_fraction=1, subsampling=1, color='tab:blue'),
                                'FaceMotion':dict(fig_fraction=1, subsampling=1, color='tab:purple'),
                                #'Pupil':dict(fig_fraction=1, subsampling=1, color='tab:red'),
                                'CaImaging':dict(fig_fraction=3, subsampling=1, 
                                                 subquantity='dF/F', color='tab:green',
                                                 roiIndices=np.random.choice(range(data.nROIs),8)),
                                'CaImagingRaster':dict(fig_fraction=2, subsampling=1,
                                                       roiIndices='all',
                                                       normalization='per-line',
                                                       subquantity='dF/F')},
                                Tbar=20, figsize=(7,4))
    fig.suptitle(data.filename)
    return fig


# %% [markdown]
# ### 2) Plotting the the average visually-evoked activity

# %%
protocols = ['moving-dots', 'random-dots', 'looming-stim', 'static-patch',
             'Natural-Images-4-repeats', 'drifting-gratings']

STAT_TEST = {}
for protocol in protocols:
    # a default stat test
    STAT_TEST[protocol] = dict(interval_pre=[-1,0],
                               interval_post=[1,2],
                               test='ttest',
                               positive=True)
STAT_TEST['looming-stim']['interval_post'] = [2, 3]
STAT_TEST['drifting-gratings']['interval_post'] = [1.5, 2.5]
STAT_TEST['moving-dots']['interval_post'] = [1.5, 2.5]
STAT_TEST['random-dots']['interval_post'] = [1.5, 2.5]
STAT_TEST['static-patch']['interval_post'] = [0.5, 1.5]
    
def plot_average_visually_evoked_activity(data,
                                          roiIndex=None,
                                          with_sd=True):
    
    # prepare array for final results (averaged over sessions)
    RESULTS = {}
    for protocol in protocols:
        RESULTS[protocol] = {'significant':[], 'response':[], 'session':[]}


    fig, AX = plt.subplots(5, len(protocols),
                              figsize=(7,4.5))
    pt.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    for p, protocol in enumerate(protocols):

        episodes = EpisodeData(data, 
                               quantities=['dFoF'],
                               protocol_name=protocol,
                               verbose=False)
        varied_keys = [k for k in episodes.varied_parameters.keys() if k!='repeat']
        varied_values = [episodes.varied_parameters[k] for k in varied_keys]

        AX[0][p].annotate(protocol.replace('Natural-Images-4-repeats', 'natural-images'),
                          (0.5,1.4),
                          xycoords='axes fraction', ha='center')

        i=0
        for values in itertools.product(*varied_values):

            stim_cond = episodes.find_episode_cond(key=varied_keys, value=values)
            plot_trial_average(episodes, 
                               roiIndex=roiIndex,
                               condition=stim_cond,
                               with_stat_test=True,
                               stat_test_props=STAT_TEST[protocol],
                               with_std=True,
                               with_std_over_trials = (with_sd if (roiIndex is not None) else False),
                               with_std_over_rois = (with_sd if (roiIndex is None) else False),
                               AX=[[AX[i][p]]])

            if len(varied_keys)==1:
                AX[i][p].annotate('%s=%s' % (varied_keys[0], values[0]),
                                  (0,0), fontsize=4,
                                  rotation=90, ha='right',
                                  xycoords='axes fraction')
                

            RESULTS[protocol]['significant'].append([])
            RESULTS[protocol]['response'].append([])
            RESULTS[protocol]['session'].append([])
            i+=1

    if roiIndex is None:
        AX[-1][0].annotate('single session \n --> mean$\\pm$s.d. over n=%i ROIs' % data.nROIs, (0, 0),
                           xycoords='axes fraction')
    else:
        AX[-1][0].annotate('roi #%i \n --> mean$\\pm$s.d. over n=10 trials' % (1+roiIndex), (0, 0),
                           xycoords='axes fraction')

    pt.set_common_ylims(AX)
    for ax in pt.flatten(AX):
        ax.axis('off')
        if np.isfinite(ax.dataLim.x0):
            pt.draw_bar_scales(ax,
                               Xbar=1., Xbar_label='1s',
                               Ybar=1, Ybar_label='1$\\Delta$F/F', fontsize=7)
    pt.set_common_xlims(AX)
    
    return fig

# %% [markdown]
# ### Testing

# %%
filename = os.path.join(os.path.expanduser('~'), 'DATA' , 'Cibele', 'PV_BB_V1', 'NWBs', '2025_02_17-16-21-22.nwb') 
data = Data(filename,
            verbose=False)
data.build_dFoF(verbose=False, smoothing=1)
fig = plot_full_time_course(data)
fig.subplots_adjust(left=0.2, top=0.95)
fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'fig.png'))

# %%
roiIndices = np.random.choice(range(data.nROIs),8, replace=False)
t0 = 15*60
fig1, ax = plot_raw(data, [t0, t0+10*60],
                  settings={'Locomotion':dict(fig_fraction=1, subsampling=1, color='tab:blue'),
                            'FaceMotion':dict(fig_fraction=1, subsampling=1, color='tab:purple'),
                            #'Pupil':dict(fig_fraction=1, subsampling=1, color='tab:red'),
                            'CaImaging':dict(fig_fraction=3, subsampling=1, 
                                             subquantity='dF/F', color='tab:green',
                                             roiIndices=roiIndices),
                            'CaImagingRaster':dict(fig_fraction=2, subsampling=1,
                                                   roiIndices='all',
                                                   normalization='per-line',
                                                   subquantity='dF/F')},
                            Tbar=20, figsize=(3,5))
t0 = 17.5*60 # 4.7*60
ax.plot([t0, t0+75], [1,1], lw=4, color='lightgrey')
fig2, _ = plot_raw(data, [t0, t0+75],
                  settings={'Locomotion':dict(fig_fraction=1, subsampling=1, color='tab:blue'),
                            'FaceMotion':dict(fig_fraction=1, subsampling=1, color='tab:purple'),
                            #'Pupil':dict(fig_fraction=1, subsampling=1, color='tab:red'),
                            'CaImaging':dict(fig_fraction=3, subsampling=1, 
                                             subquantity='dF/F', color='tab:green',
                                             roiIndices=roiIndices),
                            'CaImagingRaster':dict(fig_fraction=2, subsampling=1,
                                                   roiIndices='all',
                                                   normalization='per-line',
                                                   subquantity='dF/F'),
                           'VisualStim': {'fig_fraction': 1e-3, 'color': 'black'}},
                            Tbar=1, figsize=(2.5,5))
#fig1.savefig('../figures/raw-full.svg')
#fig2.savefig('../figures/raw-zoom.svg')

# %%
fig, ax = pt.figure(figsize=(2,3))
import physion
physion.dataviz.imaging.show_CaImaging_FOV(data, NL=2, ax=ax, with_annotation=False)
#fig.savefig('../figures/FOV.svg')

# %%
t0 = 17.5*60
#np.random.seed(10)
#roiIndices = np.random.choice(range(data.nROIs),8, replace=False)

fig2, _ = plot_raw(data, [t0, t0+2*60],
                  settings={'Locomotion':dict(fig_fraction=1, subsampling=1, color='tab:blue'),
                            'FaceMotion':dict(fig_fraction=1, subsampling=1, color='tab:purple'),
                            #'Pupil':dict(fig_fraction=1, subsampling=1, color='tab:red'),
                            'CaImaging':dict(fig_fraction=3, subsampling=1, 
                                             subquantity='dF/F', color='tab:green',
                                             roiIndices=roiIndices),
                            'CaImagingRaster':dict(fig_fraction=2, subsampling=1,
                                                   roiIndices='all',
                                                   normalization='per-line',
                                                   subquantity='dF/F'),
                           'VisualStim': {'fig_fraction': 1e-3, 'color': 'black'}},
                            Tbar=1, figsize=(4,5))

# %%
fig = plot_average_visually_evoked_activity(data, roiIndex=1, with_sd=True)
#fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'fig.png'))

# %%
fig = plot_average_visually_evoked_activity(data, roiIndex=2, with_sd=True)
#fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'fig.png'))

# %%
fig = plot_average_visually_evoked_activity(data)
#fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'fig.png'))

# %%

# %%
if 'yann' in os.path.expanduser('~'):
    datafolder = os.path.join(os.path.expanduser('~'), 'CURATED' , 'NDNF-December-2022') # for yann
else: # means baptiste
    datafolder = os.path.join(os.path.expanduser('~'), 'Documents', 'ICMProjet','modulation-V1-processing','data','NDNF-December-2022')
    
FILENAMES = ['2022_12_14-13-27-41.nwb',
             '2022_12_15-18-13-25.nwb',
             '2022_12_15-18-49-40.nwb',
             '2022_12_16-11-00-09.nwb',
             '2022_12_16-12-03-30.nwb',
             '2022_12_16-12-47-57.nwb',
             '2022_12_16-13-40-07.nwb',
             '2022_12_16-14-29-38.nwb',
             '2022_12_20-11-49-18.nwb',
             '2022_12_20-12-31-08.nwb',
             '2022_12_20-14-08-45.nwb']

# %% [markdown]
# ### Single page for a single session

# %%
