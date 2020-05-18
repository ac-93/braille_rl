import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from cycler import cycler
sns.set(style="darkgrid")


def plot_error_band(axs, x_data, y_data, min, max, data_name, title=None, colour=None, error_band=False, x_label='Epoch'):

    axs.plot(x_data, y_data, color=colour, alpha=1.0)

    if error_band:
        axs.fill_between(x_data, min, max, color=colour, alpha=0.25)

    axs.set(xlabel=x_label, ylabel=data_name)
    axs.set_xlim([0, len(x_data)-1])
    axs.set_ylim([-0, 1.1])
    axs.plot([0, len(x_data)+1], [1, 1], 'k-', lw=1, dashes=[2, 2], label='_nolegend_') # ref line
    axs.set_title(title)

    for item in ([axs.title]):
        item.set_fontsize(20)

    for item in ([axs.xaxis.label, axs.yaxis.label]):
        item.set_fontsize(16)

    for item in  axs.get_xticklabels() + axs.get_yticklabels():
        item.set_fontsize(12)

def plot_progress(progess_file, show_plot=True):
    fig, axs = plt.subplots(1, 2, figsize=(18,6))

    data = pd.read_csv(progess_file, sep="\t")
    data_len = len(data)


    plot_error_band(axs[0], data['Epoch'], data['AverageEpRet'],        data['MinEpRet'],     data['MaxEpRet'],     'Episode Return',          colour='r' )
    plot_error_band(axs[1], data['Epoch'], data['AverageTestEpRet'],    data['MinTestEpRet'], data['MaxTestEpRet'], 'Test Episode Return',     colour='b' )

    if show_plot:
        plt.show()
    fig.savefig(os.path.join(os.path.dirname(progess_file), 'training_curves.png'), dpi=320, pad_inches=0.01, bbox_inches='tight')


# get sub dirs in folder
def get_immediate_subdirectories(a_dir):
    return sorted([name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))])

def plot_average_seeds(progress_dir, show_plot=True, axs=None):

    if show_plot:
        fig, axs = plt.subplots(1, 2, figsize=(18,6))

    sub_dirs = get_immediate_subdirectories(progress_dir)

    # get paths to progress files of sub dirs
    progress_files = []
    dataframes = []
    for sub_dir in sub_dirs:
        for filename in os.listdir(os.path.join(progress_dir, sub_dir)):
            if 'progress' in filename:
                progress_file = os.path.join(progress_dir, sub_dir, filename)
                progress_files.append(progress_file)
                dataframes.append(pd.read_csv(progress_file, sep="\t"))

    # collect relevant stuff in np array
    average_train_returns = np.zeros( (len(dataframes[0]), len(dataframes)) )
    average_test_returns = np.zeros( (len(dataframes[0]), len(dataframes)) )
    for (i,dataframe) in enumerate(dataframes):
        average_train_returns[:, i] = dataframe['AverageEpRet'].values
        average_test_returns[:, i] = dataframe['AverageTestEpRet'].values

    # plot train and test with error bands
    plot_error_band(axs[0],
                    range(len(dataframes[0])),
                    np.mean(average_train_returns, axis=1),
                    np.min(average_train_returns, axis=1),
                    np.max(average_train_returns, axis=1),
                    'Episodic Return',
                    colour='r' if show_plot else None,
                    error_band=True if show_plot else None)

    plot_error_band(axs[1],
                    range(len(dataframes[0])),
                    np.mean(average_test_returns, axis=1),
                    np.min(average_test_returns, axis=1),
                    np.max(average_test_returns, axis=1),
                    'Episodic Return',
                    colour='b' if show_plot else None,
                    error_band=True if show_plot else None)

    if show_plot:
        plt.show()
        fig.savefig(os.path.join(os.path.dirname(progress_dir), 'seeded_training_curves.png'), dpi=320, pad_inches=0.01, bbox_inches='tight')


if __name__ == '__main__':
    # standard plot both training and testing progress

    # # robot
    # progess_file = 'saved_models/robot/discrete/arrows/dd_dqn/dd_dqn_s10/progress.txt'
    # progess_file = 'saved_models/robot/discrete/arrows/disc_sac/disc_sac_s10/progress.txt'
    # progess_file = 'saved_models/robot/discrete/alphabet/dd_dqn/dd_dqn_s10/progress.txt'
    # progess_file = 'saved_models/robot/discrete/alphabet/disc_sac/disc_sac_s10/progress.txt'
    # progess_file = 'saved_models/robot/continuous/arrows/cont_sac/cont_sac_s10/progress.txt'
    # plot_progress(progess_file)

    # # sim
    # progess_file = 'saved_models/sim/discrete/arrows/dd_dqn/dd_dqn_s1/progress.txt'
    # progess_file = 'saved_models/sim/discrete/arrows/disc_sac/disc_sac_s1/progress.txt'
    # progess_file = 'saved_models/sim/discrete/alphabet/dd_dqn/dd_dqn_s1/progress.txt'
    # progess_file = 'saved_models/sim/discrete/alphabet/disc_sac/disc_sac_s1/progress.txt'
    # progess_file = 'saved_models/sim/continuous/arrows/td3/td3_s1/progress.txt'
    # progess_file = 'saved_models/sim/continuous/arrows/cont_sac/cont_sac_s1/progress.txt'
    # progess_file = 'saved_models/sim/continuous/alphabet/td3/td3_s1/progress.txt'
    progess_file = 'saved_models/sim/continuous/alphabet/cont_sac/cont_sac_s1/progress.txt'
    plot_progress(progess_file)

    # plot average progress plots over seeded runs
    # progess_dir = 'saved_models/sim/continuous/arrows/cont_sac'
    # plot_average_seeds(progess_dir)
