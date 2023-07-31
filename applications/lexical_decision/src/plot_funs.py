import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm.notebook import tqdm
from scipy import stats
from sklearn.metrics import r2_score

def true_vs_estimated(ground_truths, post_means, post_sds, param_names, dpi=300,
                      figsize=(20, 4), show=True, filename=None, font_size=12, n_drift=4):
    """ Plots a scatter plot with abline of the estimated posterior means vs true values.
    Parameters
    ----------
    ground_truths: np.array of shape (batch_size, n_time_points, param_dim)
        Array of true parameters.
    micro_samples: np.array of shape (batch_size, n_time_points, param_dim)
        Array of estimated parameters.
    param_names: list(str)
        List of parameter names for plotting.
    dpi: int, default:300
        Dots per inch (dpi) for the plot.
    figsize: tuple(int, int), default: (20,4)
        Figure size.
    show: boolean, default: True
        Controls if the plot will be shown
    filename: str, default: None
        Filename if plot shall be saved
    font_size: int, default: 12
        Font size
    """
    
    # Plot settings
    plt.rcParams['font.size'] = font_size
    cm = plt.cm.get_cmap('inferno')

    # Determine n_subplots dynamically
    n_row = int(np.ceil(len(param_names) / 6))
    n_col = int(np.ceil(len(param_names) / n_row))

    # Initialize figure
    f, axarr = plt.subplots(n_row, n_col, figsize=figsize, gridspec_kw={'width_ratios': np.hstack([[1]*n_drift, [1, 1.2]])})
    if n_row > 1:
        axarr = axarr.flat
        
    # --- Plot true vs estimated posterior means on a single row --- #
    for j in range(len(param_names)):
        
        # Plot analytic vs estimated
        if j == len(param_names) - 1:
            img = axarr[j].scatter(x=post_means[:, j], y=ground_truths[:, j], c=post_sds[:, j], alpha=0.6, cmap=cm)
            f.colorbar(img, ax=axarr[j])
        else:
            axarr[j].scatter(x=post_means[:, j], y=ground_truths[:, j], c=post_sds[:, j], alpha=0.6, cmap=cm)

        # get axis limits and set equal x and y limits
        lower_lim = min(axarr[j].get_xlim()[0], axarr[j].get_ylim()[0])
        upper_lim = max(axarr[j].get_xlim()[1], axarr[j].get_ylim()[1])
        axarr[j].set_xlim((lower_lim, upper_lim))
        axarr[j].set_ylim((lower_lim, upper_lim))
        axarr[j].plot(axarr[j].get_xlim(), axarr[j].get_xlim(), '--', color='black')

        
        # Compute NRMSE
        rmse = np.sqrt(np.mean( (post_means[:, j] - ground_truths[:, j])**2 ))
        nrmse = rmse / (ground_truths[:, j].max() - ground_truths[:, j].min())
        axarr[j].text(0.1, 0.9, 'NRMSE={:.3f}'.format(nrmse),
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform=axarr[j].transAxes,
                     size=10)
        
        # Compute R2
        r2 = r2_score(ground_truths[:, j], post_means[:, j])
        axarr[j].text(0.1, 0.8, '$R^2$={:.3f}'.format(r2),
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform=axarr[j].transAxes, 
                     size=10)
        
        if j == 0:
            # Label plot
            axarr[j].set_xlabel('Estimated')
            axarr[j].set_ylabel('True')
            

        axarr[j].set_title(param_names[j])
        axarr[j].spines['right'].set_visible(False)
        axarr[j].spines['top'].set_visible(False)
    
    # Adjust spaces
    f.tight_layout()


    if show:
        plt.show()

    if filename is not None:
        f.savefig(filename)


def plot_dynamic_posteriors(micro_samples, par_labels, par_names, 
                            ground_truths=None, color_pred='#852626'):
    """
    Inspects the dynamic posterior given a single data set. Assumes six dynamic paramters.
    """
    
    # assert len(micro_samples.shape) == 3, "Dynamic posterior should be 3-dimensional!" 
    assert ground_truths is None or len(ground_truths.shape) == 2,'Ground truths should be 2-dimensional!'
        
    post_means = micro_samples.mean(axis=0)
    post_stds = micro_samples.std(axis=0)
    
    post_max = np.array(post_means).max(axis=0).max()
    upper_y_ax = post_max + 1

    sigma_factors = [1]
    alphas = [0.6]

    time = np.arange(post_means.shape[0])
    f, axarr = plt.subplots(2, 3, figsize=(18, 8))
    for i, ax in enumerate(axarr.flat):
        
        ax.plot(time, post_means[:, i], color=color_pred, alpha=0.8, label='Posterior mean', lw=1)
        for sigma_factor, alpha in zip(sigma_factors, alphas):
            ci_upper = post_means[:, i] + sigma_factor * post_stds[:, i]
            ci_lower = post_means[:, i] - sigma_factor * post_stds[:, i]
            ax.fill_between(time, ci_upper, ci_lower, color=color_pred, alpha=alpha, linewidth=0, label='Posterior sd')
        if ground_truths is not None:
            ax.plot(time, ground_truths[:, i], color='black', linestyle='solid', alpha=0.8, label='True Dynamic', lw=1)
        sns.despine(ax=ax)
        ax.set_xlabel('Time (t)', fontsize=18)
        ax.set_ylabel('Parameter value ({})'.format(par_names[i]), fontsize=18)
        ax.set_title(par_labels[i] + ' ({})'.format(par_names[i]), fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=16)
        if i < 4:
            ax.set_ylim(0, upper_y_ax)
        else:
            ax.set_ylim(0)
        ax.grid(False)

        f.subplots_adjust(hspace=0.5)
        if i == 0:
            f.legend(fontsize=16, loc='center', 
                     bbox_to_anchor=(0.5, -0.05),fancybox=False, shadow=False, ncol=4)

    f.tight_layout()


def animate_func(time):
    true_param = micro_true[:, time, :]
    pred_param_mean = micro_post.mean(axis=0)[:, time, :]
    pred_param_sd = micro_post.std(axis=0)[:, time, :]

    for j in range(len(PARAM_LABELS)):
        axarr[j].clear()

        # Plot analytic vs estimated
        if j == len(PARAM_LABELS) - 1:
            img = axarr[j].scatter(x=pred_param_mean[:, j], y=true_param[:, j], c=pred_param_sd[:, j], alpha=0.6, cmap=cm)
            # f.colorbar(img, ax=axarr[j])
        else:
            axarr[j].scatter(x=pred_param_mean[:, j], y=true_param[:, j], c=pred_param_sd[:, j], alpha=0.6, cmap=cm)

        # get axis limits and set equal x and y limits
        lower_lim = min(axarr[j].get_xlim()[0], axarr[j].get_ylim()[0])
        upper_lim = max(axarr[j].get_xlim()[1], axarr[j].get_ylim()[1])
        axarr[j].set_xlim((lower_lim, upper_lim))
        axarr[j].set_ylim((lower_lim, upper_lim))
        axarr[j].plot(axarr[j].get_xlim(), axarr[j].get_xlim(), '--', color='black')

        
        # Compute NRMSE
        rmse = np.sqrt(np.mean( (pred_param_mean[:, j] - true_param[:, j])**2 ))
        nrmse = rmse / (true_param[:, j].max() - true_param[:, j].min())
        axarr[j].text(0.1, 0.9, 'NRMSE={:.3f}'.format(nrmse),
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform=axarr[j].transAxes,
                     size=10)
        
        # Compute R2
        r2 = r2_score(true_param[:, j], pred_param_mean[:, j])
        axarr[j].text(0.1, 0.8, '$R^2$={:.3f}'.format(r2),
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform=axarr[j].transAxes, 
                     size=10)
        
        if j == 0:
            # Label plot
            axarr[j].set_xlabel('Estimated')
            axarr[j].set_ylabel('True')
            

        axarr[j].set_title(PARAM_LABELS[j])
        axarr[j].spines['right'].set_visible(False)
        axarr[j].spines['top'].set_visible(False)

    plt.figtext(0.02, 0.5, 'Time: {}'.format(time), fontsize=20)
