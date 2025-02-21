import numpy as np
import matplotlib.pyplot as plt

from math import log10
from actsim import actsim

def actdyntest():
    """
    Simulate various activation-dynamics models across a range of input frequencies
    and demonstrate the "pumping" effect. Plots partial traces (excitation/activation)
    for three representative frequencies (lowest, middle, highest), then
    plots mean activation vs frequency for all tested models.
    """

    # Define models to test
    models = [
        'McLean2003',
        'McLean2003Improved',
        'DeGroote2016Original',
        'DeGroote2016'
    ]

    # Parameters
    umax = 1.0
    frequencies = np.logspace(np.log10(1.0), np.log10(100.0), 20)  # from 1 to 100 Hz, 20 points

    # Storage for mean activation results
    # Dictionary: { model_name : [mean_a_for_each_freq, ...] }
    meanacts = {model: np.zeros(len(frequencies)) for model in models}

    # For plotting partial excitations at: first, middle, and last frequencies
    plot_indices = [0, len(frequencies)//2, len(frequencies) - 1]

    # Loop over each model
    for model in models:
        # Prepare a figure for the "first, middle, last" frequency plots
        fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
        fig.suptitle(f"{model}: Example excitations/activations")

        # For each frequency in 'frequencies':
        for i, freq in enumerate(frequencies):
            uperiod = 1.0 / freq
            dutycycle = 0.5
            # Decide whether we want to plot for this frequency
            do_plot = (i in plot_indices)

            if do_plot:
                # Map the frequency index to one of the subplots (0,1,2)
                ax_idx = plot_indices.index(i)
                ax = axes[ax_idx]  # pick the correct subplot
                ax.set_title(f"Frequency = {freq:.2f} Hz")

                # Run simulation and plot on this subplot
                # Make sure your 'actsim' function can plot to a chosen axis, or
                # you can do a temporary hack like `plt.sca(ax)` inside 'actsim'.
                # Here, we'll just do the simplest approach: pass `makeplot=True`
                # and rely on that. Then re-limit ourselves to the right subplot
                # by setting the current axis manually:
                plt.sca(ax)
                _, _, mean_a = actsim(
                    model=model,
                    umax=umax,
                    uperiod=uperiod,
                    dutycycle=dutycycle,
                    makeplot=True
                )
                # Store the mean activation for final summary
                meanacts[model][i] = mean_a

                # Optionally add custom legend only on the first subplot
                if ax_idx == 0:
                    ax.legend(['excitation', 'activation'])

            else:
                # Run simulation without plotting
                _, _, mean_a = actsim(
                    model=model,
                    umax=umax,
                    uperiod=uperiod,
                    dutycycle=dutycycle,
                    makeplot=False
                )
                meanacts[model][i] = mean_a

        # Tidy up the figure for partial traces
        for ax in axes:
            ax.set_ylim([-0.2, 1.2])
        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()

    # Finally, plot mean activation vs. frequency for all models
    plt.figure(figsize=(6,4))
    for model in models:
        plt.semilogx(frequencies, meanacts[model], '-o', label=model)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Mean Activation")
    plt.title("Response to square-wave excitation")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    actdyntest()
