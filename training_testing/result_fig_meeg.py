import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
from typing import Dict, List, Tuple, Optional
plt.rcParams.update({
    'font.size': 7,
    'axes.titlesize': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.titlesize': 7, 
    'figure.labelsize': 7, 
    'pdf.fonttype': 42,
})

def plot_all_mappings(
    df: pd.DataFrame,
    mapping: Dict[str, List[str]],
    times_col: str = "times",
    freqs_col: str = "freqs",
    cols_tfr: Tuple[str, str, str] = ("x_tfr_mag_mean", "e_tfr_mag_mean", "u_tfr_mag_mean"),
    cols_ts: Tuple[str, str, str] = ("x_mean_abs", "e_mean_abs", "u_mean_abs"),
    title_cols: Tuple[str, str, str] = ("Unseparated", "Evoked", "Ongoing/Induced"),
    suptitle: Optional[str] = None,
    cmap: str = "viridis",
    scale: str = "db",        # 'linear' | 'log' | 'db'
    std_scale: float = 1.0,   # envelope scale for time-series
    verbose: bool = True,     # print why pairs are skipped
) -> plt.Figure:
    """
    Plot one row per (model, test) pair with:
      - Three TFR panels and one time-series panel.
      - Only show frequencies from 2–30 Hz with y-ticks at 5, 15, 25 Hz.
        In each row, only the first TFR axis shows y-tick labels; the other TFRs show ticks but no labels.
      - Only show times from -0.2 to 0.4 (seconds).
      - One shared color scale (and single colorbar with 3 ticks) across all rows/panels.
      - Only the top row shows the TFR column titles.
      - Only the first row shows the time-series legend.
    """
    f_low, f_high = 2.0, 30.0
    ytick_vals = np.array([5.0, 15.0, 25.0])
    t_low, t_high = -200, 400
    xticks_common = np.linspace(t_low, t_high, 4)

    pairs: List[Tuple[str, str]] = [(m, t) for m, tests in mapping.items() for t in tests]

    events = []
    per_event = []

    def _is_1d_numeric(arr):
        try:
            a = np.asarray(arr)
            return a.ndim == 1 and np.isfinite(a).any()
        except Exception:
            return False

    def _stack_mean(arrs: List[np.ndarray]) -> np.ndarray:
        arrs = [np.asarray(a) for a in arrs]
        return np.nanmean(np.stack(arrs, axis=0), axis=0)

    skipped = []
    for model_dt, test_dt in pairs:
        sel = (df["stimulation_type_train"] == model_dt) & (df["stimulation_type_test"] == test_dt)
        sub = df.loc[sel]
        if sub.empty:
            skipped.append((model_dt, test_dt, "no rows"))
            continue

        # Validate axes
        t_val = sub.iloc[0].get(times_col, None)
        f_val = sub.iloc[0].get(freqs_col, None)
        if (t_val is None) or (f_val is None) or (not _is_1d_numeric(t_val)) or (not _is_1d_numeric(f_val)):
            skipped.append((model_dt, test_dt, "invalid times/freqs"))
            continue
        times = np.asarray(t_val)*1000
        freqs = np.asarray(f_val)

        # Determine crop masks for this row
        t_mask = (times >= t_low) & (times <= t_high)
        f_mask = (freqs >= f_low) & (freqs <= f_high)
        if t_mask.sum() < 2 or f_mask.sum() < 1:
            skipped.append((model_dt, test_dt, "insufficient points after cropping to time/freq ranges"))
            continue

        # Collect and validate shapes, then crop
        ok = True
        x_tfr_list, e_tfr_list, u_tfr_list = [], [], []
        x_ts_list, e_ts_list, u_ts_list = [], [], []
        for _, r in sub.iterrows():
            try:
                # TFRs must be (F,T)
                x_tfr = np.asarray(r[cols_tfr[0]]); e_tfr = np.asarray(r[cols_tfr[1]]); u_tfr = np.asarray(r[cols_tfr[2]])
                if x_tfr.shape != (len(freqs), len(times)) or e_tfr.shape != (len(freqs), len(times)) or u_tfr.shape != (len(freqs), len(times)):
                    ok = False; break
                # Crop to desired freq/time window
                x_tfr = x_tfr[f_mask][:, t_mask]
                e_tfr = e_tfr[f_mask][:, t_mask]
                u_tfr = u_tfr[f_mask][:, t_mask]
                x_tfr_list.append(x_tfr); e_tfr_list.append(e_tfr); u_tfr_list.append(u_tfr)

                # Time series must be (T,) -> crop
                x_ts = np.asarray(r[cols_ts[0]]); e_ts = np.asarray(r[cols_ts[1]]); u_ts = np.asarray(r[cols_ts[2]])
                if x_ts.shape != (len(times),) or e_ts.shape != (len(times),) or u_ts.shape != (len(times),):
                    ok = False; break
                x_ts_list.append(x_ts[t_mask]); e_ts_list.append(e_ts[t_mask]); u_ts_list.append(u_ts[t_mask])
            except Exception:
                ok = False; break
        if not ok:
            skipped.append((model_dt, test_dt, "shape mismatch in TFR/time-series"))
            continue
        if skipped:
            print(skipped)
            return False
            

        # Aggregate across subjects (mean for cropped TFRs)
        x_tfr_mean = _stack_mean(x_tfr_list)  # (F', T')
        e_tfr_mean = _stack_mean(e_tfr_list)  # (F', T')
        u_tfr_mean = _stack_mean(u_tfr_list)  # (F', T')

        # Time-series: mean and std of abs across subjects (cropped)
        x_ts = np.vstack([np.abs(a) for a in x_ts_list])  # (N, T')
        e_ts = np.vstack([np.abs(a) for a in e_ts_list])
        u_ts = np.vstack([np.abs(a) for a in u_ts_list])
        x_ts_mean, x_ts_std = np.nanmean(x_ts, axis=0), np.nanstd(x_ts, axis=0)
        e_ts_mean, e_ts_std = np.nanmean(e_ts, axis=0), np.nanstd(e_ts, axis=0)
        u_ts_mean, u_ts_std = np.nanmean(u_ts, axis=0), np.nanstd(u_ts, axis=0)
        if "vis" in test_dt:
            test_dt = 'VIS'
        if "aud" in test_dt:
            test_dt = 'AUD'
        if "vis" in model_dt:
            model_dt = 'VIS'
        if "aud" in model_dt:
            model_dt = 'AUD'
        if "st" in model_dt:
            model_dt = r"TMS$_{st}$"
        if "st" in test_dt:
            test_dt = r"TMS$_{st}$"
        events.append(f"{model_dt} → {test_dt}")
        per_event.append(dict(
            times=times[t_mask], freqs=freqs[f_mask],          # cropped axes
            x_tfr=x_tfr_mean, e_tfr=e_tfr_mean, u_tfr=u_tfr_mean,
            x_ts_mean=x_ts_mean, x_ts_std=x_ts_std,
            e_ts_mean=e_ts_mean, e_ts_std=e_ts_std,
            u_ts_mean=u_ts_mean, u_ts_std=u_ts_std,
        ))


    if not events:
        raise ValueError("No valid (model, test) pairs found with usable times/freqs and shapes.")

    # Prepare values for global color scaling (across all rows, after cropping)
    all_vals = []
    for ev in per_event:
        x_plot = ev["x_tfr"]; e_plot = ev["e_tfr"]; u_plot = ev["u_tfr"]
        if scale == "db":
            x_plot = 20.0 * np.log10(x_plot)
            e_plot = 20.0 * np.log10(e_plot)
            u_plot = 20.0 * np.log10(u_plot)
        ev["x_plot"], ev["e_plot"], ev["u_plot"] = x_plot, e_plot, u_plot
        all_vals.extend([x_plot.ravel(), e_plot.ravel(), u_plot.ravel()])

    all_vals = np.concatenate(all_vals)
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        raise ValueError("No finite values for color scaling.")

    # Global color limits
    if scale in ("linear", "db"):
        vmin, vmax = np.percentile(all_vals, [1, 99])
    elif scale == "log":
        pos_vals = all_vals[all_vals > 0]
        if pos_vals.size == 0:
            raise ValueError("Log scale requested but no positive values available.")
        vmin = float(np.percentile(pos_vals, 1))
        vmax = float(np.percentile(all_vals, 99))
    else:
        raise ValueError("scale must be 'linear', 'log', or 'db'")

    # Create figure
    n_rows = len(events)
    w_in = 180 / 25.4
    h_in = 215 / 25.4
    fig, axes = plt.subplots(
        n_rows, 5, figsize=(w_in, h_in), constrained_layout=True,
        gridspec_kw={'width_ratios': [1, 1, 1, 0.08, 1]}
    )
    fig.supxlabel("Time (ms)")
    fig.supylabel("Frequency (Hz)")
    for ax in axes[:, 3].ravel():  # spacer column
        ax.set_visible(False)
    if n_rows == 1:
        axes = np.atleast_2d(axes)

    im_first = None  # for shared colorbar

    for i, (label, ev) in enumerate(zip(events, per_event)):
        ax_x, ax_e, ax_u, ax_ts = axes[i, 0], axes[i, 1], axes[i, 2], axes[i, 4]
        axs = [ax_x, ax_e, ax_u, ax_ts]

        # Row-specific axes and ticks (cropped to requested ranges)
        times_i = ev["times"]
        freqs_i = ev["freqs"]
        # Display extents anchored at the requested ranges for consistent tick placements
        extent_i = [t_low, t_high, f_low, f_high]
        xticks_i = xticks_common
        yticks_i = ytick_vals

        # TFRs
        tfr_plots = (ev["x_plot"], ev["e_plot"], ev["u_plot"])
        for j, (ax, arr, col_title) in enumerate(zip(axs[:3], tfr_plots, title_cols)):
            if scale == "log":
                im = ax.imshow(arr, aspect='auto', origin='lower',
                               extent=[times_i[0], times_i[-1], freqs_i[0], freqs_i[-1]],
                               cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
            else:
                im = ax.imshow(arr, aspect='auto', origin='lower',
                               extent=[times_i[0], times_i[-1], freqs_i[0], freqs_i[-1]],
                               cmap=cmap, vmin=vmin, vmax=vmax)
            if im_first is None:
                im_first = im
            if i == 0:  # only top row shows titles
                ax.set_title(col_title)
            if j == 0:
                ax.set_ylabel(label)
            else:
                ax.set_ylabel("")

            # Limits and ticks (force display window and tick positions)
            ax.set_xlim(t_low, t_high)
            ax.set_ylim(f_low, f_high-2)
            ax.set_xticks(xticks_i)
            ax.set_yticks(yticks_i)

            # Only first TFR axis in the row shows y tick labels
            if j == 0:
                ax.set_yticklabels([str(int(v)) for v in yticks_i])
            else:
                ax.set_yticklabels([])
            if i == len(events)-1:
                ax.set_xticklabels([str(int(v)) for v in xticks_i])
            else:
                ax.set_xticklabels([])

            ax.set_xlabel("")

        # Time-series panel
        ax_ts = axs[3]
        e_m, e_s = ev["e_ts_mean"], ev["e_ts_std"]
        u_m, u_s = ev["u_ts_mean"], ev["u_ts_std"]
        x_m, x_s = ev["x_ts_mean"], ev["x_ts_std"]

        l1 = ax_ts.plot(times_i, x_m, color='#2ca02c', label='Unseparated', linewidth=2.0)[0]
        ax_ts.fill_between(times_i, x_m - std_scale * x_s, x_m + std_scale * x_s, color='#2ca02c', alpha=0.2)
        l2 = ax_ts.plot(times_i, e_m, color='#1f77b4', label='Evoked', linewidth=2.0)[0]
        ax_ts.fill_between(times_i, e_m - std_scale * e_s, e_m + std_scale * e_s, color='#1f77b4', alpha=0.2)
        l3 = ax_ts.plot(times_i, u_m, color='#ff7f0e', label='Ongoing/Induced', linewidth=2.0)[0]
        ax_ts.fill_between(times_i, u_m - std_scale * u_s, u_m + std_scale * u_s, color='#ff7f0e', alpha=0.2)

        # Time limits and ticks
        ax_ts.set_xlim(t_low, t_high)
        ax_ts.set_xticks(xticks_i)
        ax_ts.set_xlabel("")
        ax_ts.set_ylabel("")
        y_max = max(
            float(np.max(x_m + std_scale * x_s)),
            float(np.max(e_m + std_scale * e_s)),
            float(np.max(u_m + std_scale * u_s)),
        )
        
        ax_ts.set_ylim(-0.2, y_max)
        yticks = [0.0, 1.0, 2.0]
        ax_ts.set_yticks(yticks)
        # Optional: pretty labels (compact)
        ax_ts.set_yticklabels([f"{v:.2g}" for v in yticks])
        if i == 0:
            ax_ts.set_title("Absolute mean")
        if i == 2:
    
            pad = -0.03  # fig fraction gap to the left of the axis
            label_text = "Amplitude (a.u.)"
        
            # Create the figure-level label near this axis
            ts_lbl = fig.text(
                ax_ts.get_position().x0 - pad, 0.5, label_text,
                rotation=90, va="center", ha="right", transform=fig.transFigure)
        if i == len(events)-1:
            leg = ax_ts.legend(
            handles=[l1, l2, l3],
            frameon=False,
            loc="center right",
            bbox_to_anchor=(0.95, 0.065),
            bbox_transform=fig.transFigure,
            borderaxespad=0.0
            )
            leg.set_in_layout(False)
        else:
            if ax_ts.get_legend() is not None:
                ax_ts.legend_.remove()

    # Single shared colorbar with 3 ticks (shared color scale)
    cbar = fig.colorbar(
        im_first,
        ax=axes[:, :3].ravel().tolist(),
        location='bottom',
        orientation='horizontal',
        fraction=0.05,   # width of the colorbar relative to axes
        pad=0.01         # gap between TFR axes and colorbar
    )
    
    if scale == "db":
        ticks = np.round(np.linspace(vmin, vmax, 3), 0)
        ticks[0] += 1
        ticks[-1] -= 1
        cbar.set_ticks(ticks)
        cbar.set_label("Power (dB)", rotation=90)
        cbar.ax.xaxis.set_label_position('bottom')
        cbar.ax.xaxis.set_label_coords(-0.08, -0.5)
        cbar.ax.xaxis.label.set_verticalalignment('bottom')
        cbar.ax.xaxis.label.set_horizontalalignment('center')
    else:
        ticks = np.linspace(vmin, vmax, 3)
        cbar.set_ticks(ticks)
        cbar.set_label("Power (a.u.)", loc='left')

    for ax in axes.ravel():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
    cbar.outline.set_visible(False)
    
    if suptitle:
        fig.suptitle(suptitle)

    return fig
