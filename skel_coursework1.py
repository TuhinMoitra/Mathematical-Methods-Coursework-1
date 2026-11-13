"""
skel_coursework1.py -- load the data for Coursework One.

    from skel_coursework1 import load_session
    d = load_session("coursework1_data.npz")

d is a dictionary containing:

    d["dff"]             (cells x frames) dF/F of every cell
    d["t"]               (frames,) time of each imaging frame, in seconds
    d["running_speed"]   (frames,) running speed of the mouse, cm/s
    d["movie_frame"]     (frames,) frame of natural_movie_one on screen at each imaging
                         frame (0-899), or -1 when the movie is not being shown
    d["movie_trials"]    DataFrame: start, stop (s) of each of the 10 movie repeats
    d["grating_trials"]  DataFrame: start, stop (s), direction_deg, temporal_freq_hz,
                         is_blank, for every drifting grating presentation
    d["epochs"]          DataFrame: stimulus, start, stop (s) of each stimulus block
    d["meta"]            dict: session ID, visual area, imaging depth, Cre line,
                         calcium indicator, frame rate, number of cells, ...

Useful helper:

    frames_between(d["t"], start, stop)  -> boolean mask of imaging frames in [start, stop)

Run this file directly (python skel_coursework1.py coursework1_data.npz) for a quick overview plot.
"""

import json
import sys

import numpy as np
import pandas as pd


def load_session(path):
    z = np.load(path)
    d = {k: z[k] for k in ("dff", "t", "running_speed", "movie_frame", "cell_ids")}
    d["movie_trials"] = pd.DataFrame(z["movie_trials"], columns=["start", "stop"])
    gt = pd.DataFrame(z["grating_trials"], columns=list(z["grating_columns"]))
    gt["is_blank"] = gt["is_blank"].astype(bool)
    d["grating_trials"] = gt
    d["epochs"] = pd.DataFrame({
        "stimulus": z["epoch_names"],
        "start": z["epoch_times"][:, 0],
        "stop": z["epoch_times"][:, 1],
    })
    d["meta"] = json.loads(str(z["meta"]))
    return d


def frames_between(t, start, stop):
    """Boolean mask selecting imaging frames with start <= t < stop."""
    return (t >= start) & (t < stop)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    d = load_session(sys.argv[1] if len(sys.argv) > 1 else "coursework1_data.npz")
    m = d["meta"]
    print(f"Session {m['session_id']}: {m['n_cells']} cells, {m['area']}, {m['imaging_depth_um']} um, "
          f"{m['cre_line']}, {m['indicator']}, {m['frame_rate_hz']:.2f} Hz, {m['duration_s'] / 60:.1f} min")
    print(d["epochs"].to_string(index=False))

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].imshow(d["dff"], aspect="auto", cmap="gray_r", vmin=0, vmax=np.nanpercentile(d["dff"], 99),
                 extent=[d["t"][0], d["t"][-1], d["dff"].shape[0], 0], interpolation="none")
    ax[0].set_ylabel("Cell")
    ax[1].plot(d["t"], d["running_speed"], lw=0.5, color="k")
    ax[1].set_ylabel("Running speed (cm/s)")
    ax[1].set_xlabel("Time (s)")
    kinds = sorted(d["epochs"]["stimulus"].unique())
    for _, e in d["epochs"].iterrows():
        ax[1].axvspan(e["start"], e["stop"], alpha=0.15, color="C%d" % kinds.index(e["stimulus"]))
        ax[1].text(e["start"], ax[1].get_ylim()[1], e["stimulus"], fontsize=7, va="top")
    fig.tight_layout()
    plt.show()
