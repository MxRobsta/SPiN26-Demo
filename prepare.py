import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from pathlib import Path


PLOT_FS = 100
AUDIO_FS = 500

WAVEHEIGHT = 0.8
TOPWAVECENTRE = 2


def load_json(fpath):
    with open(fpath, "r") as file:
        data = json.load(file)
    return data


def prep_audio(waveform: np.ndarray, y_centre: float | int, scale=1):
    """Prep the audio by scaling and shifting so it's centred on y_centre"""

    waveform /= np.max(np.abs(waveform))
    waveform *= scale
    waveform += y_centre

    return waveform


def plot_target(audio, target, partner, segment_file, ftemplate):
    """Plot all segments for the given target"""

    segments = load_json(segment_file)

    # Do the plots for all segments
    for segment in segments:
        dataset = segment["session"].split("_")[0]
        start = min(s["start_time"] for s in segment["prior_segments"])
        end = segment["target_segment"]["end_time"]
        start = int(start * AUDIO_FS)
        end = int(end * AUDIO_FS)

        target_snip = audio[target][start:end]
        partner_snip = audio[partner][start:end]

        fpath = ftemplate.format(
            dataset=dataset,
            session=segment["session"],
            device=segment["device"],
            pid=target,
            seg=segment["id"],
        )
        animate_waveform(
            target_snip,
            partner_snip,
            segment["target_segment"],
            segment["prior_segments"],
            fpath,
        )


def animate_waveform(target_snip, partner_snip, target_seg, prior_segments, fpath):
    """Animate the waveform so it plays with a sliding line"""

    # Setup figure and zxes
    fig, ax = plt.subplots(figsize=[9, 3])

    # Prep time info and move signals
    dt = 0.01
    signal_seconds = target_snip.shape[0] / AUDIO_FS
    t = np.linspace(0, signal_seconds, len(target_snip))
    session_start_time = min(seg["start_time"] for seg in prior_segments)
    target_start_time = target_seg["start_time"] - session_start_time

    target_snip = prep_audio(target_snip, TOPWAVECENTRE, scale=WAVEHEIGHT)
    partner_snip = prep_audio(partner_snip, 0, scale=WAVEHEIGHT)
    # Zero out partner snip during target section
    partner_snip[int(target_start_time * AUDIO_FS) :] = 0

    # Make the base plot
    ax.set_xlabel("Time/s")
    ax.set_yticks([0, TOPWAVECENTRE], ["Partner", "Target"])
    ax.plot(t, target_snip, "C0", alpha=0.5)
    ax.plot(t, partner_snip, "C1", alpha=0.5)

    ax.vlines(target_start_time, -1, 3, "red")

    target_seg["text"] = "Transcribe Here"
    for seg in prior_segments + [target_seg]:
        # Write the transcript on the plot
        middle = (seg["start_time"] + seg["end_time"]) / 2
        middle -= session_start_time

        if seg["pid"] == target_seg["pid"]:
            y = TOPWAVECENTRE + WAVEHEIGHT
        else:
            y = WAVEHEIGHT

        ax.text(middle, y, seg["text"], va="bottom", ha="center")

    def update(frame, targ_line, part_line):
        end = int((frame + dt) * AUDIO_FS) + 1
        end = min(end, target_snip.shape[-1])

        x = t[:end]
        ytarg = target_snip[:end]
        ypart = partner_snip[:end]

        targ_line.set_data(x, ytarg)
        part_line.set_data(x, ypart)

        return targ_line, part_line

    (targ_line,) = ax.plot([], [], "C0")
    (part_line,) = ax.plot([], [], "C1")

    ani = FuncAnimation(
        fig,
        update,
        fargs=(targ_line, part_line),
        frames=np.arange(0, signal_seconds, dt),
        blit=True,
        interval=dt * 1000,
        repeat=False,
    )

    if not Path(fpath).parent.exists():
        Path(fpath).parent.mkdir(parents=True)

    ani.save(fpath, writer="ffmpeg", fps=1 / dt, dpi=450)
    plt.tight_layout()
    plt.close(fig)
