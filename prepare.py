import csv
import json
import hydra
import logging
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from omegaconf import DictConfig
import os
from pathlib import Path
import soundfile as sf

PLOT_FS = 100
AUDIO_FS = 500
INPUT_FS = 16000

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


def rms_norm(audio: np.ndarray, target_rms: float):
    """Normalise an audio array to a target RMS"""

    in_rms = np.sqrt(np.mean(np.square(audio)))
    audio *= target_rms / in_rms

    return audio


def animate_waveform(target_snip, partner_snip, target_seg, prior_segments, fpath):
    """Animate the waveform so it plays with a sliding line"""

    # Setup figure and zxes
    fig, ax = plt.subplots(figsize=[9, 3])

    # Prep time info and move signals
    dt = 0.01
    signal_seconds = target_snip.shape[0] / AUDIO_FS
    t = np.linspace(0, signal_seconds, len(target_snip))
    session_start_time = target_seg["start_time"] - 5
    target_start_time = 5

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


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig):
    """Entry point to script: animates all viable segments"""

    # Basic session info
    session = cfg.session
    device = cfg.device
    target_pid = cfg.target_pid

    with open(cfg.paths.session_info, "r") as file:
        session_info = list(
            filter(lambda x: x["session"] == session, csv.DictReader(file))
        )[0]
    wearer_pid = session_info[f"pos{session_info[f'{device}_pos']}"]
    partner_pids = [session_info[f"pos{i}"] for i in range(1, 5)]
    partner_pids = [x for x in partner_pids if x not in [target_pid, wearer_pid]]

    # Load in the audio
    noisy_fpath = cfg.paths.noisy_session.format(session=session, device=device)
    noisy_audio, _ = sf.read(noisy_fpath)
    if device == "aria":
        noisy_audio = noisy_audio[:, 2]
    else:
        noisy_audio = np.sum(noisy_audio[:, :2], axis=1)

    ref_audios = {}
    ct_audios = {}
    transcripts = {}
    for pid in [target_pid, wearer_pid] + partner_pids:
        fpath = cfg.paths.ref_session.format(device=device, session=session, pid=pid)
        ref_audios[pid] = sf.read(fpath)[0]
        fpath = cfg.paths.ct_session.format(session=session, pid=pid)
        ct_audios[pid] = sf.read(fpath)[0]

        with open(cfg.paths.transcript.format(session=session, pid=pid), "r") as file:
            transcripts[pid] = json.load(file)

    # Load in the test manifest
    manifest = load_json(
        cfg.paths.manifest_ftemp.format(device=device, session=session, pid=target_pid)
    )

    # Run for everything
    decimation_factor = INPUT_FS // AUDIO_FS
    for i, segment in enumerate(manifest):
        target_start_time = segment["target_segment"]["start_time"]
        start_time = target_start_time - cfg.context_time
        end_time = segment["target_segment"]["end_time"]

        start_sample = int(start_time * INPUT_FS)
        end_sample = int(end_time * INPUT_FS)

        target_snip = ref_audios[target_pid][start_sample:end_sample:decimation_factor]
        partner_snip = np.sum(
            np.fromiter(
                (
                    ref_audios[p][start_sample:end_sample:decimation_factor]
                    for p in partner_pids
                ),
                np.ndarray,
            )
        )

        # Animate
        anim_fpath = Path(
            cfg.paths.sample_ftemp.format(
                ftype="video",
                session=session,
                device=device,
                pid=target_pid,
                seg=i,
                fext="mp4",
            )
        )
        if anim_fpath.exists() and not cfg.overwrite:
            logging.info(f"Video file found at {str(anim_fpath)}. Skipping...")
        else:
            animate_waveform(
                target_snip,
                partner_snip,
                segment["target_segment"],
                segment["prior_segments"],
                anim_fpath,
            )

        # Save audio

        audio_snip = [x[start_sample:end_sample] for x in ref_audios.values()]
        audio_snip = rms_norm(np.sum(audio_snip, axis=0), cfg.rms)
        print(audio_snip.shape)

        audio_fpath = Path(
            cfg.paths.sample_ftemp.format(
                ftype="audio",
                session=session,
                device=device,
                pid=target_pid,
                seg=i,
                fext="wav",
            )
        )
        if audio_fpath.exists() and not cfg.overwrite:
            logging.info(f"Audio file found at {str(audio_fpath)}. Skipping...")
        else:
            sf.write(audio_fpath, audio_snip, INPUT_FS)

        # FFMPEG merge

        mix_fpath = Path(
            cfg.paths.sample_ftemp.format(
                ftype="mix",
                session=session,
                device=device,
                pid=target_pid,
                seg=i,
                fext="mp4",
            )
        )
    if mix_fpath.exists() and not cfg.overwrite:
        logging.info(f"Mixed file found at {str(mix_fpath)}. Skipping...")
    else:
        os.system(
            f"ffmpeg -y -hide_banner -loglevel error -i {str(anim_fpath)} -i {str(audio_fpath)} -c:v copy -c:a aac {str(mix_fpath)}"
        )


if __name__ == "__main__":
    main()
