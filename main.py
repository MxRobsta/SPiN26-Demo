import hydra
from omegaconf import DictConfig
import streamlit as st


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig):

    for atype in [cfg.device, "ct"]:
        sample_fpath = cfg.paths.sample_ftemp.format(
            ftype="mix",
            session=cfg.session,
            device=cfg.device,
            mic=atype,
            pid=cfg.target_pid,
            seg=cfg.target_seg,
            fext="mp4",
        )

        st.video(sample_fpath)


if __name__ == "__main__":
    main()
