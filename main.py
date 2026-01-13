import hydra
from omegaconf import DictConfig
import streamlit as st


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig):

    with open("README.md", "r") as file:
        text = file.read()
    st.markdown(text)

    st.subheader("Clean Speech Sample")
    st.audio(f"data/{cfg.target_pid}.wav")

    for atype in [cfg.device, "ct"]:

        if atype == cfg.device:
            st.subheader("Aria Audio (noisy)")
        else:
            st.subheader("Close-talk Audio (clean-ish)")

        sample_fpath = cfg.paths.sample_ftemp.format(
            ftype="mix",
            session=cfg.session,
            device=cfg.device,
            mic=atype,
            pid=cfg.target_pid,
            seg=0,
            fext="mov",
        )

        st.video(sample_fpath)
        if atype == cfg.device:
            cola, colb = st.columns([1, 5])

            with cola:
                st.text("Response")
            with colb:
                st.text_input(label="thing", label_visibility="collapsed")

            with cola:
                show_ans = st.checkbox("Show answer")
            if show_ans:
                with colb:
                    st.text("I did have a separate app before for Sudoku.")


if __name__ == "__main__":
    main()
