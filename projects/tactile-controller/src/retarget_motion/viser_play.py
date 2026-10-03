"""Browser playback of converted motion clips, with a library browser.

    # browse everything under data/retarget_motion/motions/
    uv run python -m retarget_motion.viser_play

    # one clip, or one source, or a name filter
    uv run python -m retarget_motion.viser_play --clip motions/LAFAN/walk1_subject1.npz
    uv run python -m retarget_motion.viser_play --source AMASS --filter dance
    uv run python -m retarget_motion.viser_play --port 8090

Then forward the printed port (VS Code does it for you) and open localhost:PORT.

Playback is KINEMATIC: each frame's `qpos` is written straight onto
`assets.TACTILE_XML` and only forward kinematics runs, so what you see is the
reference itself -- never a simulation of it drifting away.  mjviser supplies
play/pause/speed/reset; this module adds the library browser (filter box, clip
dropdown, prev/next/random) and a frame scrubber on top of the same server.

With 17k+ AMASS clips a dropdown listing everything is unusable, so the dropdown
shows the first `--max-options` matches of the filter box and prev/next walk the
FULL filtered list.

Reuses `rl.viz.viser_play`'s two mjviser workarounds (mesh LOD off, camera
re-aim) rather than re-deriving them -- each cost a debugging session.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import assets
from rl.viz.viser_play import aim_camera, disable_mesh_lod
from retarget_motion import paths, schema

MAX_OPTIONS = 200


def find_clips(root: Path, source: str | None = None,
               pattern: str | None = None) -> list[Path]:
    """All converted clips under `root`, optionally restricted to one source
    sub-tree and/or a case-insensitive substring of the relative path."""
    base = root / source if source else root
    if not base.exists():
        return []
    clips = sorted(base.rglob("*.npz"))
    if pattern:
        low = pattern.lower()
        clips = [c for c in clips if low in str(c.relative_to(root)).lower()]
    return clips


def _label(clip: Path, root: Path) -> str:
    return str(clip.relative_to(root))


def serve(clips: list[Path], root: Path, host: str = "0.0.0.0", port: int = 8080,
          max_options: int = MAX_OPTIONS) -> None:
    """Serve the browser. Blocks until Ctrl-C."""
    import mujoco
    import viser
    from mjviser import Viewer

    if not clips:
        raise SystemExit("no clips found -- run `python -m retarget_motion.convert` first")

    model = mujoco.MjModel.from_xml_path(
        str(assets.ASSET_DIR / assets.TACTILE_XML)
    )
    data = mujoco.MjData(model)

    # Mutable playback state, shared with the mjviser step callback.
    st: dict = {"qpos": None, "k": 0, "clip": None, "n": 0, "scrubbing": False}

    def load_clip(path: Path) -> None:
        m = schema.load(path)
        qpos = m.to_qpos()
        if qpos.shape[1] != model.nq:
            raise ValueError(
                f"{path.name}: clip has {qpos.shape[1]} qpos entries, "
                f"{assets.TACTILE_XML} wants {model.nq}"
            )
        st.update(qpos=qpos, k=0, clip=m, n=len(qpos))
        # mjviser paces the loop by the model timestep -> real-time playback.
        model.opt.timestep = 1.0 / float(m.fps)

    def step(mdl, dat) -> None:
        dat.qpos[:] = st["qpos"][st["k"]]
        dat.qvel[:] = 0.0
        mujoco.mj_forward(mdl, dat)
        if not st["scrubbing"]:
            st["k"] = (st["k"] + 1) % st["n"]

    def reset(mdl, dat) -> None:
        st["k"] = 0
        step(mdl, dat)

    load_clip(clips[0])
    server = viser.ViserServer(host=host, port=port)
    disable_mesh_lod(server)
    aim_camera(server)
    reset(model, data)
    # Constructed here, RUN at the very bottom: Viewer.run() builds mjviser's own
    # GUI and then blocks in its tick loop, so our panels have to be registered
    # in between.
    viewer = Viewer(model, data, step_fn=step, reset_fn=reset, server=server)

    # ---- library browser -------------------------------------------------
    with server.gui.add_folder("Motion library"):
        gui_filter = server.gui.add_text("Filter", initial_value="")
        gui_count = server.gui.add_text("Matches", initial_value="", disabled=True)
        gui_clip = server.gui.add_dropdown(
            "Clip", options=[_label(c, root) for c in clips[:max_options]],
            initial_value=_label(clips[0], root),
        )
        gui_prev = server.gui.add_button("< Prev")
        gui_next = server.gui.add_button("Next >")
        gui_rand = server.gui.add_button("Random")
        gui_info = server.gui.add_markdown("")

    with server.gui.add_folder("Frame"):
        gui_scrub = server.gui.add_checkbox("Scrub (pause advance)", False)
        gui_frame = server.gui.add_slider("Frame", 0, max(st["n"] - 1, 1), 1, 0)

    matches: dict = {"list": clips, "idx": 0}

    def refresh_info() -> None:
        m = st["clip"]
        gui_info.content = (
            f"**{Path(m.src_path).name or '?'}**\n\n"
            f"- source: `{m.src_dataset or '?'}`\n"
            f"- {st['n']} frames @ {m.fps:g} fps = **{m.duration:.1f} s**\n"
            f"- raw fps: {m.src_fps:g}  |  ground shift: {m.src_ground_dz:+.3f} m\n"
            f"- {len(m.body_names)} bodies, {len(m.joint_names)} joints\n"
            f"- clip {matches['idx'] + 1} / {len(matches['list'])}"
        )
        gui_frame.max = max(st["n"] - 1, 1)
        gui_frame.value = 0

    def select(i: int) -> None:
        lst = matches["list"]
        if not lst:
            return
        matches["idx"] = i % len(lst)
        path = lst[matches["idx"]]
        load_clip(path)
        label = _label(path, root)
        if label not in gui_clip.options:      # outside the dropdown window
            gui_clip.options = [label] + list(gui_clip.options)[: max_options - 1]
        gui_clip.value = label
        refresh_info()

    @gui_filter.on_update
    def _(_evt) -> None:
        lst = find_clips(root, None, gui_filter.value or None)
        matches["list"] = lst or clips
        gui_count.value = f"{len(matches['list'])} clips"
        gui_clip.options = [_label(c, root) for c in matches["list"][:max_options]]
        select(0)

    @gui_clip.on_update
    def _(_evt) -> None:
        target = root / gui_clip.value
        for i, c in enumerate(matches["list"]):
            if c == target:
                if i != matches["idx"]:
                    select(i)
                return

    @gui_prev.on_click
    def _(_evt) -> None:
        select(matches["idx"] - 1)

    @gui_next.on_click
    def _(_evt) -> None:
        select(matches["idx"] + 1)

    @gui_rand.on_click
    def _(_evt) -> None:
        select(random.randrange(len(matches["list"])))

    @gui_scrub.on_update
    def _(_evt) -> None:
        st["scrubbing"] = gui_scrub.value

    @gui_frame.on_update
    def _(_evt) -> None:
        if st["scrubbing"]:
            st["k"] = int(gui_frame.value) % st["n"]

    gui_count.value = f"{len(clips)} clips"
    refresh_info()

    print(f"[retarget_motion] {len(clips)} clips -- "
          f"http://localhost:{port}  (Ctrl-C to stop)", flush=True)
    viewer.run()      # blocks until Ctrl-C


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", type=str, default=None,
                    help="play a single .npz (path, absolute or repo-relative)")
    ap.add_argument("--source", type=str, default=None, choices=list(paths.SOURCES),
                    help="restrict the library to one source")
    ap.add_argument("--filter", type=str, default=None,
                    help="initial case-insensitive substring filter")
    ap.add_argument("--root", type=str, default=str(paths.MOTION_DIR))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--max-options", type=int, default=MAX_OPTIONS)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if args.clip:
        p = Path(args.clip)
        if not p.is_absolute():
            p = (paths.REPO_ROOT / p).resolve()
        if not p.exists():
            raise SystemExit(f"no such clip: {p}")
        clips, root = [p], p.parent
    else:
        clips = find_clips(root, args.source, args.filter)

    serve(clips, root, host=args.host, port=args.port,
          max_options=args.max_options)
    return 0


if __name__ == "__main__":
    sys.exit(main())
