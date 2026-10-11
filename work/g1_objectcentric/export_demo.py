"""Export real simulated RGB, metric depth, state/action and phase timestamps.

Produces a LeRobot v2.1-shaped GR00T dataset with Parquet/MP4/modality.json.
Local schema/video round-trip is checked; the GR00T loader itself requires a
separate installed GR00T environment and is not claimed to have run here.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import sys
import json
import argparse
import struct
import zlib
import numpy as np
import mujoco
import pyarrow as pa
import pyarrow.parquet as pq
import imageio_ffmpeg
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from work.g1_objectcentric.task import ObjectCentricTask, ROOT
from work.g1_objectcentric.geometry import log_so3
from work.g1_objectcentric.scenarios import Scenario

FPS, WIDTH, HEIGHT = 10, 320, 240
TEXT = "Use both arms to pick up the box, carry it to the green area and release it."


def write_png(path, array):
    h, w = array.shape[:2]
    raw = b"".join(b"\0"+row.tobytes() for row in array)
    chunk = lambda kind, content: struct.pack(">I", len(content))+kind+content+struct.pack(">I", zlib.crc32(kind+content)&0xffffffff)
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR", struct.pack(">2I5B", w, h, 8, 2, 0, 0, 0))+
                         chunk(b"IDAT", zlib.compress(raw, 6))+chunk(b"IEND", b""))


def eef_vector(poses):
    return np.concatenate([np.r_[p[:3, 3], log_so3(p[:3, :3])] for p in poses])


def stats(array):
    array = np.asarray(array, dtype=float)
    return {k: value.tolist() for k, value in dict(min=np.min(array, axis=0), max=np.max(array, axis=0),
            mean=np.mean(array, axis=0), std=np.std(array, axis=0), q01=np.quantile(array, .01, axis=0),
            q99=np.quantile(array, .99, axis=0), count=np.array([len(array)])).items()}


def camera_record(scene):
    cam = mujoco.mjv_averageCamera(scene.camera[0], scene.camera[1])
    forward, up = np.array(cam.forward), np.array(cam.up)
    right = np.cross(forward, up)
    c2w = np.eye(4)
    c2w[:3, :3] = np.column_stack([right, up, -forward])
    c2w[:3, 3] = cam.pos
    return dict(T_world_camera_opengl=c2w.tolist(), frustum={n:float(getattr(cam, n)) for n in
                  ("frustum_bottom", "frustum_top", "frustum_near", "frustum_far", "frustum_center", "frustum_width")})


def export(episodes=2):
    destination = ROOT/"dataset"
    meta, data, videos = destination/"meta", destination/"data"/"chunk-000", destination/"videos"/"chunk-000"
    for directory in (meta, data, videos):
        directory.mkdir(parents=True, exist_ok=True)
    env = ObjectCentricTask()
    render_model = mujoco.MjModel.from_binary_path(str(ROOT.parent/"box_transfer"/"appearance.mjb"))
    render_data = mujoco.MjData(render_model)
    camera_options = [("front", 150), ("side", 75)]
    renderer = mujoco.Renderer(render_model, height=HEIGHT, width=WIDTH)
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    all_states, all_actions, episode_records, pose_logs = [], [], [], []
    total_frames = 0
    camera_records = {}
    try:
        for episode in range(episodes):
            env.reset(seed=70+episode, mode="feedback", scenario=Scenario())
            records, depths = [], []
            writers = {}
            for name, _ in camera_options:
                folder = videos/("observation.images."+name)
                folder.mkdir(exist_ok=True)
                writer = imageio_ffmpeg.write_frames(str(folder/f"episode_{episode:06d}.mp4"), (WIDTH, HEIGHT),
                          fps=FPS, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p", quality=6,
                          macro_block_size=1, ffmpeg_log_level="error", output_params=["-threads", "1"])
                writer.send(None)
                writers[name] = writer
            next_frame = 0.
            try:
                while True:
                    env.targets()
                    if env.elapsed+1e-8 >= next_frame:
                        state = np.r_[eef_vector(env.fk(env.data.qpos[env.arm_q])), env.data.qpos[env.arm_q]].astype(np.float32)
                        action = np.r_[eef_vector(env.fk(env.command_q)), env.command_q].astype(np.float32)
                        index = len(records)
                        records.append(dict(timestamp=index/FPS, frame_index=index, episode_index=episode,
                                       index=total_frames+index, task_index=0, phase=env.phase,
                                       **{"observation.state":state.tolist(), "action":action.tolist()}))
                        all_states.append(state)
                        all_actions.append(action)
                        pose_logs.append(dict(episode=episode, frame=index, physics_timestamp=env.elapsed,
                                              object_pose=env.object_pose().tolist(), phase=env.phase))
                        render_data.qpos[:] = env.data.qpos
                        mujoco.mj_kinematics(render_model, render_data)
                        mujoco.mj_camlight(render_model, render_data)
                        frame_depths = []
                        for name, azimuth in camera_options:
                            camera = mujoco.MjvCamera()
                            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                            camera.lookat[:] = [.27, .04, .87]
                            camera.distance, camera.azimuth, camera.elevation = 1.35, azimuth, -20
                            renderer.update_scene(render_data, camera=camera, scene_option=options)
                            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                            camera_records[name] = camera_record(renderer.scene)
                            rgb = renderer.render().copy()
                            writers[name].send(rgb)
                            if episode == 0 and env.phase == "CARRY" and not (destination/f"preview-{name}.png").exists():
                                write_png(destination/f"preview-{name}.png", rgb)
                            renderer.enable_depth_rendering()
                            renderer.update_scene(render_data, camera=camera, scene_option=options)
                            frame_depths.append(renderer.render().copy())
                            renderer.disable_depth_rendering()
                        depths.append(frame_depths)
                        next_frame += 1/FPS
                    if not env.step():
                        break
            finally:
                for writer in writers.values():
                    writer.close()
            schema = pa.schema([
                pa.field("timestamp", pa.float32()),
                *[pa.field(k, pa.int64()) for k in ("frame_index", "episode_index", "index", "task_index")],
                pa.field("phase", pa.string()), pa.field("observation.state", pa.list_(pa.float32(), 26)),
                pa.field("action", pa.list_(pa.float32(), 26)),
            ])
            table = pa.Table.from_pylist(records, schema=schema)
            pq.write_table(table, data/f"episode_{episode:06d}.parquet", compression="zstd")
            np.savez_compressed(destination/f"depth_{episode:06d}.npz", depth_m=np.array(depths, dtype=np.float32),
                                timestamps=np.arange(len(records))/FPS, camera_names=np.array([c[0] for c in camera_options]))
            episode_records.append(dict(episode_index=episode, tasks=[TEXT], length=len(records),
                                        simulation_result=env.result()))
            total_frames += len(records)
            print(f"Exported simulated episode {episode}: {len(records)} frames, success={env.result()['success']}", flush=True)
    finally:
        renderer.close()
    eef_names = [f"{side}_{axis}" for side in ("left", "right") for axis in ("x", "y", "z", "rx", "ry", "rz")]
    joint_names = [env.model.joint(int(j)).name for arm in env.arms for j in arm["joints"]]
    features = {"observation.state":dict(dtype="float32", shape=[26], names=eef_names+joint_names),
                "action":dict(dtype="float32", shape=[26], names=eef_names+joint_names)}
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        features[key] = dict(dtype="float32" if key == "timestamp" else "int64", shape=[1], names=None)
    for name, _ in camera_options:
        features["observation.images."+name] = dict(dtype="video", shape=[HEIGHT, WIDTH, 3], names=["height", "width", "channel"],
                info={"video.fps":FPS, "video.codec":"h264", "video.pix_fmt":"yuv420p", "video.is_depth_map":False,
                      "video.has_audio":False, "video.channels":3})
    features["phase"] = dict(dtype="string", shape=[1], names=None)
    info = dict(codebase_version="v2.1", robot_type="unitree_g1_fixed_base_palms_sim", total_episodes=episodes,
                total_frames=total_frames, total_tasks=1, total_videos=2*episodes, total_chunks=1, chunks_size=1000,
                fps=FPS, splits={"train":f"0:{episodes}"}, features=features,
                data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    modality = {kind:{"left_eef":dict(start=0, end=6), "right_eef":dict(start=6, end=12), "arms":dict(start=12, end=26)}
                for kind in ("state", "action")}
    modality["video"] = {name:dict(original_key="observation.images."+name) for name, _ in camera_options}
    modality["annotation"] = {"human.task_description":dict(original_key="task_index")}
    for name, content in {"info":info, "modality":modality,
                          "stats":{"observation.state":stats(all_states), "action":stats(all_actions)},
                          "cameras":dict(cameras=camera_records, coordinate_frame="MuJoCo world; OpenGL camera coordinates", depth_units="metres",
                                         simulated=True, real_camera_calibration=False),
                          "provenance":dict(source="physics simulation", controller="CEM trajectory plus object-centric feedback",
                                            vla_weights_used=False, real_robot_data=False, gr00t_loader_validated=False)}.items():
        (meta/(name+".json")).write_text(json.dumps(content, indent=2), encoding="utf-8")
    (meta/"tasks.jsonl").write_text(json.dumps(dict(task_index=0, task=TEXT))+"\n", encoding="utf-8")
    (meta/"episodes.jsonl").write_text("".join(json.dumps(r)+"\n" for r in episode_records), encoding="utf-8")
    (destination/"sim-pose-timestamps.jsonl").write_text("".join(json.dumps(r)+"\n" for r in pose_logs), encoding="utf-8")
    for ep in range(episodes):
        table = pq.read_table(data/f"episode_{ep:06d}.parquet")
        assert table.schema.field("observation.state").type == pa.list_(pa.float32(), 26)
        assert table.schema.field("action").type == pa.list_(pa.float32(), 26)
        rows = table.to_pylist()
        assert len(rows) == episode_records[ep]["length"]
        assert all(len(r["observation.state"]) == len(r["action"]) == 26 for r in rows)
        assert all(abs(r["timestamp"]-i/FPS) < 1e-6 for i, r in enumerate(rows))
        for name, _ in camera_options:
            reader = imageio_ffmpeg.read_frames(str(videos/("observation.images."+name)/f"episode_{ep:06d}.mp4"))
            metadata = next(reader)
            assert tuple(metadata["size"]) == (WIDTH, HEIGHT)
            assert sum(1 for _ in reader) == len(rows)
    (destination/"validation.json").write_text(json.dumps(dict(passed=True, episodes=episodes, total_frames=total_frames,
                        checks=["Parquet roundtrip", "state/action dimensions", "timestamp cadence", "H264 decoding", "video frame counts"],
                        gr00t_loader_tested=False), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    export(parser.parse_args().episodes)
