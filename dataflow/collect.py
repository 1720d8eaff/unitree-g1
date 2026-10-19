"""Collect actual CEM-driven MuJoCo episodes, then export train/validation data.

Success comes from physical telemetry, never a prescribed success count. A keep
quota limits accepted examples; eligible quota exclusions remain in the manifest.
This is simulation, not human demonstrations, GR00T inference, or real robot data.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fresh(path):
    path = Path(path).resolve()
    for protected in (ROOT/"work/g1_objectcentric/results", ROOT/"work/g1_objectcentric/dataset", ROOT/"results"):
        if path==protected or protected in path.parents or path in protected.parents:
            raise ValueError("Output overlaps archived evidence")
    path.mkdir(parents=True, exist_ok=False)
    return path


def collect(output, *, episodes=220, keep=174, seed=20261019, export=True, width=160, height=120):
    from benchmark.protocol import SuccessEvaluator
    from benchmark.runner import telemetry
    from work.g1_objectcentric.task import ObjectCentricTask
    from work.g1_objectcentric.geometry import log_so3
    from work.g1_objectcentric.scenarios import suite
    if (type(episodes) is not int or type(keep) is not int or type(seed) is not int
        or not 1<=keep<=episodes<=10000 or seed<0):
        raise ValueError("Require 1 <= keep <= episodes <= 10000 and nonnegative seed")
    output = fresh(output)
    (output/"raw").mkdir()
    plan = [dict(episode_id=f"sim-{i:04d}", scene_id=i%24, seed=seed+i*997) for i in range(episodes)]
    source_files = [ROOT/p for p in ["dataflow/collect.py", "work/g1_objectcentric/task.py",
        "work/g1_objectcentric/reference-parameters.json", "work/box_transfer/scene.xml", "work/box_lift/task.py",
        "benchmark/protocol.py"]]
    write_json(output/"plan.json", dict(source="simulation", action_source="CEM with simulated pose feedback",
        real_robot_data=False, vla_weights_used=False, requested_keep=keep, planned_episodes=plan,
        selection_rule="terminal-success eligible; first eligible episodes up to quota; all exclusions retained",
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in source_files}))
    env, scenes, rows, accepted = ObjectCentricTask(), suite(), [], 0
    start_wall = time.perf_counter()
    for index, spec in enumerate(plan):
        began = datetime.now(timezone.utc).isoformat()
        scene = scenes[spec["scene_id"]]
        env.reset(seed=spec["seed"], mode="feedback", scenario=scene)
        evaluator = SuccessEvaluator(float(env.initial_z))
        evaluator.update(telemetry(env))
        frames, states, actions, phases, stamps, physics, next_frame = [], [], [], [], [], [], 0.
        while True:
            env.targets()
            if env.elapsed+1e-8 >= next_frame:
                eef = np.concatenate([np.r_[p[:3,3],log_so3(p[:3,:3])] for p in env.fk(env.data.qpos[env.arm_q])])
                target_eef = np.concatenate([np.r_[p[:3,3],log_so3(p[:3,:3])] for p in env.fk(env.command_q)])
                states.append(np.r_[eef,env.data.qpos[env.arm_q]])
                actions.append(np.r_[target_eef,env.command_q])
                phases.append(env.phase); stamps.append(float(env.elapsed)); frames.append(env.data.qpos.copy())
                next_frame += .1
            running = env.step()
            if env.steps%5==0 or not running:
                sample = telemetry(env)
                evaluator.update(sample); physics.append(sample)
            if not running:break
        result = evaluator.result()
        eligible = bool(result["terminal_success"])
        take = eligible and accepted<keep
        accepted += take
        name = spec["episode_id"]
        trajectory = output/"raw"/(name+".npz")
        np.savez_compressed(trajectory, source=np.array("simulation"), real_robot_data=np.array(False),
            qpos=np.asarray(frames), state=np.asarray(states,dtype=np.float32), action=np.asarray(actions,dtype=np.float32),
            phase=np.asarray(phases), physics_timestamp=np.asarray(stamps), timestamp=np.arange(len(frames))*.1)
        trace = output/"raw"/(name+".telemetry.jsonl.gz")
        with gzip.open(trace,"wt",encoding="utf-8",newline="\n") as f:
            for sample in physics:f.write(json.dumps(sample,allow_nan=False)+"\n")
        rows.append(dict(**spec, source="simulation", robot="unitree_g1_fixed_base_palms_sim", task="box_transfer",
            operator_role="automated_CEM_controller", started_at=began, ended_at=datetime.now(timezone.utc).isoformat(),
            frames=len(frames), accepted=take, eligible=eligible,
            exclusion_reason=None if take else "quota_reached" if eligible else "terminal_success_failed",
            scenario=asdict(scene), criteria=asdict(evaluator.criteria), initial_z_m=float(env.initial_z), task_result=result,
            artifacts={"trajectory":dict(path=str(trajectory.relative_to(output)),sha256=digest(trajectory)),
                       "telemetry":dict(path=str(trace.relative_to(output)),sha256=digest(trace))}))
        print(f"{index+1}/{episodes} eligible={eligible} selected={accepted}/{keep}",flush=True)
    if export and accepted:
        export_lerobot(output,rows,seed=seed,width=width,height=height)
    else:
        write_json(output/"calibration.json",dict(source="simulation",image_exported=False,real_camera_calibration=False))
    calibration=dict(path="calibration.json",sha256=digest(output/"calibration.json"))
    for row in rows:row["artifacts"]["calibration"]=calibration
    with (output/"episodes.jsonl").open("x",encoding="utf-8",newline="\n") as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+"\n")
    info=dict(source="simulation",real_robot_data=False,vla_weights_used=False,planned_episodes=episodes,
        eligible_episodes=sum(r["eligible"] for r in rows),accepted_episodes=accepted,rejected_episodes=episodes-accepted,
        accepted_frames=sum(r["frames"] for r in rows if r["accepted"]),total_frames=sum(r["frames"] for r in rows),
        requested_keep=keep,quota_met=accepted==keep,wall_s=time.perf_counter()-start_wall,
        interpretation="Collection eligibility and quota, not independent policy test performance.")
    write_json(output/"summary.json",info)
    print(json.dumps(info,indent=2),flush=True)
    return info


def train_stats(arrays):
    a=np.concatenate(arrays)
    return {k:v.tolist() for k,v in dict(min=a.min(0),max=a.max(0),mean=a.mean(0),std=a.std(0),
        q01=np.quantile(a,.01,axis=0),q99=np.quantile(a,.99,axis=0),count=np.array([len(a)])).items()}


def export_lerobot(output, rows, *, seed, width, height):
    import imageio_ffmpeg
    import pyarrow as pa
    import pyarrow.parquet as pq
    from dataflow.camera import RGBDCameras
    from work.g1_objectcentric.task import ObjectCentricTask
    from work.g1_objectcentric.scenarios import Scenario
    selected=[r for r in rows if r["accepted"]]
    order=np.random.default_rng(seed).permutation(len(selected))
    n_train=max(1,int(.8*len(selected)))
    groups={"train":[selected[i] for i in order[:n_train]],"validation":[selected[i] for i in order[n_train:]]}
    fit=[np.load(output/r["artifacts"]["trajectory"]["path"],allow_pickle=False) for r in groups["train"]]
    try: stats={key:train_stats([a[key] for a in fit]) for key in ("state","action")}
    finally:
        for a in fit:a.close()
    stats={"observation.state":stats["state"],"action":stats["action"]}
    env=ObjectCentricTask(); cameras={}; calibration={}
    try:
        for partition,episodes in groups.items():
            if not episodes:continue
            root=output/"datasets"/partition
            for folder in [root/"meta",root/"data/chunk-000",root/"videos/chunk-000"]:folder.mkdir(parents=True)
            episode_meta=[];global_frame=0
            for ep,row in enumerate(episodes):
                scene=Scenario(**row["scenario"])
                env.reset(seed=row["seed"],mode="feedback",scenario=scene)
                key=scene.size_scale
                if key not in cameras:cameras[key]=RGBDCameras(env.model,width,height)
                camera=cameras[key]
                with np.load(output/row["artifacts"]["trajectory"]["path"],allow_pickle=False) as a:
                    writers={}
                    try:
                        for name in ("front","side"):
                            directory=root/"videos/chunk-000"/("observation.images."+name);directory.mkdir(exist_ok=True)
                            w=imageio_ffmpeg.write_frames(str(directory/f"episode_{ep:06d}.mp4"),(width,height),fps=10,
                                codec="libx264",pix_fmt_in="rgb24",pix_fmt_out="yuv420p",macro_block_size=1,
                                ffmpeg_log_level="error",output_params=["-threads","1"])
                            w.send(None);writers[name]=w
                        records=[]
                        for frame,qpos in enumerate(a["qpos"]):
                            images=camera.frame(qpos,depth=False)
                            for name,w in writers.items():w.send(images[name][0])
                            records.append(dict(timestamp=frame/10,frame_index=frame,episode_index=ep,index=global_frame+frame,
                                task_index=0,phase=str(a["phase"][frame]),**{"observation.state":a["state"][frame].tolist(),"action":a["action"][frame].tolist()}))
                    finally:
                        for w in writers.values():w.close()
                    calibration=camera.calibration
                schema=pa.schema([pa.field("timestamp",pa.float32()),
                    *[pa.field(k,pa.int64()) for k in ("frame_index","episode_index","index","task_index")],
                    pa.field("phase",pa.string()),pa.field("observation.state",pa.list_(pa.float32(),26)),pa.field("action",pa.list_(pa.float32(),26))])
                pq.write_table(pa.Table.from_pylist(records,schema=schema),root/"data/chunk-000"/f"episode_{ep:06d}.parquet")
                episode_meta.append(dict(episode_index=ep,tasks=["Use both arms to transfer the box."],length=len(records),
                    source_episode=row["episode_id"],source="simulation"))
                global_frame+=len(records)
                print(f"export {partition} {ep+1}/{len(episodes)}",flush=True)
            features={"observation.state":dict(dtype="float32",shape=[26]),"action":dict(dtype="float32",shape=[26])}
            for k in ("timestamp","frame_index","episode_index","index","task_index"):
                features[k]=dict(dtype="float32" if k=="timestamp" else "int64",shape=[1],names=None)
            features["phase"]=dict(dtype="string",shape=[1],names=None)
            for name in ("front","side"):
                features["observation.images."+name]=dict(dtype="video",shape=[height,width,3],names=["height","width","channel"],
                    info={"video.fps":10,"video.codec":"h264","video.pix_fmt":"yuv420p","video.channels":3,"video.is_depth_map":False,"video.has_audio":False})
            info=dict(codebase_version="v2.1",robot_type="unitree_g1_fixed_base_palms_sim",total_episodes=len(episodes),
                total_frames=global_frame,total_tasks=1,total_videos=2*len(episodes),fps=10,total_chunks=1,chunks_size=1000,
                splits={"train":f"0:{len(episodes)}"},features=features,
                data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
            modality={kind:{"left_eef":dict(start=0,end=6),"right_eef":dict(start=6,end=12),"arms":dict(start=12,end=26)} for kind in ("state","action")}
            modality["video"]={n:dict(original_key="observation.images."+n) for n in ("front","side")}
            modality["annotation"]={"human.task_description":dict(original_key="task_index")}
            for name,value in {"info":info,"modality":modality,"stats":stats,
                "provenance":dict(source="simulation",real_robot_data=False,vla_weights_used=False,partition=partition,
                    source_episodes=[r["episode_id"] for r in episodes],normalization_fit_episodes=[r["episode_id"] for r in groups["train"]],
                    gr00t_loader_validated=False,selection="terminal success then quota; not a policy evaluation set")}.items():write_json(root/"meta"/(name+".json"),value)
            for name,lines in [("episodes",episode_meta),("tasks",[dict(task_index=0,task="Use both arms to transfer the box.")])]:
                with (root/"meta"/(name+".jsonl")).open("x",encoding="utf-8",newline="\n") as f:
                    for line in lines:f.write(json.dumps(line)+"\n")
    finally:
        for c in cameras.values():c.close()
    write_json(output/"calibration.json",dict(source="simulation",real_camera_calibration=False,cameras=calibration))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",required=True,type=Path);p.add_argument("--episodes",type=int,default=220)
    p.add_argument("--keep",type=int,default=174);p.add_argument("--seed",type=int,default=20261019)
    p.add_argument("--no-export",action="store_true")
    a=p.parse_args();collect(a.output_dir,episodes=a.episodes,keep=a.keep,seed=a.seed,export=not a.no_export)
