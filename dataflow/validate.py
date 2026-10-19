"""Recompute dataset counts, hashes, partitions, normalization and video lengths."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from .collect import digest, train_stats, write_json


def stored_path(root, name):
    """Read manifests collected on either Windows or POSIX hosts."""
    path=(Path(root)/str(name).replace("\\","/")).resolve()
    if Path(root).resolve() not in path.parents:raise ValueError("Escaped artifact path")
    return path


def validate(root, *, videos=True):
    from benchmark.protocol import Criteria, SuccessEvaluator
    import pyarrow.parquet as pq
    root=Path(root).resolve()
    rows=[json.loads(s) for s in (root/"episodes.jsonl").read_text(encoding="utf-8").splitlines() if s]
    summary=json.loads((root/"summary.json").read_text())
    inventory=root/"SHA256.json"
    if inventory.exists():
        for name,expected in json.loads(inventory.read_text())["files"].items():
            path=(root/name).resolve()
            if root not in path.parents or digest(path)!=expected:raise ValueError("Dataset inventory hash mismatch")
    if len({r["episode_id"] for r in rows})!=len(rows):raise ValueError("Duplicate episode IDs")
    frames=0; accepted={};eligible=0
    for row in rows:
        if row["source"]!="simulation" or type(row["accepted"]) is not bool or type(row["eligible"]) is not bool:
            raise ValueError("Invalid source or acceptance annotation")
        if row["accepted"] and not row["eligible"]:raise ValueError("Ineligible episode accepted")
        if not row["accepted"] and not row.get("exclusion_reason"):raise ValueError("Missing exclusion reason")
        for artifact in row["artifacts"].values():
            p=stored_path(root,artifact["path"])
            if root not in p.parents or not p.is_file() or digest(p)!=artifact["sha256"]:
                raise ValueError("Missing, escaped or changed artifact")
        with np.load(stored_path(root,row["artifacts"]["trajectory"]["path"]),allow_pickle=False) as a:
            n=len(a["timestamp"])
            if str(a["source"])!="simulation" or bool(a["real_robot_data"]):raise ValueError("Wrong trajectory provenance")
            if n!=row["frames"] or a["state"].shape!=(n,26) or a["action"].shape!=(n,26):raise ValueError("Episode/frame confusion")
            if not all(np.isfinite(a[k]).all() for k in ("timestamp","state","action","qpos")):raise ValueError("Nonfinite trajectory")
            if n<2 or not np.all(np.diff(a["timestamp"])>0):raise ValueError("Invalid frame timeline")
        frames+=row["frames"];eligible+=row["eligible"]
        replay=SuccessEvaluator(row["initial_z_m"],Criteria(**row["criteria"]))
        with gzip.open(stored_path(root,row["artifacts"]["telemetry"]["path"]),"rt",encoding="utf-8") as trace:
            for line in trace:replay.update(json.loads(line))
        calculated_result=replay.result()
        # The collection evaluator also saw t=0; stored telemetry starts at
        # 0.01 s in REACH, before any grasp or terminal stability interval.
        for key,value in calculated_result.items():
            recorded=row["task_result"][key]
            if key=="samples":
                if value+1!=recorded:raise ValueError("Telemetry sample count changed")
            elif isinstance(value,bool):
                if value!=recorded:raise ValueError("Replayed terminal eligibility differs")
            elif abs(value-recorded)>1e-9:raise ValueError("Replayed physical result differs")
        if row["accepted"]:accepted[row["episode_id"]]=row
    calculated={"planned_episodes":len(rows),"eligible_episodes":eligible,"accepted_episodes":len(accepted),
        "rejected_episodes":len(rows)-len(accepted),"total_frames":frames,
        "accepted_frames":sum(r["frames"] for r in accepted.values())}
    if any(summary[k]!=v for k,v in calculated.items()):raise ValueError("Summary does not match physical files")
    partitions={};parsed={};decoded=0
    for directory in sorted((root/"datasets").glob('*')):
        if not directory.is_dir():continue
        info=json.loads((directory/"meta/info.json").read_text())
        provenance=json.loads((directory/"meta/provenance.json").read_text())
        records=[json.loads(s) for s in (directory/"meta/episodes.jsonl").read_text().splitlines() if s]
        ids=[r["source_episode"] for r in records]
        if len(set(ids))!=len(ids) or set(ids)-set(accepted):raise ValueError("Duplicate or unknown source episode")
        if ids!=provenance["source_episodes"] or provenance["source"]!="simulation" or provenance["real_robot_data"]:
            raise ValueError("Invalid partition provenance")
        partitions[directory.name]=ids;parsed[directory.name]=[];n_frames=0
        for record in records:
            ep=record["episode_index"]
            path=directory/"data/chunk-000"/f"episode_{ep:06d}.parquet"
            table=pq.read_table(path).to_pylist()
            if len(table)!=record["length"] or len(table)!=accepted[record["source_episode"]]["frames"]:
                raise ValueError("Parquet frame count changed")
            for i,row in enumerate(table):
                if (row["episode_index"]!=ep or row["frame_index"]!=i or row["index"]!=n_frames+i
                    or abs(row["timestamp"]-i/10)>1e-5 or len(row["observation.state"])!=26 or len(row["action"])!=26):
                    raise ValueError("Parquet schema or timeline mismatch")
            parsed[directory.name].extend(table)
            n_frames+=len(table)
            if videos:
                import imageio_ffmpeg
                for name in ("front","side"):
                    reader=imageio_ffmpeg.read_frames(str(directory/"videos/chunk-000"/("observation.images."+name)/f"episode_{ep:06d}.mp4"))
                    next(reader)
                    if sum(1 for _ in reader)!=len(table):raise ValueError("Video length differs from trajectory")
                    decoded+=1
        if n_frames!=info["total_frames"] or len(records)!=info["total_episodes"]:raise ValueError("Dataset info count mismatch")
    if partitions:
        merged=[identifier for ids in partitions.values() for identifier in ids]
        if len(set(merged))!=len(merged) or set(merged)!=set(accepted):raise ValueError("Partition overlap or missing selected data")
        training=parsed["train"]
        for name in (root/"datasets").iterdir():
            if not name.is_dir():continue
            p=json.loads((name/"meta/provenance.json").read_text())
            if p["normalization_fit_episodes"]!=partitions["train"]:raise ValueError("Normalization leakage")
            s=json.loads((name/"meta/stats.json").read_text())
            for key in ("observation.state","action"):
                expected=train_stats([np.array([r[key] for r in training],dtype=np.float32)])
                if any(not np.allclose(s[key][k],v,atol=1e-7,rtol=1e-6) for k,v in expected.items()):
                    raise ValueError("Statistics not computed exclusively from training episodes")
    return dict(passed=True,source="simulation",real_robot_data=False,**calculated,
        partition_episodes={k:len(v) for k,v in partitions.items()},decoded_videos=decoded,
        official_gr00t_loader_tested=False,checks=["all artifact hashes","episode/frame counts","complete partitions",
            "physical success replay","train-only normalization","Parquet timelines"]+(["full video decode/frame counts"] if videos else []))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path)
    p.add_argument("--no-videos",action="store_true");p.add_argument("--output",type=Path)
    a=p.parse_args();report=validate(a.root,videos=not a.no_videos)
    if a.output:write_json(a.output,report)
    print(json.dumps(report,indent=2))
