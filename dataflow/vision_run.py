"""Run the control baseline with RGB-D box measurements instead of pose truth.

CEM remains the action source. Ground truth is read by the evaluator only, to
measure tracking error; the observe override uses images/depth/calibration.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import numpy as np
from .camera import RGBDCameras
from .rgbd import track
from .collect import fresh, write_json


def run(output, *, scene_id=0, seed=20261019):
    from benchmark.protocol import SuccessEvaluator
    from benchmark.runner import telemetry
    from work.g1_objectcentric.task import ObjectCentricTask
    from work.g1_objectcentric.scenarios import suite
    if not 0<=scene_id<24:raise ValueError("Unknown scene")
    class VisionTask(ObjectCentricTask):
        def observe(self):
            now=float(self.elapsed)
            if now+1e-9>=self.next_image:
                images=self.camera.frame(self.data.qpos)
                if self.scenario.dropout_start_s<=now<self.scenario.dropout_start_s+self.scenario.dropout_duration_s:
                    images={name:(np.zeros_like(rgb),depth) for name,(rgb,depth) in images.items()}
                measurement=track(images,self.camera.calibration,half_size=np.array([.065,.08,.05])*self.scenario.size_scale,timestamp=now)
                if measurement is not None:
                    self.vision_history.append(measurement)
                    self.measurements.append(measurement)
                self.next_image=now+.1
            valid=[m for m in self.vision_history if m.timestamp<=now-self.scenario.pose_delay_s+1e-9]
            if not valid:return None
            m=valid[-1];pose=np.eye(4);pose[:3,3]=m.position
            # This controller consumes position feedback; orientation is not a
            # measured full 6DoF pose and is deliberately not used for control.
            return m.timestamp,pose
    output=fresh(output)
    write_json(output/"manifest.json",dict(source="simulation",action_source="CEM",sensor_source="rendered RGB and metric depth",
        real_robot=False,gr00t_weights_used=False,scene_id=scene_id,seed=seed,
        assumptions=["known orange box","known size","upright top face","calibrated simulated cameras"],
        full_6dof_visual_tracking=False))
    env=VisionTask();env.reset(seed=seed,mode="feedback",scenario=suite()[scene_id])
    env.camera=RGBDCameras(env.model,320,240);env.next_image=0.;env.vision_history=deque(maxlen=64);env.measurements=[]
    evaluator=SuccessEvaluator(float(env.initial_z));evaluator.update(telemetry(env))
    rows=[];errors=[];previous=0
    try:
        while True:
            running=env.step()
            if len(env.measurements)>previous:
                m=env.measurements[-1]
                # Evaluation only: truth is never passed back to the observer.
                truth=env.object_pose()[:3,3]
                error=float(np.linalg.norm(m.position-truth))
                errors.append(error);rows.append(dict(source="simulation",capture_sim_s=m.timestamp,
                    position_from_rgbd=m.position.tolist(),yaw_mod_pi=m.yaw_mod_pi,points=m.points,
                    evaluation_only_truth=truth.tolist(),evaluation_error_m=error))
                previous=len(env.measurements)
            if env.steps%5==0 or not running:evaluator.update(telemetry(env))
            if not running:break
    finally:env.camera.close()
    with (output/"tracking.jsonl").open("x",encoding="utf-8",newline="\n") as f:
        for row in rows:f.write(json.dumps(row)+"\n")
    result=dict(source="simulation",real_robot=False,gr00t_weights_used=False,action_source="CEM",
        sensor_source="RGB-D segmentation and backprojection; no current object-pose truth fed to observe",
        measurements=len(errors),mean_tracking_error_cm=float(np.mean(errors)*100) if errors else None,
        max_tracking_error_cm=float(max(errors)*100) if errors else None,task_result=evaluator.result(),
        stale_pose_steps=env.stale_steps,full_6dof_visual_tracking=False)
    write_json(output/"summary.json",result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--scene",type=int,default=0);p.add_argument("--seed",type=int,default=20261019)
    a=p.parse_args();print(json.dumps(run(a.output_dir,scene_id=a.scene,seed=a.seed),indent=2))
