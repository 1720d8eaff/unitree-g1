"""Known orange upright box tracker from RGB segmentation and metric depth.

This narrow simulation baseline assumes known box dimensions and camera
calibration. It is not a general learned detector or 6DoF real-object tracker.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BoxMeasurement:
    timestamp: float
    position: np.ndarray
    yaw_mod_pi: float
    points: int
    extent_xy: tuple


def track(frames, calibration, *, half_size=(.065,.08,.05), timestamp=0.):
    import cv2
    size=np.asarray(half_size,dtype=float)
    if size.shape!=(3,) or np.any(size<=0) or not np.isfinite(size).all():raise ValueError("Need known positive box half-size")
    if not np.isfinite(timestamp) or timestamp<0:raise ValueError("Invalid capture time")
    clouds=[]
    for name,(rgb,depth) in frames.items():
        rgb=np.asarray(rgb);depth=np.asarray(depth)
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or depth.shape!=rgb.shape[:2]:
            raise ValueError("Need paired H,W RGB8 and metric depth")
        c=calibration[name];k=np.asarray(c["K"],dtype=float);t=np.asarray(c["T_world_camera_opengl"],dtype=float)
        if k.shape!=(3,3) or t.shape!=(4,4) or not np.isfinite(k).all() or not np.isfinite(t).all() or min(k[0,0],k[1,1])<=0:
            raise ValueError("Invalid camera calibration")
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        mask=cv2.inRange(hsv,np.array([5,90,45]),np.array([40,255,255])).astype(bool)
        mask&=np.isfinite(depth)&(depth>.05)&(depth<5.)
        v,u=np.nonzero(mask)
        if len(v)<10:continue
        d=depth[v,u]
        camera=np.column_stack(((u-k[0,2])*d/k[0,0],-(v-k[1,2])*d/k[1,1],-d))
        clouds.append(camera@t[:3,:3].T+t[:3,3])
    if not clouds:return None
    points=np.concatenate(clouds)
    if len(points)<30:return None
    top_z=float(np.quantile(points[:,2],.98))
    top=points[np.abs(points[:,2]-top_z)<.006]
    if len(top)<15:return None
    rectangle=cv2.minAreaRect(top[:,:2].astype(np.float32))
    corners=cv2.boxPoints(rectangle)
    edges=np.roll(corners,-1,axis=0)-corners
    lengths=np.linalg.norm(edges,axis=1)
    if np.any(np.sort(lengths)[:2]<.5*min(size[:2])*2) or np.max(lengths)>1.3*max(size[:2])*2:return None
    index=int(np.argmin(np.abs(lengths-2*size[0])))
    yaw=float((np.arctan2(edges[index,1],edges[index,0])+np.pi/2)%np.pi-np.pi/2)
    return BoxMeasurement(timestamp,np.array([*rectangle[0],top_z-size[2]]),yaw,len(points),tuple(rectangle[1]))
