"""Stationary automatic ETH acquisition using existing SDK planning/control.

Planning and saved FK use the original URDF and RAW angles, never the previous
hand-eye result or fitted offsets. No camera, CAN or motor is opened here.
"""
from collections import deque
from collections import Counter
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
from scipy.spatial.transform import Rotation

from .click_hover import HoverPlanner, import_sdk, load_config, in_workspace
from .geometry import checked, diversity, errors
from .kinematics import CorrectedKinematics, JOINT_NAMES
from .gravity_parameters import configure_gravity_model_exclusions,validate_gravity_link_mass_reductions


class CollectionPlanner(HoverPlanner):
    def __init__(self, cfg):
        import pinocchio as pin
        self.pin, self.cfg, self.lock = pin, cfg, threading.Lock()
        self.ik = import_sdk(cfg['sdk_root'])
        urdf = Path(cfg['collection_urdf']).resolve()
        self.robot_metadata = dict(source='SDK-native-MIT-snapshot-stationary',
            urdf=str(urdf), urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
            base_frame='base_link', ee_frame='gripper_end', joint_names=JOINT_NAMES,
            channel=cfg['channel'], kinematic_model='nominal_raw_angles_no_offsets')
        source = CorrectedKinematics(self.robot_metadata)
        self.offsets = np.asarray(cfg.get('collection_planning_offsets_rad',[0.]*6),float)
        if (self.offsets.shape!=(6,) or not np.isfinite(self.offsets).all()
                or np.any(np.abs(self.offsets)>np.radians(5.))):
            raise ValueError('Invalid collection planning offsets')
        self._initialize_robot_model(source)
        self._configure_collection_gravity(urdf)
        self.robot_metadata['gravity_parameters'] = dict(self.gravity,
            excluded_urdf_links=self.gravity_exclusion['links'],
            removed_urdf_mass_kg=self.gravity_exclusion['removed_mass_kg'],
            scope='Configured point masses with official SDK gravity feedforward; nominal RAW-angle model')
        self.robot_metadata['gravity_model'] = self.gravity_exclusion

    def _configure_collection_gravity(self,urdf):
        configure_gravity_model_exclusions(self,urdf,
            self.cfg.get('collection_gravity_excluded_urdf_links',[]),
            self.cfg.get('collection_gravity_link_mass_reductions_kg',{}))

    def nominal_fk(self, raw):
        # Capture FK must retain orientation, unlike HoverPlanner.tcp().
        return self._capture_fk(raw,np.zeros(6))

    def planning_fk(self, raw):
        return self._capture_fk(raw,self.offsets)

    def _capture_fk(self, raw, offsets):
        with self.lock:
            self.pin.framesForwardKinematics(self.model, self.data, self.validate_feedback(raw)+offsets)
            return checked(self.data.oMf[self.tcp_id].homogeneous.copy())

    def check_height(self,plan,cancelled=lambda:False):
        for q in plan.points:
            if cancelled():raise InterruptedError('Collection planning cancelled')
            if self.tcp(q)[2] < self.cfg['collection_min_tcp_z_m']-1e-6:
                raise ValueError('Path below collection minimum TCP height')
        return plan

    def plan_home(self,raw,cancelled=lambda:False):
        return self.check_height(super().plan_home(raw,cancelled),cancelled)


def apply_sampling_region(poses,region):
    """Map source poses into the configured visible region, once, in base metres.

    This changes desired sampling locations only, never recorded FK or fitted
    extrinsics. Retain source coordinates for repeatable configuration edits.
    """
    if region is None:return [dict(row,xyz_rpy=list(row['xyz_rpy'])) for row in poses]
    if not isinstance(region,dict):raise ValueError('Invalid sampling_region')
    vectors=dict(offset_m=3,low_offset_m=3,normal_x_range_m=2,low_x_range_m=2,
                 y_range_m=2,z_range_m=2)
    for name,count in vectors.items():
        values=np.asarray(region[name],float)
        if values.shape!=(count,) or not np.isfinite(values).all():
            raise ValueError('Invalid sampling_region.'+name)
        if name.endswith('range_m') and values[0]>=values[1]:
            raise ValueError('Reversed sampling_region.'+name)
        if name.endswith('offset_m') and np.max(np.abs(values))>.2:
            raise ValueError('Sampling offset exceeds 0.2 m')
    for name,lo,hi in (('low_pose_max_z_m',.12,.3),('downward_pitch_min_deg',30.,90.)):
        value=region[name]
        if not isinstance(value,(int,float)) or not np.isfinite(value) or not lo<=value<=hi:
            raise ValueError('Invalid sampling_region.'+name)
    mapped=[]
    for row in poses:
        source=np.asarray(row.get('source_xyz_rpy',row['xyz_rpy']),float)
        if source.shape!=(6,) or not np.isfinite(source).all():raise ValueError('Invalid source pose')
        low=source[2]<=region['low_pose_max_z_m'] and source[4]>=np.radians(region['downward_pitch_min_deg'])
        xyzrpy=source.copy();xyzrpy[:3]+=np.asarray(region['low_offset_m' if low else 'offset_m'])
        xr=region['low_x_range_m' if low else 'normal_x_range_m']
        xyzrpy[:3]=np.clip(xyzrpy[:3],[xr[0],region['y_range_m'][0],region['z_range_m'][0]],
                          [xr[1],region['y_range_m'][1],region['z_range_m'][1]])
        mapped.append(dict(row,xyz_rpy=xyzrpy.tolist(),source_xyz_rpy=source.tolist(),
                           sampling_region_kind='low_downward' if low else 'normal'))
    return mapped


def load_settings(path, split='train', *, mount='external'):
    if mount not in ('external','wrist'):raise ValueError('Unknown camera mount')
    path = Path(path).resolve()
    settings = json.loads(path.read_text())
    settings['_path'] = str(path)
    base = (path.parent/settings['base_config']).resolve()
    cfg = load_config(base, overrides=dict(
        result='unused_for_collection.json',
        gravity_config=str((path.parent/settings['gravity_config']).resolve()),
        serial=settings['serial'], tcp_offset_m=[0., 0., 0.],tcp_frame='gripper_end',
        table_reference=None, height_compensation_m=0.,height_reference='table',hover_m=.1,
        planning_mode='sdk_cartesian',
        cartesian_outer_loop=dict(enabled=False),
        settle_tolerance_m=.012, settle_timeout_s=5., home_position_tolerance_m=.009,
        home_tolerance_rad=.016))
    cfg['collection_urdf'] = str((path.parent/settings['urdf']).resolve())
    cfg['camera_fps'] = settings['fps']
    cfg.pop('result',None)
    cfg.pop('table_reference',None)
    # An original URDF is required: fitted offsets must not be baked into it.
    original = Path(cfg['sdk_root'])/'urdf/RS/urdf/ReBot_Arm_RS.urdf'
    if mount == 'external' and Path(cfg['collection_urdf']).read_bytes() != original.read_bytes():
        raise ValueError('Collection URDF must equal the original SDK URDF')
    poses = apply_sampling_region(settings[split+'_poses'],settings.get('sampling_region'))
    serial='332322072170' if mount=='wrist' else '250122070127'
    if settings['fps'] != 30 or settings['serial'] != serial:
        raise ValueError(f'This configuration requires {mount} camera {serial} at 30 Hz')
    if mount=='wrist':
        cfg['manual_disable_only']=True
        cfg['home_completion_mode']='trajectory'
    ranges = dict(stationary_s=(.5, 3.), static_mm=(.1, 2.), static_deg=(.01, .5),
                  joint_span_rad=(.0001, .004), observation_timeout_s=(2., 30.),
                  max_pnp_px=(.05, 1.), ambiguity_gap_px=(.01, 1.),
                  min_tcp_z_m=(.06 if mount=='wrist' else .10, .30), min_move_duration_s=(3., 15.))
    defaults=dict(max_frame_gap_s=.12,vision_static_mm=3.,vision_static_deg=2.,gripper_span_rad=.005)
    settings={**defaults,**settings}
    ranges.update(max_frame_gap_s=(.05,.2),vision_static_mm=(.5,5.),
                  vision_static_deg=(.5,5.),gripper_span_rad=(.001,.02))
    for key,(low,high) in ranges.items():
        v = settings[key]
        if isinstance(v,bool) or not np.isfinite(v) or not low <= v <= high:
            raise ValueError('Invalid collection setting: '+key)
    if not poses or len(poses) > 150:
        raise ValueError('Need 1..150 target poses')
    for row in poses:
        pose=np.asarray(row['xyz_rpy'],float)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise ValueError('Each pose needs finite XYZ metres and RPY radians')
        if not in_workspace(pose[:3],cfg) or pose[2] < settings['min_tcp_z_m']:
            raise ValueError('Target pose violates configured collection workspace')
    target=settings['target']
    marker_m=target.get('marker_m') if mount=='wrist' else .06
    if mount=='wrist' and marker_m not in (.06,.1):
        raise ValueError('Wrist collection target must be measured 60 or 100 mm')
    if target != dict(type='aruco',dictionary='DICT_4X4_50',id=0,marker_m=marker_m):
        raise ValueError(f'Expected {marker_m*1000:g} mm DICT_4X4_50 ID 0 target')
    if mount=='wrist':settings['mount']='wrist'
    settings['split']=split
    failures=settings.setdefault('max_consecutive_capture_failures',3)
    if type(failures) is not int or not 0<=failures<=10:
        raise ValueError('max_consecutive_capture_failures must be 0..10 (0 disables automatic capture-failure pause)')
    cfg['collection_min_tcp_z_m']=settings['min_tcp_z_m']
    links=settings.get('gravity_excluded_urdf_links',[])
    if (not isinstance(links,list) or any(not isinstance(name,str) for name in links) or
            len(set(links))!=len(links) or
            not set(links)<= {'camera_mount_link','camera_link'}):
        raise ValueError('Only absent wrist-camera URDF links may be excluded from collection gravity')
    cfg['collection_gravity_excluded_urdf_links']=links.copy()
    reductions=validate_gravity_link_mass_reductions(settings.get('gravity_link_mass_reductions_kg'))
    if reductions:cfg['collection_gravity_link_mass_reductions_kg']=reductions
    return settings,cfg,poses


def endpoint(planner, raw, row,cancelled=lambda:False):
    xyzrpy=np.asarray(row['xyz_rpy'],float)
    pose=planner.pin.SE3(Rotation.from_euler('xyz',xyzrpy[3:]).as_matrix(),xyzrpy[:3])
    seeds=[np.asarray(raw).copy()]
    for elbow in ((.7,.5),(1.4,1.),(2.,1.9),(2.3,2.2)):
        q=seeds[0].copy();q[1:3]=elbow
        seeds.append(np.clip(q,planner.raw_lower,planner.raw_upper))
    solutions=[]
    data=planner.model.createData()
    for seed in seeds:
        if cancelled():raise InterruptedError('Collection IK cancelled')
        result=planner.ik.solve_ik(planner.model,data,planner.tcp_id,pose,seed+planner.offsets,
            planner.ik.IKParams(max_iter=1000,tolerance=1e-7),controlled_joints=6,
            position_only=False)
        if result.success:
            try:q=planner.validate_raw(result.q-planner.offsets)
            except ValueError:continue
            if max(errors(planner.planning_fk(q),pose.homogeneous)) < .001:
                solutions.append(q)
    if not solutions:
        raise ValueError('Official full-pose IK cannot reach this pose')
    return min(solutions,key=lambda q:np.linalg.norm(q-raw))


def collection_plan(planner,raw,row,settings,cancelled=lambda:False):
    """Same official SE3 MIN_JERK + CLIK as reference move_to_traj.

    Reuse the existing fully checked SDK sampler; keep every point and the sole
    MIT sender. No Cartesian outer loop is applied during calibration capture.
    """
    from .sdk_hover import hover_plan
    raw=planner.command_from_feedback(raw)
    goal=endpoint(planner,raw,row,cancelled)
    plan=hover_plan(planner,raw,np.asarray(row['xyz_rpy'][:3]),cancelled,endpoint_goal=goal)
    plan.kind='calibration'
    plan.tracking_outer_loop=False
    # Time stretching preserves all SDK samples, never speeds up a path.
    old_duration=plan.duration
    plan.duration=max(old_duration,settings['min_move_duration_s'])
    scale=old_duration/plan.duration
    for name in ('nominal_peak_joint_speed_rad_s','nominal_peak_tcp_speed_m_s'):
        plan.diagnostics[name]*=scale
    plan.diagnostics['nominal_peak_joint_accel_rad_s2']*=scale**2
    plan.diagnostics.update(duration_s=plan.duration, collection_raw_nominal=True,
        target_update_hz=len(plan.points)/plan.duration,
        replay_sample_interval_s=plan.duration/len(plan.points))
    return planner.check_height(plan,cancelled)


def preflight(planner,poses,settings):
    """Compile ALL inter-pose paths offline, not merely isolated IK endpoints."""
    raw=np.zeros(6);accepted=[];rejected=[]
    for i,row in enumerate(poses):
        try:
            # Clamping a source grid to a narrower field may collapse targets.
            candidate=np.eye(4);candidate[:3,3]=row['xyz_rpy'][:3]
            candidate[:3,:3]=Rotation.from_euler('xyz',row['xyz_rpy'][3:]).as_matrix()
            for old in accepted:
                previous=np.eye(4);previous[:3,3]=old['xyz_rpy'][:3]
                previous[:3,:3]=Rotation.from_euler('xyz',old['xyz_rpy'][3:]).as_matrix()
                mm,deg=errors(candidate,previous)
                if mm<5 and deg<5:raise ValueError('Duplicate effective target after sampling-region adjustment')
            plan=collection_plan(planner,raw,row,settings)
            home=planner.plan_home(plan.goal)
            accepted.append(dict(index=i,**row,goal_raw_rad=plan.goal.tolist(),
                duration_s=plan.duration,sample_count=len(plan.points),
                home_prechecked=True,home_duration_s=home.duration))
            raw=plan.goal.copy()
        except ValueError as exc:
            rejected.append(dict(index=i,**row,error=str(exc)))
    return dict(hardware_accessed=False,split=settings['split'],accepted=accepted,
                rejected=rejected,estimated_motion_seconds=sum(r['duration_s'] for r in accepted),
                note='Nominal zero-start preflight only; execution replans from fresh measured angles. '
                     'TCP workspace checks are not full-arm or environment collision checking.')


class StationaryWindow:
    def __init__(self,settings):
        self.settings=settings;self.rows=deque();self.reason='Waiting for stationary window';self.ready=False

    def clear(self):
        self.rows.clear();self.ready=False;self.reason='Waiting for stationary window'

    def add(self,frame,pose,meta,observation):
        self.ready=False
        if self.rows and frame.acquired <= self.rows[-1][0].acquired:
            return False
        if self.rows and frame.acquired-self.rows[-1][0].acquired > self.settings['max_frame_gap_s']:
            self.clear();self.reason='Frame gap: restarting stationary window'
        self.rows.append((frame,pose,meta,observation))
        duration=self.settings['stationary_s']
        while len(self.rows)>2 and frame.acquired-self.rows[1][0].acquired >= duration:
            self.rows.popleft()
        if len(self.rows)<5 or frame.acquired-self.rows[0][0].acquired < duration:
            return False
        movements=[errors(self.rows[0][1],r[1]) for r in self.rows]
        span=np.max(movements,axis=0)
        qs=np.asarray([r[2]['q_rad'] for r in self.rows])
        if (span[0]>self.settings['static_mm'] or span[1]>self.settings['static_deg']
                or np.max(np.ptp(qs,axis=0))>self.settings['joint_span_rad']):
            self.reason='Robot not stationary';return False
        if all('gripper_q_rad' in r[2] for r in self.rows):
            if np.max(np.ptp([r[2]['gripper_q_rad'] for r in self.rows],axis=0)) > self.settings['gripper_span_rad']:
                self.reason='Gripper opening is moving';return False
        uv=np.asarray([r[3]['image_points'] for r in self.rows])
        if np.max(np.linalg.norm(np.ptp(uv,axis=0),axis=1))>2.:
            self.reason='Marker image is moving';return False
        visual=[errors(np.asarray(self.rows[0][3]['T_camera_target']),np.asarray(r[3]['T_camera_target']))
                for r in self.rows]
        mm,deg=np.max(visual,axis=0)
        if mm>self.settings['vision_static_mm'] or deg>self.settings['vision_static_deg']:
            self.reason='Marker pose is unstable';return False
        self.reason='Stationary';self.ready=True;return True


def sample_from_window(window,dataset):
    if not window.ready:raise ValueError('No accepted stationary window')
    frame,g,meta,obs=window.rows[-1]
    for old in dataset['samples']:
        mm,deg=errors(g,np.asarray(old['T_base_ee']))
        if mm<5 and deg<5:
            raise ValueError('Duplicate pose (<5 mm and <5 deg)')
    movements=np.asarray([errors(window.rows[0][1],r[1]) for r in window.rows])
    return dict(T_base_ee=g.tolist(),**obs,camera_frame=frame.metadata,
        robot_reads=[r[2] for r in window.rows],static_span_mm=float(movements[:,0].max()),
        static_span_deg=float(movements[:,1].max()),
        stationary_frames=len(window.rows),stationary_duration_s=frame.acquired-window.rows[0][0].acquired,
        synchronization='Stationary acquisition; host timestamps recorded, not hardware-trigger synchronized')


def coverage(dataset):
    samples=dataset['samples']
    validation=dataset.get('acquisition',{}).get('split')=='validation'
    minimum=15 if validation else 40
    result=dict(sample_count=len(samples),minimum_samples=minimum,physical_accuracy_verified=False)
    if samples:
        qs=np.degrees([s['robot_reads'][-1]['q_rad'] for s in samples])
        result.update(joint_min_deg=np.min(qs,axis=0).tolist(),joint_max_deg=np.max(qs,axis=0).tolist(),
                      tcp_span_m=np.ptp([np.asarray(s['T_base_ee'])[:3,3] for s in samples],axis=0).tolist(),
                      groups=dict(Counter(s.get('requested_pose',{}).get('group','unknown') for s in samples)))
    try:
        result['diversity']=diversity([np.asarray(s['T_base_ee']) for s in samples]) if len(samples)>=3 else None
    except ValueError as exc:
        result['diversity_error']=str(exc)
    ready='READY_FOR_OFFLINE_VALIDATION' if validation else 'READY_FOR_OFFLINE_FIT'
    result['status']=(ready if len(samples)>=minimum and result.get('diversity')
                      else 'REVIEW_COVERAGE_OR_COLLECT_MORE')
    planned=dataset.get('acquisition',{}).get('planned_groups',{})
    result['planned_groups']=planned
    result['undersampled_groups']=[name for name,count in planned.items()
        if result.get('groups',{}).get(name,0)<min(3,count)]
    if result['status']==ready and result['undersampled_groups']:
        result['status']='REVIEW_MISSING_POSE_GROUPS'
    return result
