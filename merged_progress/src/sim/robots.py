"""Robot configuration abstraction — makes the stack robot-agnostic (G1 + Talos).

The controller / WBC / planner only need a small set of model facts: the base
and torso bodies, the foot sites and the 4 contact-corner sites per foot, the
home keyframe pose, and a bent-knee crouch. This module supplies them per robot
and handles the model-loading differences:

  * G1   — bundled torque g1.xml with a baked bent-knee 'stand' keyframe and
           corner sites already in the model.
  * Talos— pal_talos scene_motor.xml (already torque); BOX feet, so we add 4
           contact-corner sites at the foot-box bottom via mjSpec at load time,
           and put it in a bent-knee crouch numerically (no baked keyframe).
"""
import numpy as np
import mujoco

from src.utils.config import load_params, resolve


def g1_model_cfg(params):
    """G1 config = the existing params['model'] block (+ ankle joint names)."""
    cfg = dict(params["model"])
    cfg.setdefault("ankle_pitch_joints",
                   ["left_ankle_pitch_joint", "right_ankle_pitch_joint"])
    return cfg


# Talos foot box (leg_*_6_link): geom pos [-0.005,0,-0.1], half-size [0.1,0.06,0.006].
# Bottom corners (z = -0.106) relative to the leg_6_link body frame.
_TALOS_FOOT_BOX = dict(cx=-0.005, hx=0.10, hy=0.06, zc=-0.106)
_TALOS_LEG = ["1", "2", "3", "4", "5", "6"]  # hip_yaw,hip_roll,hip_pitch,knee,ank_pitch,ank_roll


def talos_model_cfg():
    return {
        "scene_flat": "mujoco_menagerie-pal_talos/scene_motor.xml",
        "scene_incline": "mujoco_menagerie-pal_talos/scene_motor.xml",
        "keyframe": None,                      # built numerically (crouch)
        "base_body": "base_link",
        "torso_body": "torso_2_link",
        "feet": {
            "left":  {"site": "left_foot",  "body": "leg_left_6_link",
                       "corners": ["talos_lf_hl", "talos_lf_hr", "talos_lf_tl", "talos_lf_tr"]},
            "right": {"site": "right_foot", "body": "leg_right_6_link",
                       "corners": ["talos_rf_hl", "talos_rf_hr", "talos_rf_tl", "talos_rf_tr"]},
        },
        # bent-knee crouch (legs) + "holding a tray" arms (elbow bent forward).
        # leg: 3=hip_pitch, 4=knee, 5=ankle_pitch ; arm: 2=shoulder_roll, 4=elbow.
        "crouch": {f"leg_{s}_3_joint": -0.45 for s in ("left", "right")}
                  | {f"leg_{s}_4_joint": 0.85 for s in ("left", "right")}
                  | {f"leg_{s}_5_joint": -0.42 for s in ("left", "right")}
                  | {f"arm_{s}_2_joint": 0.20 for s in ("left", "right")}
                  | {f"arm_{s}_4_joint": -1.60 for s in ("left", "right")},
        "ankle_pitch_joints": ["leg_left_5_joint", "leg_right_5_joint"],
    }


def _add_talos_corner_sites(spec):
    """Add 4 contact-corner sites at the foot-box bottom of each Talos foot."""
    b = _TALOS_FOOT_BOX
    corners = {  # name-suffix -> (x, y)
        "hl": (b["cx"] - b["hx"], +b["hy"]), "hr": (b["cx"] - b["hx"], -b["hy"]),
        "tl": (b["cx"] + b["hx"], +b["hy"]), "tr": (b["cx"] + b["hx"], -b["hy"]),
    }
    for side, sl in (("left", "lf"), ("right", "rf")):
        body = spec.body(f"leg_{side}_6_link")
        for suf, (x, y) in corners.items():
            s = body.add_site()
            s.name = f"talos_{sl}_{suf}"
            s.pos = [x, y, b["zc"]]
            s.size = [0.005, 0.005, 0.005]


def model_cfg(robot, params=None):
    params = params or load_params()
    return g1_model_cfg(params) if robot == "g1" else talos_model_cfg()


def load_robot_model(robot, params, terrain=None, scene_key="scene_flat", decorate=None):
    """Compile the robot model (G1 or Talos) with optional terrain. Returns
    (model, mcfg). Talos gets contact-corner sites injected via mjSpec.
    ``decorate(spec, mcfg)`` is an optional hook to attach extra geoms (e.g. a
    tray + frappe to the torso) before compiling."""
    mcfg = model_cfg(robot, params)
    scene = str(resolve(mcfg[scene_key]))
    if robot == "talos" or terrain is not None or decorate is not None:
        spec = mujoco.MjSpec.from_file(scene)
        if robot == "talos":
            _add_talos_corner_sites(spec)
        if terrain is not None:
            terrain.apply(spec)
        if decorate is not None:
            decorate(spec, mcfg)
        model = spec.compile()
    else:
        model = mujoco.MjModel.from_xml_path(scene)
    return model, mcfg
