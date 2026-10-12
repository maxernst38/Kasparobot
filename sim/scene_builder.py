"""Builds the MuJoCo XML for the scene.

Generated from code so the 40 piece bodies, the arm and the camera stay
consistent with `sim.board_layout`.  `write_scene` drops the XML into
sim/assets/, where `simulate` can open it.
"""

import pathlib

from . import board_layout as L
from . import board_texture
from . import piece_meshes

ASSETS = pathlib.Path(__file__).parent / "assets"
SCENE_PATH = ASSETS / "chess_scene.xml"
PLAIN_SCENE_PATH = ASSETS / "chess_scene_plain.xml"   # no type markers

# Spare pieces parked off board, used when a pawn promotes.
STOCK_PIECES = ("q", "r", "b", "n")


def _rgba(v):
    return " ".join(f"{c:g}" for c in v)


def _head_geoms(ptype):
    """Geoms above the stem that give each type its silhouette."""
    if ptype == "p":
        return [f'<geom type="sphere" size="0.010" pos="0 0 0.038"/>']
    if ptype == "n":
        return [f'<geom type="box" size="0.008 0.013 0.011" pos="0 0 0.0425"'
                f' euler="0.25 0 0"/>']
    if ptype == "b":
        return ['<geom type="capsule" fromto="0 0 0.030 0 0 0.046" size="0.0085"/>',
                '<geom type="sphere" size="0.004" pos="0 0 0.051"/>']
    if ptype == "r":
        return ['<geom type="cylinder" fromto="0 0 0.030 0 0 0.047" size="0.0125"/>',
                '<geom type="cylinder" fromto="0 0 0.047 0 0 0.050" size="0.0135"/>']
    if ptype == "q":
        return ['<geom type="sphere" size="0.012" pos="0 0 0.042"/>',
                '<geom type="cylinder" fromto="0 0 0.0525 0 0 0.0555" size="0.0135"/>']
    if ptype == "k":
        return ['<geom type="cylinder" fromto="0 0 0.030 0 0 0.046" size="0.0115"/>',
                '<geom type="sphere" size="0.010" pos="0 0 0.0535"/>',
                '<geom type="box" size="0.0015 0.0015 0.004" pos="0 0 0.064"/>',
                '<geom type="box" size="0.005 0.0015 0.0015" pos="0 0 0.0625"/>']
    raise ValueError(ptype)


def _piece_body(name, color, ptype, pos, markers=True):
    """One piece: a turned mesh for looks, primitive geoms for physics.

    MuJoCo collides meshes by their convex hull, which would fill in the waist
    the gripper closes on, so collision stays on the primitives.  Those sit in
    geom group 3, which the renderer does not draw.
    """
    body_rgba = L.WHITE_RGBA if color == "w" else L.BLACK_RGBA
    g = [f'<geom type="cylinder" fromto="0 0 0 0 0 0.008" size="{L.PIECE_BASE_R}"/>',
         '<geom type="cylinder" fromto="0 0 0.008 0 0 0.013" size="0.0135"/>',
         f'<geom type="cylinder" fromto="0 0 {L.NECK_Z0} 0 0 {L.NECK_Z1}"'
         f' size="{L.NECK_R}"/>']
    g += _head_geoms(ptype)
    marker = ""
    if markers:
        top = L.PIECE_HEIGHT[ptype]
        marker = (f'\n      <geom name="{name}_mark" type="cylinder"'
                  f' fromto="0 0 {top:.4f} 0 0 {top + 0.0015:.4f}"'
                  f' size="{L.MARKER_R[ptype]}" rgba="{_rgba(L.MARKER_RGBA[ptype])}"'
                  f' contype="0" conaffinity="0" mass="1e-6"'
                  f' group="0"/>')
    visual = (f'<geom type="mesh" mesh="mesh_{ptype}"'
              f' rgba="{_rgba(body_rgba)}" contype="0" conaffinity="0"'
              f' mass="0" group="1"/>')
    geoms = "\n      ".join(
        [visual] + [s.replace("<geom ", f'<geom group="3" rgba="{_rgba(body_rgba)}" ')
                    for s in g])
    return (f'''    <body name="{name}" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"
          childclass="piece">
      <freejoint name="{name}_free"/>
      {geoms}{marker}
    </body>''')


def _arm():
    jr = L.JOINT_RANGE
    bx, by, bz = L.ARM_BASE
    return f'''    <body name="arm_base" pos="{bx} {by} {bz}">
      <geom type="cylinder" fromto="0 0 0 0 0 {L.PEDESTAL_H}" size="0.055"
            rgba="0.25 0.26 0.28 1" mass="4"/>
      <body name="link1" pos="0 0 {L.PEDESTAL_H}" childclass="arm" gravcomp="1">
        <joint name="base_yaw" axis="0 0 1" range="{jr['base_yaw'][0]} {jr['base_yaw'][1]}"/>
        <geom type="cylinder" fromto="0 0 0 0 0 {L.COLUMN_H}" size="0.040"/>
        <body name="link2" pos="0 0 {L.COLUMN_H}" gravcomp="1">
          <joint name="shoulder" axis="0 1 0" range="{jr['shoulder'][0]} {jr['shoulder'][1]}"/>
          <geom type="sphere" size="0.038" rgba="0.25 0.26 0.28 1"/>
          <geom type="capsule" fromto="0 0 0 0 0 {L.L_UPPER}" size="0.028"/>
          <body name="link3" pos="0 0 {L.L_UPPER}" gravcomp="1">
            <joint name="elbow" axis="0 1 0" range="{jr['elbow'][0]} {jr['elbow'][1]}"/>
            <geom type="sphere" size="0.030" rgba="0.25 0.26 0.28 1"/>
            <geom type="capsule" fromto="0 0 0 0 0 {L.L_FORE}" size="0.022"/>
            <body name="link4" pos="0 0 {L.L_FORE}" gravcomp="1">
              <joint name="wrist_pitch" axis="0 1 0" range="{jr['wrist_pitch'][0]} {jr['wrist_pitch'][1]}"/>
              <geom type="sphere" size="0.024" rgba="0.25 0.26 0.28 1"/>
              <body name="palm" pos="0 0 0" gravcomp="1">
                <joint name="wrist_roll" axis="0 0 1" range="{jr['wrist_roll'][0]} {jr['wrist_roll'][1]}"/>
                <geom type="cylinder" fromto="0 0 0.012 0 0 0.040" size="0.020"
                      rgba="0.25 0.26 0.28 1"/>
                <site name="tool" pos="0 0 {L.L_TOOL}" size="0.004"
                      rgba="1 0.3 0.1 0.6"/>
                <site name="tool_axis" pos="0 0 0.07" size="0.002 0.03"
                      type="cylinder" rgba="1 0.3 0.1 0.25"/>
                <body name="finger_left" pos="0 0 0" gravcomp="1">
                  <joint name="finger_left" type="slide" axis="1 0 0"
                         range="{L.FINGER_RANGE[0]} {L.FINGER_RANGE[1]}"
                         damping="2" armature="0.002"/>
                  <geom type="box" size="0.003 0.008 0.025"
                        pos="{L.FINGER_X} 0 0.062" rgba="0.75 0.76 0.78 1"/>
                  <geom name="pad_left" type="box" size="0.0035 0.010 0.009"
                        pos="{L.FINGER_X - 0.0005} 0 {L.L_TOOL}"
                        rgba="0.15 0.5 0.35 1" friction="2.2 0.05 0.002"
                        solimp="0.98 0.995 0.0005" solref="0.004 1"/>
                </body>
                <body name="finger_right" pos="0 0 0" gravcomp="1">
                  <joint name="finger_right" type="slide" axis="-1 0 0"
                         range="{L.FINGER_RANGE[0]} {L.FINGER_RANGE[1]}"
                         damping="2" armature="0.002"/>
                  <geom type="box" size="0.003 0.008 0.025"
                        pos="{-L.FINGER_X} 0 0.062" rgba="0.75 0.76 0.78 1"/>
                  <geom name="pad_right" type="box" size="0.0035 0.010 0.009"
                        pos="{-(L.FINGER_X - 0.0005)} 0 {L.L_TOOL}"
                        rgba="0.15 0.5 0.35 1" friction="2.2 0.05 0.002"
                        solimp="0.98 0.995 0.0005" solref="0.004 1"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>'''


def build_scene(markers=True, start_position=True):
    """Return the scene XML as a string.

    markers
        Add the coloured type discs on top of each piece.  Turn them off to
        render images that look like an unmarked set (see :mod:`sim.dataset`).
    start_position
        Place the 32 pieces in the opening position.  Either way the bodies
        exist; :class:`sim.sim_env.ChessSim` repositions them on reset.
    """
    bodies, welds = [], []

    back = {"a": "r", "b": "n", "c": "b", "d": "q", "e": "k",
            "f": "b", "g": "n", "h": "r"}
    spawn = []          # (name, color, type, x, y)
    for color in ("w", "b"):
        home_rank = "1" if color == "w" else "8"
        pawn_rank = "2" if color == "w" else "7"
        idx = {t: 0 for t in "pnbrqk"}
        for f in L.FILES:
            for sq, t in ((f + home_rank, back[f]), (f + pawn_rank, "p")):
                name = f"{color}_{t}_{idx[t]}"
                idx[t] += 1
                spawn.append((name, color, t) + L.square_to_xy(sq))
    for color in ("w", "b"):
        for t in STOCK_PIECES:
            spawn.append((f"stock_{color}_{t}", color, t) + L.stock_slot(color, t))

    parked = 0
    for name, color, t, x, y in spawn:
        if not start_position and not name.startswith("stock_"):
            x, y = 0.62, -0.55 + 0.03 * parked   # stacked off to one side
            parked += 1
        bodies.append(_piece_body(name, color, t, (x, y, 0.0005), markers))
        welds.append(f'    <weld name="grasp_{name}" body1="palm" body2="{name}"'
                     f' active="false" torquescale="1" solimp="0.95 0.99 0.001"'
                     f' solref="0.005 1"/>')

    cam_pos, cam_quat = L.camera_pose()
    mesh_assets = "\n".join(
        f'    <mesh name="mesh_{k}" file="meshes/{k}.obj"/>'
        for k in sorted(piece_meshes.PROFILES))
    piece_xml = "\n".join(bodies)
    weld_xml = "\n".join(welds)

    return f'''<mujoco model="kasparobot">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic"
          impratio="10" noslip_iterations="3">
    <flag multiccd="enable"/>
  </option>

  <visual>
    <global offwidth="1280" offheight="1280" azimuth="-90" elevation="-35"/>
    <headlight ambient="0.33 0.33 0.33" diffuse="0.30 0.30 0.30" specular="0.05 0.05 0.05"/>
    <quality shadowsize="4096" offsamples="8"/>
    <map znear="0.01"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient" width="256" height="256"
             rgb1="0.29 0.33 0.40" rgb2="0.07 0.08 0.11"/>
    <texture name="floor_tex" type="2d" builtin="checker" width="300" height="300"
             rgb1="0.20 0.21 0.24" rgb2="0.24 0.25 0.28"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="10 10"
              reflectance="0.02" specular="0"/>
    <!-- Checker and rank/file coordinates are baked into one image by
         sim.board_texture, so the printed squares cannot drift out of step
         with the logical board. -->
    <texture name="board_tex" type="2d" file="board.png"/>
    <material name="board_mat" texture="board_tex" texrepeat="1 1"
              specular="0.05" shininess="0.05" reflectance="0"/>
    <material name="table_mat" rgba="0.52 0.44 0.36 1" specular="0.05"/>
    <!-- Turned piece meshes, generated by sim.piece_meshes.  Visual only. -->
{mesh_assets}
  </asset>

  <default>
    <default class="piece">
      <geom type="cylinder" density="900" friction="1.4 0.03 0.003"
            solimp="0.96 0.995 0.001" solref="0.005 1" condim="4"/>
    </default>
    <default class="arm">
      <joint damping="4" armature="0.03" frictionloss="0.02"/>
      <geom type="capsule" density="1400" rgba="0.82 0.84 0.86 1" condim="3"/>
    </default>
  </default>

  <worldbody>
    <!-- Both lights sit over Black's side, so the arm - which works from the
         opposite side - throws its shadow away from the board rather than
         across it. -->
    <light pos="0.55 0.60 1.55" dir="-0.3 -0.35 -1" diffuse="0.35 0.35 0.35"
           specular="0.04 0.04 0.04" castshadow="true"/>
    <light pos="-0.55 0.50 1.5" dir="0.3 -0.3 -1" diffuse="0.28 0.28 0.28"
           specular="0 0 0" castshadow="false"/>

    <geom name="floor" type="plane" size="4 4 0.1" pos="0 0 -0.75"
          material="floor_mat"/>
    <geom name="table" type="box" material="table_mat"
          size="{L.TABLE_HALF_X} {L.TABLE_HALF_Y} {L.TABLE_THICK / 2}"
          pos="0 0 {-L.PLATE_THICK - L.TABLE_THICK / 2}"/>
    <geom name="board" type="box" material="board_mat"
          size="{L.PLATE_HALF} {L.PLATE_HALF} {L.PLATE_THICK / 2}"
          pos="0 0 {-L.PLATE_THICK / 2}" friction="0.8 0.02 0.002" condim="4"/>

    <!-- The floating camera: a mocap body, so both its position and its aim
         can be changed at run time through data.mocap_pos / mocap_quat.  It
         is parked obliquely over Black's side, looking down at the board
         across it; the arm works from the opposite side and so never stands
         between the lens and the squares. -->
    <body name="cam_rig" mocap="true" pos="{cam_pos[0]:.6f} {cam_pos[1]:.6f} {cam_pos[2]:.6f}"
          quat="{cam_quat[0]:.6f} {cam_quat[1]:.6f} {cam_quat[2]:.6f} {cam_quat[3]:.6f}">
      <geom type="box" size="0.022 0.022 0.012" rgba="0.1 0.1 0.12 1"
            contype="0" conaffinity="0" mass="0.1" group="2"/>
      <geom type="cylinder" fromto="0 0 -0.012 0 0 -0.030" size="0.011"
            rgba="0.05 0.05 0.3 1" contype="0" conaffinity="0" mass="0.01"
            group="2"/>
      <camera name="{L.CAMERA_NAME}" pos="0 0 0" quat="1 0 0 0" fovy="{L.CAM_FOVY:.4f}"/>
    </body>

    <!-- A wide view of the whole cell, for recordings. -->
    <camera name="oblique" pos="0.95 -1.05 0.85"
            xyaxes="0.7251 0.6888 0 -0.3456 0.3638 0.8652"/>
    <!-- Pulled back far enough to take in the floating camera rig as well. -->
    <camera name="wide" pos="1.05 -1.15 0.95"
            xyaxes="0.7234 0.6906 0 -0.1960 0.2053 0.9590"/>

{_arm()}

{piece_xml}
  </worldbody>

  <equality>
{weld_xml}
  </equality>

  <actuator>
    <position name="base_yaw" joint="base_yaw" kp="120" kv="18" forcerange="-120 120"
              ctrlrange="{L.JOINT_RANGE['base_yaw'][0]} {L.JOINT_RANGE['base_yaw'][1]}"/>
    <position name="shoulder" joint="shoulder" kp="300" kv="40" forcerange="-200 200"
              ctrlrange="{L.JOINT_RANGE['shoulder'][0]} {L.JOINT_RANGE['shoulder'][1]}"/>
    <position name="elbow" joint="elbow" kp="200" kv="26" forcerange="-150 150"
              ctrlrange="{L.JOINT_RANGE['elbow'][0]} {L.JOINT_RANGE['elbow'][1]}"/>
    <position name="wrist_pitch" joint="wrist_pitch" kp="80" kv="10" forcerange="-60 60"
              ctrlrange="{L.JOINT_RANGE['wrist_pitch'][0]} {L.JOINT_RANGE['wrist_pitch'][1]}"/>
    <position name="wrist_roll" joint="wrist_roll" kp="30" kv="4" forcerange="-30 30"
              ctrlrange="{L.JOINT_RANGE['wrist_roll'][0]} {L.JOINT_RANGE['wrist_roll'][1]}"/>
    <position name="finger_left" joint="finger_left" kp="90" kv="3" forcerange="-18 18"
              ctrlrange="{L.FINGER_RANGE[0]} {L.FINGER_RANGE[1]}"/>
    <position name="finger_right" joint="finger_right" kp="90" kv="3" forcerange="-18 18"
              ctrlrange="{L.FINGER_RANGE[0]} {L.FINGER_RANGE[1]}"/>
  </actuator>
</mujoco>
'''


def write_scene(path=None, **kwargs):
    board_texture.write(labels=kwargs.get("labels", True))
    piece_meshes.write_all()
    kwargs.pop("labels", None)
    if path is None:
        path = SCENE_PATH if kwargs.get("markers", True) else PLAIN_SCENE_PATH
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_scene(**kwargs))
    return path


if __name__ == "__main__":
    print(write_scene())
