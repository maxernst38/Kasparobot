# Kasparobot

A computer vision system that analyzes chess boards, converts positions to FEN notation, and
provides move recommendations using Stockfish — plus a MuJoCo simulation of the robot that
plays the moves: a 5-DOF arm working under a floating camera aimed obliquely at the board.

## Features

- **Board Detection**: Uses YOLO for real-time chess piece recognition
- **Position Analysis**: Converts detected board state to FEN (Forsyth-Edwards Notation)
- **Move Recommendations**: Integrates with Stockfish engine for strategic analysis
- **Physical simulation**: A MuJoCo cell with a 5-DOF arm that picks and places every piece,
  handling captures, castling, en passant and promotion
- **Floating board camera**: An aimable oblique camera renders the board and reads the
  position back out, closing the camera → FEN → engine → arm → camera loop
- **Synthetic training data**: Renders labelled camera frames of a turned Staunton set to
  retrain the detector

## Setup

Create and activate the environment:
```bash
conda env create -f environment.yml      # installs requirements.txt too
conda activate chess
```

Or without conda, on Python 3.10+:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` lists the seven direct dependencies; pip resolves torch and the rest of
the YOLO stack behind `ultralytics`.

Install the most recent Stockfish distribution:
```bash
wget https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1/stockfish-ubuntu-x86-64-avx2.tar
tar -xvf stockfish-ubuntu-x86-64-avx2.tar
sudo mv stockfish/stockfish-ubuntu-x86-64-avx2 /usr/local/bin/stockfish
sudo chmod +x /usr/local/bin/stockfish
```
The engine is located via `$STOCKFISH_PATH`, then `PATH`, then the usual install
locations, so an install anywhere works:
```bash
export STOCKFISH_PATH=/path/to/stockfish
```

Detector weights are not in the repository — they are a 40 MB binary. Fetch a published
one, or train your own from simulated frames (see [model/README.md](model/README.md)):
```bash
python scripts/fetch_model.py <release asset URL>
```
The simulator itself needs no weights: `python main.py --mode random --viewer` runs
with nothing but the conda environment.

## Usage

`main.py` runs the simulator — see **Simulation** below for the full set of options. The
quickest look at it, needing nothing but the environment:

```bash
MUJOCO_GL=glfw python main.py --mode random --viewer
```

### Training the Model

Train or refine a YOLO model on your chess piece dataset:
```bash
python train.py --data_yaml path/to/data.yaml \
                --model yolo12m.pt \
                --epochs 100 \
                --batch_size 4 \
                --img_size 480
```

**Arguments:**
- `--data_yaml`: Path to the dataset's YAML file (required)
- `--model`: YOLO model variant (default: `yolo12m.pt`)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 4)
- `--img_size`: Image size for training (default: 480)

The simulator can generate the dataset for you — see **Building a dataset** below. Pass
`--img_size 640` to match the render resolution; the 480 default throws away detail on pieces
that are only 30–45 px tall.

### Getting Move Recommendations

Analyze a chess board image and get Stockfish recommendations:
```bash
python get_stockfish_response.py [image.jpg]
```

---

## Simulation

A MuJoCo scene containing a table, a 456 mm tournament board with printed rank and file
coordinates, 32 pieces plus promotion spares, a 5-DOF arm with a parallel-jaw gripper, and a
floating camera aimed obliquely at the board. The arm physically carries out every move of
a real game.

The pieces are turned meshes, not primitives — see **The pieces** below.
Squares are 57 mm and the board is framed by a 24 mm border carrying the a-h and 1-8 labels
on all four sides, each edge turned to read correctly from outside that edge — so whichever
side the camera is on, the coordinates nearest it are the right way up. The checker and the labels are one generated image
(`sim/board_texture.py`) rather than hand-placed geometry, so the printed squares cannot
drift out of step with the logical board — change `SQUARE` in `sim/board_layout.py` and the
texture is rebuilt to match.

```bash
python main.py                      # Stockfish vs Stockfish, arm plays every move
python main.py --mode random        # no engine needed
python main.py --mode human         # you play White from the keyboard
python main.py --viewer             # open the interactive MuJoCo window
python main.py --camera-view        # watch what the board camera sees
python main.py --video game.mp4     # record it
python -m sim.selftest               # verify the whole stack

# play from what the camera sees, rather than from the game state:
python main.py --perceive --detector yolo --weights runs/.../weights/best.pt
python main.py --perceive --markers   # or the built-in marker detector
```

Rendering needs a GL backend. Under WSL or a desktop session use `MUJOCO_GL=glfw`; on a
headless machine with an NVIDIA driver use `MUJOCO_GL=egl`.

```bash
MUJOCO_GL=glfw python main.py --viewer
```

### The arm

Five joints: `base_yaw`, `shoulder`, `elbow`, `wrist_pitch`, `wrist_roll`, plus two gripper
fingers.

Five degrees of freedom cannot reach an arbitrary 6-DOF pose, but chess does not need one.
The yaw picks the vertical plane containing the target square; the three co-planar pitch
joints place the tool anywhere in that plane *and* set its approach direction; the roll spins
the jaws about that approach axis. Pinning the approach to straight down leaves a closed-form
inverse kinematics — a planar 2-R solve — so there is no iterative solver to tune or to fail
mid-game. `sim/arm_kinematics.py` holds the derivation; `sim.selftest` checks it against
MuJoCo's own forward kinematics at 936 poses covering every square, every discard slot and
every promotion spare.

Pieces are grasped by the stem, which is the same shape on every piece, so one grasp routine
covers pawns and kings alike. Before each pick the controller looks at which neighbouring
squares are occupied and rotates the jaws to the clearest heading. The gripper opens to
36 mm inside a 50 mm square, so an open jaw clears the bases of pieces on adjacent squares.

A friction grasp is backed up by a weld constraint that freezes the pose the jaws actually
achieved. A game is hundreds of lifts, and one silent slip would corrupt the board; the weld
makes a bad grasp show up as a misplaced piece rather than a piece that quietly stays put.

### The pieces

Every piece but the knight is a surface of revolution, because a real set is turned on a
lathe. `sim/piece_meshes.py` does the same thing: each type is a handful of control points,
smoothed into a profile and revolved, plus the features that are not revolutions — the
rook's crenellations, the queen's crown, the king's cross, and the knight's head, which is a
side-on silhouette extruded across and tapered at the cheeks.

The meshes are **visual only**. Collision stays on the primitive geoms, for a reason worth
knowing if you edit this: MuJoCo collides a mesh by its *convex hull*, which would fill in
the waist of the piece and leave the parallel jaws nothing to close on. The primitives sit in
geom group 3, which MuJoCo does not render, so they do the physics without appearing in any
image. The mesh tops are pinned to `PIECE_HEIGHT`, which is what perception back-projects at.

Resolution is set by `SEGMENTS` and `SMOOTHING` in that module. The default 40 segments
costs about the same per frame as the old primitives did — on this machine 60.7 ms against
59.9 ms at 640x640, because the renderer rather than the triangle count is the limit. Raising
it to 48 segments cost 77 ms, so turn it up only if you are rendering silhouettes much larger
than a board's worth.

### Pieces, plain or marked

By default the set renders plain, which is what you want for training images.

The optional `--markers` flag puts a coloured disc on each piece's crown. Those discs are
what `MarkerPerception` reads piece *type* from, so they are what makes the camera → FEN loop
work before a detector exists:

```bash
python main.py --markers --perceive      # --perceive needs them, and says so if missing
python -m sim.camera_view --markers
```

There is no cheap substitute. Without the discs, type has to come from shape: the knight is
54 mm tall and 13.0 mm across the crown, the queen 56 mm and 13.5 mm — 2 mm and 0.5 mm apart.
Size alone cannot separate them, which is exactly the job a trained detector does.

### The camera

The camera rides a mocap body, so where it is and where it points can both be changed at run
time. Its aim is spherical — an azimuth around the board, an elevation above the board plane,
and a distance — and the default is oblique:

```python
sim.move_camera(azimuth=90, elevation=52, distance=0.95)
sim.move_camera(elevation=90)        # straight down, for comparison
sim.move_camera(azimuth=0)           # from the h-file side
```

The same three knobs are on the command line, and the defaults live in
`sim/board_layout.py`:

```bash
python main.py --camera-azimuth 0 --camera-view    # side view, same tilt
python -m sim.camera_view --azimuth 180
```

Azimuth is measured around the board from `+x`: **0 and 180 are the two side views, 90 is
over Black (the default), 270 is over White.** Measured over sampled positions, 0 and 180
read 310/310 squares correctly and 90 reads 326/326 — but **270 drops to 248/310**, because
that is the arm's own side and its column stands between the lens and the near rank. Keep the
camera off 270 unless you also move the arm. The field of view is **computed, not tuned**:
`camera_fovy()` projects the corners of the board and its border and returns the angle that
just contains them, so re-aiming the camera cannot silently crop a corner off the board.

The default azimuth puts the camera over Black's side, opposite the arm. That is the one
choice that really matters: the arm works from the far side, so a parked arm is always
*beyond* the squares rather than in front of them. It can appear in the picture — as it would
in any real cell — but it never covers a square, and the lights are placed on the camera's
side so the arm throws its shadow away from the board rather than across it.

An oblique view costs two things a perpendicular one does not:

**Parallax is directional and large.** A piece is 48–68 mm tall, and its crown appears
displaced away from the camera. Back-projecting a crown onto `z = 0` instead of onto the
piece's own height would misplace a king by **79 mm — 1.38 squares** at the worst square on
this board. `BoardCamera.unproject` takes the height to intersect at, and the detector knows
each piece's height because the marker colour has already told it the type. `sim.selftest`
prints that slip figure on every run so a regression in the height handling is obvious.

**Pieces can hide each other.** At a shallow enough angle a tall piece covers the one behind
it. The geometry is simple: with one square of separation, crowns stay visible as long as the
elevation is above about 20°, and the default 52° has room to spare.

What an oblique camera *gains* is fidelity. Looking straight down, the board and the image
are related by a plain scale factor, which is not a problem any real camera has. Obliquely,
they are related by a genuine homography — the same object `core/detection/detect_board.py`
fits from a photograph by finding grid lines. Here it is known exactly:

```python
H = BoardCamera.from_sim(sim).homography(z=0.0)   # board metres -> pixels
```

`BoardCamera.from_sim` reads the live pose out of the model, so nothing assumes the camera is
in any particular place — move the rig and projection, perception and dataset labels all
follow.

### Watching the camera

```bash
python main.py --camera-view                  # live window while the arm plays
python main.py --camera-view --camera-plain   # raw image, no overlay
python main.py --camera-view --camera-homography   # add the flattened board
python main.py --camera-save frames/          # one annotated frame per move
python -m sim.camera_view --fen "..." --out shot.png   # a single still
python -m sim.camera_view --homography --out shot.png
```

`--camera-homography` puts a second panel beside the camera view: the image warped flat
through the board homography, which is the space the square mapping actually works in. The
squares come out square whatever angle the camera is at, and every detection gets a red dot
and a label where the mapping decided it stands. Pieces smear outwards because solid objects
are being flattened onto the board plane — which is exactly why the mapping uses the foot of
each box rather than its centre.

It is worth watching: the dots sit about a quarter of a square towards the camera, because
the foot of a box is the near rim of the piece's base rather than its centre. That is well
inside the square, but the panel shows you how much margin is left.

```bash
python -m sim.camera_view --elevation 90      # compare against straight down
python -m sim.camera_view --azimuth 0         # or from the side
```

While the arm is moving you get the raw frame, robot and all — that is honestly what the
camera sees. Once the move finishes and the board is static, the frame is redrawn with the
square grid and the detector's reading on top. The grid comes from the camera model rather
than from the picture, so any drift between the two is immediately visible.

The live window needs an OpenCV build with GUI support. If you have
`opencv-python-headless` installed, frames are written to `camera_view/` instead and a
message says so; `pip install opencv-python` gets you the window.

### Reading the board

With `--markers` on, `sim/perception.py` identifies pieces by the coloured disc on each crown.
The discs sit at HSV saturation ≈ 220 while the board tops out at 156, so a single saturation
threshold separates them cleanly, and the hue gives the piece type. The side comes from the
piece's own body: near-grey pixels sampled along the segment from the crown down to where the
piece meets the board — not a ring around the crown, which only works from directly above.
`sim.selftest` reads back sampled positions square by square.

This is a stand-in, not the end goal. `model/Kasparobot.pt` was trained on photographs of a
real set, so the markers give the system a working image → FEN path while that gap is closed
from the other side:

```bash
python -m sim.dataset --out dataset_sim --train 400 --val 80 --preview
```

That takes about two minutes for 480 frames (0.19 s each) and writes:

```
dataset_sim/
  labels.csv                  one row per visible piece, with its box
  frames.csv                  one row per image: size, camera pose, piece count
  train/images/*.png          val/images/*.png
  train/labels/*.txt          the same boxes in YOLO format
  data.yaml                   so train.py can consume it directly
```

`labels.csv` has absolute pixel boxes and the class numbering `core/ChessBoard.py` already
uses:

| split | image | width | height | piece | colour | kind | class_id | square | x_min | y_min | x_max | y_max | pixels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | train/images/train_00000.png | 640 | 640 | N | w | n | 9 | a5 | 360.0 | 194.0 | 389.0 | 237.0 | 855 |

`--no-yolo` writes only the CSVs. Otherwise both come out, so you can train immediately:

```bash
python train.py --data_yaml dataset_sim/data.yaml \
                --model yolo12m.pt --img_size 640 --batch_size 8 --epochs 100
```

Pass `--img_size 640` to match the render resolution — `train.py` defaults to 480, which
throws away detail on pieces that are only 30-45 px tall. Turn `--batch_size` down if the GPU
runs out of memory. Results land in `runs/` (already gitignored); the weights you want are
`weights/best.pt`.

Point the sim at a model you have trained by replacing `model/Kasparobot.pt`, or load it
directly:

```python
from ultralytics import YOLO
model = YOLO("runs/.../weights/best.pt")
```

**Boxes come from a segmentation render, not from projecting each piece.** From a side view
pieces stand in front of one another, and a projected box would happily label a piece that is
entirely hidden behind another. MuJoCo can render per-pixel geometry IDs, so the boxes bound
the pixels that actually belong to each piece — about 58% of the area a projected box would
claim — and anything with fewer than `--min-pixels` visible is dropped. `sim.selftest`
cross-checks the two independently: whatever segmentation finds must lie inside the
projection.

**The camera stays an oblique side view** and moves a little between frames: azimuth jitters
around 0 and 180 (both sides, unless you pass `--sides one`), elevation around 42°, distance
around 0.95 m, plus the lights. Azimuth 270 is deliberately out of reach — that is the arm's
own side, where perception measurably degrades.

```bash
python -m sim.dataset --azimuth 0 --azimuth-jitter 15 \
                      --elevation 38 --elevation-jitter 12 \
                      --distance 0.95 --distance-jitter 0.08
```

The field of view is refitted for every pose, so the board cannot drift off the edge of the
frame and truncate a label.

**Positions vary in how full the board is.** By default 65% of frames are positions from
random games — realistic counts and clustering — and 35% are scattered pieces on an otherwise
empty board (`--sparse-fraction`). A detector trained only on game positions sees crowded
back ranks and little else; the sparse frames fill in the long tail of nearly-empty boards
and odd piece mixes. A 20-frame sample came out with 2 to 32 pieces per board and all twelve
classes represented.

Pieces not in the position are hidden as well as parked — from a side view their parking
spots sit between the lens and the board, so moving them aside is not enough.

The robot itself **is** in shot by default, because it would be in shot through the real
camera too. `--hide-arm` leaves it out if you would rather have board-only frames.

### Driving the robot with a trained detector

Once you have weights, the robot can play from them through the project's own
`core.ChessBoard` pipeline:

```bash
python main.py --perceive --detector yolo \
       --weights runs/.../weights/best.pt --camera-azimuth 0
```

`sim/perception.py`'s `YoloPerception` wraps `ChessBoard`, so the weights, the class
numbering and the FEN assembly are all the ones the rest of the project already uses. Each
move the camera image goes to the detector, the boxes become a FEN, Stockfish picks from that
FEN, and the arm plays it.

Three changes were needed in the existing code to make this work, all backward compatible
with the photograph path:

- `ChessBoard(weights=...)` chooses the detector; it still defaults to `model/Kasparobot.pt`.
- `ChessBoard.detect_pieces` accepts an RGB array as well as a path, and loads the model once
  instead of rebuilding it per call — that rebuild cost about a second a frame, which is the
  whole budget for reading the board between moves.
- `map_detections_to_squares` had `row = int(warped_y // 100) + 1`. That offset shifts the
  whole board a rank and pushes the last one out of bounds: fed a known-good homography, the
  opening position read as `8/rnbqkbnr/pppppppp/...` and matched **0 of 40** squares. Without
  it, all 32 pieces land correctly. If that `+1` was silently compensating for
  `sort_grid_geometrically` missing the top row of intersections on real photographs, the fix
  belongs in the grid detection instead — worth checking against your own images.

**One stage is bypassed: `core.detection.detect_board`.** It estimates the board homography
by finding grid lines and binning them into near-horizontal and near-vertical, which only
holds for a roughly top-down, axis-aligned view. On an oblique frame the lines slant and
converge and it finds a single intersection, which is not enough for a homography. In the
simulator the camera pose is known, so `sim.perception.grid_homography()` computes the matrix
instead of fitting it — which is also what a real cell would do, calibrating once rather than
re-detecting the board every frame. Making `detect_board` itself work obliquely is a separate
job: find the board's outer quad from the green mask, order its corners, and map that to the
8x8 grid.

### Evaluating a detector

Ultralytics has this built in, and it is the right first answer:

```bash
yolo val model=runs/.../weights/best.pt data=dataset_sim/data.yaml imgsz=640
```

It prints per-class precision, recall, mAP50 and mAP50-95, and writes a confusion matrix and
PR/F1 curves into `runs/detect/val/` — which are the useful part once the numbers flatten
out. On synthetic data they flatten fast: a `yolo12m` run over 4000 frames was already at
precision/recall 1.000 and mAP50-95 0.990 by epoch 16.

Two caveats worth taking seriously.

**A val split from the same generator is not held out in the way that matters.** Same seed
stream, same view distribution, same renderer. It tells you training converged, not that the
model works. For an honest number, generate a fresh set with a different seed:

```bash
python -m sim.dataset --out dataset_test --train 0 --val 300 --seed 99999
yolo val model=runs/.../weights/best.pt data=dataset_test/data.yaml imgsz=640
```

**mAP does not tell you whether the robot reads the board.** A box can be correct by IoU and
still land on the wrong square, and one wrong square out of 32 is a wrong move. `sim/evaluate.py`
measures that end to end: it renders fresh positions, turns the detections into a FEN through
the same mapping the robot uses, and compares against what is physically on the board.

```bash
python -m sim.evaluate --weights runs/.../weights/best.pt --positions 100
```

```
square accuracy      : 100.000%  (1038/1038)
whole position right : 40/40  (100.0%)   <- what the robot needs
localisation error   : median 4.6 mm, 90th pct 6.7 mm, max 10.6 mm   (square is 57 mm)
```

The localisation figure is the one to watch as the margin, not the accuracy: it is how far
the mapped point sits from the centre of its square, against a 28.5 mm half-square budget.

### What a single image cannot tell you

`--perceive` plays from what the camera sees, but a photograph only shows where the pieces
are. Whose turn it is, who may still castle, and whether a pawn just ran past an en-passant
square are not visible in any image; those four FEN fields come from the game record. That
split is inherent to a camera-driven robot, not an artifact of the simulation.

### Layout

| File | What it holds |
| --- | --- |
| `sim/board_layout.py` | Every dimension, in one place: squares, pieces, arm links, trays, camera |
| `sim/board_texture.py` | Draws the board image: checker plus rank/file coordinates |
| `sim/piece_meshes.py` | Turns the piece meshes on a virtual lathe |
| `sim/scene_builder.py` | Generates the MuJoCo XML into `sim/assets/` |
| `sim/arm_kinematics.py` | Closed-form forward and inverse kinematics |
| `sim/arm_controller.py` | Trajectories, gripper, grasp assist, pick and place |
| `sim/move_executor.py` | Chess rules → physical transfers (no MuJoCo needed) |
| `sim/sim_env.py` | `ChessSim`: the scene, the game, the camera, one API |
| `sim/perception.py` | Camera geometry, image → FEN (marker and YOLO backends) |
| `sim/camera_view.py` | Live window / stills of the board camera |
| `sim/dataset.py` | Labelled synthetic frames: CSV + YOLO |
| `sim/evaluate.py` | Scores a detector by whether it reads the board right |
| `sim/selftest.py` | Measured checks over the whole stack |
| `main.py` | Command-line entry point: runs a game in the simulator |

World frame: `+x` runs a→h, `+y` runs 1→8, `z = 0` is the playing surface, the board is
centred on the origin and the arm stands on White's side. Squares are 57 mm.

`sim/assets/` is generated — the scene XML and the board texture are rebuilt from
`sim/board_layout.py` on every run, so it is not checked in.

### Using it from Python

```python
from sim.sim_env import ChessSim
from sim.perception import MarkerPerception, BoardCamera

sim = ChessSim(viewer=True)
sim.apply_move("e4")                 # SAN or UCI
sim.apply_move("c7c5")

reader = MarkerPerception(BoardCamera.from_sim(sim))
print(reader.read_fen(sim.capture()))

ok, problems = sim.consistent()      # does the board still match the game?
```

### Known limits

- The scene carries one spare queen, rook, bishop and knight per colour, so a position needing
  a second promotion to the same piece raises `OutOfPieces`.
- 32 discard slots fit beside the board; a game removing more than that raises `OutOfPieces`.
- The pieces are turned to Staunton proportions, but the renders are still clean synthetic
  images: no wood grain, no scratches, no depth of field. Expect a detector trained only on
  them to need real photographs mixed in before it transfers to a physical board.

## Citations

This project builds upon the following open-source tools:

**YOLO12 Object Detection:**
```
Tian, Y., Ye, Q., & Doermann, D. (2025). YOLO12: Attention-Centric Real-Time Object Detectors. 
arXiv preprint arXiv:2502.12524.
https://github.com/sunsmarterjie/yolov12 (AGPL-3.0 License)
```

**Stockfish Chess Engine:**
```
The Stockfish developers (2025). Stockfish 17.1. 
https://stockfishchess.org/
https://github.com/official-stockfish/Stockfish
```

**MuJoCo:**
```
Todorov, E., Erez, T., & Tassa, Y. (2012). MuJoCo: A physics engine for model-based control.
IEEE/RSJ International Conference on Intelligent Robots and Systems, 5026-5033.
https://github.com/google-deepmind/mujoco (Apache-2.0 License)
```

## License

This project is licensed under the **GNU Affero General Public License v3.0** —
see [LICENSE](LICENSE). AGPL applies because the detector depends on
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which is itself
AGPL-3.0; running a modified version over a network obliges you to offer the
source.

Stockfish is GPL-3.0 and is invoked as a separate binary, not linked.
