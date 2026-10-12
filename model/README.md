# Detector weights

`Kasparobot.pt` is not in the repository — it is a 40 MB binary and belongs on a
release instead. Get it either way:

**Download a published one**

    python scripts/fetch_model.py <release asset URL>

or set `KASPAROBOT_MODEL_URL` and run the script with no argument.

**Train your own** on frames from the simulator:

    python -m sim.dataset --out dataset_sim --train 400 --val 80
    python train.py --data_yaml dataset_sim/data.yaml --model yolo12m.pt --img_size 640

Then copy `runs/.../weights/best.pt` here, or point `ChessBoard(weights=...)` and
`main.py --weights` straight at it.
