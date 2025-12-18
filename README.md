# Kasparobot

A computer vision system that analyzes physical chess boards, converts positions to FEN notation, and provides move recommendations using Stockfish.

## Features

- **Board Detection**: Uses YOLO for real-time chess piece recognition
- **Position Analysis**: Converts detected board state to FEN (Forsyth-Edwards Notation)
- **Move Recommendations**: Integrates with Stockfish engine for strategic analysis

## Setup

Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate <env-name>
pip install -r requirements.txt
```
Install most recent Stockfish distribution:
```bash
wget https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1/stockfish-ubuntu-x86-64-avx2.tar
tar -xvf stockfish-ubuntu-x86-64-avx2.tar
sudo mv stockfish/stockfish-ubuntu-x86-64-avx2 /usr/local/bin/stockfish
sudo chmod +x /usr/local/bin/stockfish
```

## Usage

### Training the Model

### Interactive Chess Game

Start an interactive command-line chess game:
```bash
python main.py
```

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

### Getting Move Recommendations

Analyze a chess board image and get Stockfish recommendations:
```bash
python get_stockfish_response.py
```

**Note:** The `get_stockfish_response` function is currently a work in progress.

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

## License

Please note that YOLO12 is licensed under AGPL-3.0. Ensure compliance with this license when using or distributing this project.