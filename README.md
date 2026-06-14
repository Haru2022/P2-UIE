# P2-UIE

This folder contains the official inference script and local pipeline code for P2-UIE (accepted by **Information Fusion**).

## Create the environment

The script uses PyTorch, Diffusers, Transformers, TorchVision, OpenCV, Pillow, NumPy, tqdm, and xFormers. The versions below are pinned from the `watersd` conda environment.

```bash
conda env create -f environment.yml
conda activate P2UIE

pip install -r requirements.txt
```

If you prefer a lighter setup, you can also create the environment manually with `conda create -n P2UIE python=3.10.16` and then install the pinned packages from `requirements.txt`.

## test

```bash
python test.py
```

you may modify the way of loading images for your own datasets.
