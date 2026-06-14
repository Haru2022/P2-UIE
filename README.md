# P2-UIE

This folder contains the official inference script and local pipeline code for P2-UIE (accepted by **Information Fusion**).

## Create the environment

The script uses PyTorch, Diffusers, Transformers, TorchVision, OpenCV, Pillow, NumPy, tqdm. The versions below are pinned from the `watersd` conda environment.

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

## Self-collected Dataset

We collected a new dataset named "[P2UIE-1473](https://huggingface.co/HaruCloud9/P2UIE/tree/main/Dataset)" including the selected images from [UIEB](https://li-chongyi.github.io/proj_benchmark.html), [UVEB](https://github.com/yzbouc/UVEB), and [DRUVA](https://github.com/nishavarghese15/DRUVA). Many thanks to their oontributions to this research field.

The new collected dataset P2UIE-1473 contains the raw images, the haze images, the coarse enhanced images, and the auto-generated masks for partial-mask-learning proposed in this article. 