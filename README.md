# SAMurai

Local napari tool with YOLO detection, SAM segmentation, manual review, and measurement export. Version **0.10.6**.

![image](https://github.com/Englundlab/SAMurai/blob/main/GUI%20display.png)


## Install on Windows

Install **64-bit Python 3.12** with the Python launcher. Copy this folder to a writable location, open PowerShell here, and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Installation requires internet access. Start with CPU. For NVIDIA CUDA 12.8, replace `/cpu` with `/cu128` in the PyTorch command when creating the environment. Requirements are not a fully locked environment.

## Models and startup

The default models must be present at:

- `model/yolo/Model_2.pt`
- `model/SAM/SAM.pth` 

`model/yolo/Model_2.pt` is example of a trained YOLO model. See the training folder for example on how to train.

`model/SAM/SAM.pth` download the model from https://huggingface.co/HCMUE-Research/SAM-vit-h/blob/main/sam_vit_h_4b8939.pth

```powershell
.\.venv\Scripts\python.exe tools/verify_models.py
.\.venv\Scripts\python.exe tools/run_tests.py
.\.venv\Scripts\python.exe run.py --device cpu
```

Paths in `app/workflow_config.json` are relative to this folder; no machine-specific settings file is needed. For external models, use absolute paths in a copied `*.local.json` configuration and pass `--config path/to/settings.local.json`.

## Test the workflow

Use a small representative 2D grayscale/RGB image, not a stack. Create a project outside this folder → detect → review boxes → approve SAM → edit/review masks → check pixel size and units → export. Reopen the saved project and confirm the edits and exported measurements. Record any errors and the computer's Python version and CPU/GPU mode.

- `app/`: application and default configuration.
- `model/`: weights and [verification manifest](https://github.com/Englundlab/SAMurai/blob/main/repository/model/manifest.json).
- `tests/` and `tools/`: automated checks and utilities.

## Validation

On 2026-10-20, 31 tests, offscreen CPU startup, command-line help, model checksums, and configuration paths passed in the existing Windows Python 3.12 environment. Fresh installation, real-image inference, interactive GUI use, actual GPU execution, other platforms, and scientific accuracy remain unverified.
