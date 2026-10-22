# SAMurai

Local napari tool with YOLO detection, SAM segmentation, manual review, and measurement export. Version **0.10.6**.

![image](https://github.com/Englundlab/SAMurai/blob/main/GUI%20display.png)

## Install on Windows

Download and extract this project to a writable location. Install **64-bit Python 3.12** with the Python launcher, or use your existing installation if it includes the Python launcher. Leave Python where it is installed; the commands below create an environment inside the application folder.

Open PowerShell in the extracted project folder and run:

```powershell
cd repository
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

The `py -3.12` command uses Python 3.12 already installed on your computer to create the environment in `repository/.venv/`. The following `pip install` commands install packages into that environment, keeping them separate from your main Python installation.

Installation requires internet access. Start with CPU. For NVIDIA CUDA 12.8, replace `/cpu` with `/cu128` in the PyTorch command when creating the environment.

Run the remaining commands from `repository/`.

## Models and startup

The default models must be present at:

- `model/yolo/Model_2.pt`
- `model/SAM/SAM.pth`

`model/yolo/Model_2.pt` is an included example of a trained YOLO model. See the [training folder](training/) for an example of how to train your own.

Download the [SAM ViT-H checkpoint](https://huggingface.co/HCMUE-Research/SAM-vit-h/blob/main/sam_vit_h_4b8939.pth), rename it to `SAM.pth`, and place it in `repository/model/SAM/`. Create the `SAM` folder if needed.

```text
SAMurai-main/
├── README.md
├── training/
└── repository/
    ├── run.py
    ├── requirements.txt
    ├── app/
    │   └── workflow_config.json
    ├── model/
    │   ├── yolo/
    │   │   └── Model_2.pt     ← included
    │   └── SAM/
    │       └── SAM.pth        ← download and rename
    └── .venv/                 ← created during installation
```

Start the application. For GPU processing, replace `--device cpu` with `--device auto`:

```powershell
.\.venv\Scripts\python.exe run.py --device cpu
```


## Test the workflow

Use a small representative 2D grayscale/RGB image, not a stack. Create a project outside this folder → detect → review boxes → confirm boxes and run SAM → edit/review masks → check pixel size and units → export. Reopen the saved project and confirm the edits and exported measurements. Record any errors and the computer's Python version and CPU/GPU mode.

Inside `repository/`:

- `app/`: application and default configuration.
- `model/`: weights and [verification manifest](repository/model/manifest.json).
- `tests/` and `tools/`: automated checks and utilities.

## Software checks

Check model file sizes and hashes:

```powershell
.\.venv\Scripts\python.exe tools/verify_models.py
```

Run the automated tests:

```powershell
.\.venv\Scripts\python.exe tools/run_tests.py
```

On 2026-10-20, 31 tests, offscreen CPU startup, command-line help, model checksums, and configuration paths passed in the existing Windows Python 3.12 environment.
