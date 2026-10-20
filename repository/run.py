"""Launch Spheroid in the current Python environment; never installs packages."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
FLAGS = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0

def gpu_works(python):
    # Exercise both CUDA kernels and torchvision's compiled GPU operation.
    code = ('import torch; from torchvision.ops import nms; '
            'assert torch.cuda.is_available(); '
            'x=torch.ones((16,16),device="cuda"); (x@x).sum().item(); '
            'nms(torch.tensor([[0.,0.,2.,2.]],device="cuda"),'
            'torch.tensor([.9],device="cuda"),.5); torch.cuda.synchronize()')
    try:
        subprocess.run([str(python), '-I', '-c', code], check=True,
                       timeout=90, creationflags=FLAGS, capture_output=True)
        return True
    except (subprocess.SubprocessError, OSError):
        return False

def main():
    parser = argparse.ArgumentParser(description="Spheroid segmentation")
    parser.add_argument('--config', default=str(ROOT / 'app' / 'workflow_config.json'))
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'))
    options = parser.parse_args()
    from app.core import load_config
    requested = options.device or load_config(options.config)['inference']['device']
    device = ('cuda' if gpu_works(sys.executable) else 'cpu') if requested == 'auto' else requested
    from app.app import main as launch
    sys.argv = [sys.argv[0], '--config', options.config, '--device', device]
    print('Processing device: ' + device, flush=True)
    launch()

if __name__ == '__main__':
    main()
