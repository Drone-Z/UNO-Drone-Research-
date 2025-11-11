import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch


# python /Users/pjiang/Documents/Projects/Pothole-Detection/train_yolov3.py --run
# python /Users/pjiang/Documents/Projects/Pothole-Detection/train_yolov3.py --run

def detect_device(prefer_mps: bool) -> str:
    """
    Decide device string for training.
    
    Priority order:
        1. CUDA (if available and functional) - returns "0" for single GPU
        2. MPS (if prefer_mps is True and available) - returns "mps"
        3. CPU (fallback) - returns "cpu"
    
    Returns:
        Device string: "0" for CUDA, "mps" for Apple MPS, or "cpu" for CPU.
        YOLOv3 repo may not support 'mps'; if unsure, fallback to 'cpu'.
    """
    # Check for CUDA first (highest priority)
    # Verify CUDA is not just available but actually functional
    if torch.cuda.is_available():
        try:
            # Test if CUDA actually works by creating a tensor
            test_tensor = torch.tensor([1.0]).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            return "0"
        except (RuntimeError, AssertionError) as e:
            print(f"[WARNING] CUDA is reported as available but not functional: {e}")
            print("[WARNING] Falling back to CPU. Check NVIDIA driver installation.")
    
    # Check for MPS if preferred
    if prefer_mps and torch.backends.mps.is_available():
        return "mps"
    
    # Fallback to CPU
    return "cpu"


def main() -> None:

    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    parser = argparse.ArgumentParser(description="Launcher for Ultralytics YOLOv3 training on pothole dataset.")
    parser.add_argument(
        "--repo",
        type=str,
        default=str(script_dir / "external" / "yolov3"),
        help="Path to local clone of https://github.com/ultralytics/yolov3",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(script_dir / "dataset" / "pothole.yaml"),
        help="Path to dataset YAML.",
    )
    parser.add_argument("--img", type=int, default=416, help="Training image size.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--prefer-mps", action="store_true", help="Prefer Apple MPS if available.")
    parser.add_argument("--run", action="store_true", help="Actually run training instead of printing commands.")
    args = parser.parse_args()

    repo_path = Path(args.repo)
    data_yaml = Path(args.data)

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

    device = detect_device(prefer_mps=args.prefer_mps)

    if not repo_path.exists():
        print("[INFO] YOLOv3 repo not found. Run these commands once:")
        print(f"  mkdir -p {repo_path.parent}")
        print(f"  git clone https://github.com/ultralytics/yolov3 {repo_path}")
        print(f"  cd {repo_path} && pip install -r requirements.txt")
        print("Then rerun this launcher.")
        return

    train_py = repo_path / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(f"train.py not found in repo: {train_py}")

    # Model config and pretrained weights shipped with the repo
    model_cfg = repo_path / "models" / "yolov3.yaml"
    weights = repo_path / "yolov3.pt"
    name = "pothole_yolov3"

    cmd = [
        "python",
        str(train_py),
        "--img", str(args.img),
        "--batch", str(args.batch),
        "--epochs", str(args.epochs),
        "--data", str(data_yaml),
        "--cfg", str(model_cfg),
        "--weights", str(weights),
        "--name", name,
        "--device", device,
    ]

    print("[INFO] Training command:")
    print(" ".join(cmd))
    if device == "0":
        print("[INFO] Using CUDA device 0 for training.")
    elif device == "mps":
        print("[NOTE] If the YOLOv3 repo does not support 'mps', change --device to 'cpu'.")

    if args.run:
        env = os.environ.copy()
        try:
            result = subprocess.run(
                cmd,
                check=True,
                cwd=str(repo_path),
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print("[ERROR] Training failed with exit code:", e.returncode)
            print("\n[ERROR] STDOUT:")
            print(e.stdout if e.stdout else "(empty)")
            print("\n[ERROR] STDERR:")
            print(e.stderr if e.stderr else "(empty)")
            raise
    else:
        print("[INFO] Pass --run to execute. Otherwise, run the printed command manually.")


if __name__ == "__main__":
    main()


