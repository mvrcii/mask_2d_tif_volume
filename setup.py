#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm

console = Console()


def check_python_package(package_name):
    """Check if a Python package is installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def install_python_package(package_name):
    """Install a Python package using pip"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", package_name],
                       check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError:
        return False


def check_command_available(command):
    """Check if a command is available in PATH"""
    try:
        subprocess.run([command, "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def setup_directories():
    """Create necessary directories"""
    dirs = ["./models"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    return True


def check_and_install_packages():
    """Install packages from requirements.txt"""
    with console.status("[cyan]Installing packages from requirements.txt..."):
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            return True, "✓ Installed from requirements.txt"
        else:
            return False, f"✗ Failed: {result.stderr.strip()}"


def check_and_download_model():
    """Check and download model if needed"""
    model_dir = Path("./models/nnUNetTrainerV2__nnUNetPlans__2d")

    key_files = [
        model_dir / "fold_0" / "checkpoint_best.pth",
        model_dir / "dataset.json",
        model_dir / "plans.json"
    ]

    with console.status("Checking model..."):
        if any(f.exists() for f in key_files):
            return True, "✓"

    console.print("[yellow]○ Missing[/yellow]")

    if not check_command_available("wget"):
        return False, "✗ Need wget"

    model_dir.mkdir(parents=True, exist_ok=True)
    model_url = "https://dl.ash2txt.org/community-uploads/bruniss/nnunet_models/nnUNet_results/Dataset082_scrollmask2/nnUNetTrainerWorkshop__nnUNetPlans__2d/"

    cmd = [
        "wget", "-r", "-np", "-nH", "--cut-dirs=5",
        "--reject", "index.html?*", "-P", str(model_dir.parent), model_url
    ]

    with console.status("Downloading..."):
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True, "✓"
        except subprocess.CalledProcessError:
            return False, "✗ Download failed"


def check_gpu():
    """Check GPU availability"""
    with console.status("Checking..."):
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                return True, f"✓ {gpu_name} ({gpu_memory:.1f}GB)"
            else:
                return True, "○ CPU only"
        except ImportError:
            return True, "○ Unknown"


def verify_installation():
    """Quick verification of installation"""
    critical_packages = ["nnunetv2", "torch", "skimage"]
    model_dir = Path("./models/nnUNetTrainerV2__nnUNetPlans__2d")

    with console.status("Verifying..."):
        # Check packages
        for package in critical_packages:
            if not check_python_package(package):
                return False, f"✗ {package} missing"

        # Check model
        key_files = [
            model_dir / "dataset.json",
            model_dir / "plans.json"
        ]
        checkpoint_files = [
            model_dir / "fold_0" / "checkpoint_best.pth",
            model_dir / "fold_0" / "checkpoint_final.pth",
            model_dir / "checkpoint_best.pth",
            model_dir / "checkpoint_final.pth"
        ]

        if not (all(f.exists() for f in key_files) and any(f.exists() for f in checkpoint_files)):
            return False, "✗ Model incomplete"

    return True, "✓"


def main():
    console.print("[bold blue]Setup Scroll Mask Processor[/bold blue]")

    steps = [
        ("Directories", setup_directories),
        ("Packages", check_and_install_packages),
        ("Model", check_and_download_model),
        ("GPU", check_gpu),
        ("Verify", verify_installation)
    ]

    for i, (name, func) in enumerate(steps, 1):
        result = func()
        if isinstance(result, tuple):
            success, message = result
            if success:
                console.print(f"[dim]{i}.[/dim] {name} [green]{message}[/green]")
            else:
                console.print(f"[dim]{i}.[/dim] {name} [red]{message}[/red]")
                return 1
        else:
            # Handle old-style boolean return
            if result:
                console.print(f"[dim]{i}.[/dim] {name} [green]✓[/green]")
            else:
                console.print(f"[dim]{i}.[/dim] {name} [red]✗[/red]")
                return 1

    console.print(f"\n[green]✓ Ready![/green] Use: [cyan]python process_scroll_masks.py[/cyan]")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)
