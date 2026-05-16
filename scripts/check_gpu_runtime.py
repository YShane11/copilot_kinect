import os
import subprocess
import sys


def print_section(title):
    print(f'\n[{title}]')


def run_nvidia_smi():
    try:
        output = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=name,driver_version,memory.total',
                '--format=csv,noheader',
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=8,
        )
        print(output.strip())
    except Exception as exc:  # pylint: disable=broad-except
        print(f'nvidia-smi unavailable: {exc}')


def check_torch():
    try:
        import torch  # pylint: disable=import-outside-toplevel

        print(f'torch: {torch.__version__}')
        print(f'torch cuda build: {torch.version.cuda}')
        print(f'cuda available: {torch.cuda.is_available()}')
        print(f'cuda device count: {torch.cuda.device_count()}')
        if torch.cuda.is_available():
            print(f'cuda device 0: {torch.cuda.get_device_name(0)}')
    except Exception as exc:  # pylint: disable=broad-except
        print(f'torch unavailable: {exc}')


def check_onnxruntime():
    try:
        import onnxruntime as ort  # pylint: disable=import-outside-toplevel

        print(f'onnxruntime: {ort.__version__}')
        print(f'providers: {ort.get_available_providers()}')
    except Exception as exc:  # pylint: disable=broad-except
        print(f'onnxruntime unavailable: {exc}')


def check_imports():
    for package_name in ('ultralytics', 'insightface'):
        try:
            module = __import__(package_name)
            print(f'{package_name}: {getattr(module, "__version__", "installed")}')
        except Exception as exc:  # pylint: disable=broad-except
            print(f'{package_name}: unavailable ({exc})')


def main():
    print(f'python: {sys.executable}')
    print_section('GPU')
    run_nvidia_smi()
    print_section('Torch / YOLO')
    check_torch()
    print(f'YOLO_DEVICE: {os.getenv("YOLO_DEVICE", "auto")}')
    print_section('ONNX Runtime / InsightFace')
    check_onnxruntime()
    print(f'INSIGHTFACE_PROVIDERS: {os.getenv("INSIGHTFACE_PROVIDERS", "auto")}')
    print(f'INSIGHTFACE_CTX_ID: {os.getenv("INSIGHTFACE_CTX_ID", "auto")}')
    print_section('Imports')
    check_imports()


if __name__ == '__main__':
    main()
