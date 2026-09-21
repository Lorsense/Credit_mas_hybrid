"""Check the Qwen3.5 runtime before creating expensive distributed workers."""
import argparse
from importlib.metadata import PackageNotFoundError, version
import json

CORE = {
    "sglang": "0.5.10", "transformers": "5.3.0", "torch": "2.9.1",
    "torchvision": "0.24.1", "torchaudio": "2.9.1", "torch-memory-saver": "0.0.9",
    "ray": "2.58.0", "tensordict": "0.14.2", "peft": "0.18.1", "accelerate": "1.12.0",
}
KERNELS = {"flash-linear-attention": "0.4.2", "causal-conv1d": "1.6.0"}


def inspect_runtime(require_kernels=False):
    expected = {**CORE, **(KERNELS if require_kernels else {})}
    installed, errors = {}, []
    for package, required in expected.items():
        try:
            installed[package] = version(package)
        except PackageNotFoundError:
            errors.append(f"Missing {package}=={required}")
            continue
        # CUDA wheel local versions such as 2.9.1+cu129 are the same torch ABI.
        if installed[package].split("+", 1)[0] != required:
            errors.append(f"{package}: expected {required}, found {installed[package]}")
    return installed, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-kernels", action="store_true", help="also verify fast linear-attention training kernels")
    parser.add_argument("--check-cuda", action="store_true", help="verify CUDA/BF16 on this host (optional for CPU Ray drivers)")
    args = parser.parse_args()
    installed, errors = inspect_runtime(args.require_kernels)
    if args.check_cuda and not errors:
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            errors.append("This host needs a CUDA GPU with BF16 support")
    print(json.dumps({"installed": installed, "errors": errors}, indent=2))
    if errors:
        parser.exit(1, "Install requirements_qwen35.txt in a fresh Linux/CUDA environment; "
                    "then requirements_qwen35_kernels.txt with --no-build-isolation.\n")


if __name__ == "__main__":
    main()
