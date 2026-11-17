"""Predict one radar spectrogram image with the preserved TorchScript model.

The original training validation transform converted RGB bytes to [0, 1] and
normalized each channel with mean=0.5, std=0.5. No resizing was used there.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch


# ImageFolder orders directories alphabetically; confirmed by the notebook log.
CLASS_NAMES = (
    "16-QAM", "B-FM", "BPSK", "Barker", "CPFSK", "DSB-AM",
    "GFSK", "LFM", "PAM4", "QPSK", "Rect", "StepFM",
)
IMAGE_SIZE = (224, 224)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "radar_classifier_torchscript.pt"


def prepare_image(image_path: Path) -> torch.Tensor:
    """Match the notebook's eval transform: RGB -> ToTensor -> Normalize(.5)."""
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        if rgb.size != IMAGE_SIZE:
            raise ValueError(
                f"Expected an image of {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} pixels "
                f"(got {rgb.width}x{rgb.height}). "
                "The original notebook did not resize validation images."
            )
        array = np.asarray(rgb, dtype=np.uint8).copy()

    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
    return ((tensor - 0.5) / 0.5).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a 12-class radar spectrogram with TorchScript.")
    parser.add_argument("--image", type=Path, required=True, help="Path to a 224x224 spectrogram image")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="TorchScript .pt file")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    if not args.model.is_file():
        parser.error(f"Model not found: {args.model}")

    try:
        input_tensor = prepare_image(args.image)
        model = torch.jit.load(str(args.model), map_location="cpu")
        model.eval()
        with torch.inference_mode():
            logits = model(input_tensor)
            if logits.ndim != 2 or logits.shape != (1, len(CLASS_NAMES)):
                raise RuntimeError(f"Expected 1x{len(CLASS_NAMES)} logits, got {tuple(logits.shape)}")
            scores = torch.softmax(logits, dim=1)[0]
            index = int(torch.argmax(scores))
        print(f"Predicted class: {CLASS_NAMES[index]}")
        print(f"Softmax score: {float(scores[index]):.2%} (not calibrated)")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Prediction failed: {exc}\n")


if __name__ == "__main__":
    main()
