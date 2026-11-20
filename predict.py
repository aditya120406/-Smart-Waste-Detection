"""
Standalone Command-Line Inference Script for Smart Waste Classification.

Usage:
    python predict.py --image sample.jpg
"""

import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.inference import WasteClassifier
from src.preprocessing import validate_image_file


def main():
    parser = argparse.ArgumentParser(description="Classify a waste item image using trained CNN")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file (JPEG, PNG, etc.)")
    parser.add_argument("--model", type=str, default=None, help="Optional custom model weights path")
    args = parser.parse_args()

    image_path = os.path.abspath(args.image)
    if not os.path.exists(image_path):
        print(f"[Error] Image file not found: {image_path}")
        sys.exit(1)

    # Validate image file
    is_valid, err_msg = validate_image_file(image_path)
    if not is_valid:
        print(f"[Error] Invalid image: {err_msg}")
        sys.exit(1)

    # Initialize classifier
    classifier = WasteClassifier(model_path=args.model)
    if not classifier.is_loaded:
        print("\n" + "=" * 60)
        print(" [!] MODEL WEIGHTS NOT FOUND")
        print("=" * 60)
        print(classifier.load_error)
        print("\nPlease train the model first by running:")
        print("  python train.py")
        print("or follow the Google Colab instructions in README.md.")
        sys.exit(1)

    print(f"\n[Processing] Analyzing image: {os.path.basename(image_path)}...")
    result, error = classifier.predict(image_path)
    if error:
        print(f"[Inference Error] {error}")
        sys.exit(1)

    # Display clean formatted results
    print("\n" + "=" * 60)
    print("        SMART WASTE CLASSIFICATION RESULT")
    print("=" * 60)
    print(f" Predicted Class   : {result['display_name'].upper()}")
    print(f" Confidence Score  : {result['confidence']}%")
    print(f" Recyclable        : {'YES (Recyclable)' if result['recyclable'] else 'NO (General Trash)'}")
    print(f" Suggested Bin     : {result['disposal_bin']}")
    print("-" * 60)
    print(f" Recommendation    : {result['recommendation']}")
    print("\n Disposal Tips:")
    for tip in result.get('tips', []):
        print(f"  * {tip}")
    if result.get('fun_fact'):
        print(f"\n Fact: {result['fun_fact']}")
    print("-" * 60)
    print(" Probability Distribution:")
    for cls_name, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(prob // 5)
        print(f"  {cls_name:<12}: {prob:>5.1f}% | {bar}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
