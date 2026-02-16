"""
CRAFT validation script — mirrors the Java Craft wrapper exactly.

Usage:
    python craft_validate.py <image_path> [model_path]

Example:
    python craft_validate.py ~/projects/java/inference4j/assets/images/product_placement.jpg \
                             ~/projects/java/inference4j/assets/models/craft/model.onnx
"""

import sys
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw


# --- Same constants as Java Craft.java ---
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TARGET_SIZE = 1280
TEXT_THRESHOLD = 0.7
LOW_TEXT_THRESHOLD = 0.4
MIN_COMPONENT_AREA = 10


def round_to_multiple_of_32(value):
    return max(32, ((value + 31) // 32) * 32)


def preprocess(image_path, target_size):
    """Mirrors Craft.resizeForCraft() + Craft.imageToTensor()"""
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    scale = target_size / max(orig_w, orig_h)
    scaled_w = round_to_multiple_of_32(round(orig_w * scale))
    scaled_h = round_to_multiple_of_32(round(orig_h * scale))

    resized = img.resize((scaled_w, scaled_h), Image.BILINEAR)

    actual_scale_x = scaled_w / orig_w
    actual_scale_y = scaled_h / orig_h
    actual_scale = min(actual_scale_x, actual_scale_y)

    # To numpy, normalize with ImageNet stats
    pixels = np.array(resized, dtype=np.float32) / 255.0  # [H, W, 3]
    pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD

    # HWC -> NCHW
    tensor = pixels.transpose(2, 0, 1)[np.newaxis, ...]  # [1, 3, H, W]

    print(f"Original: {orig_w}x{orig_h}")
    print(f"Resized:  {scaled_w}x{scaled_h} (scale={actual_scale:.4f})")
    print(f"Tensor:   {tensor.shape}, dtype={tensor.dtype}")
    print(f"Pixel range: [{tensor.min():.3f}, {tensor.max():.3f}]")

    return tensor, actual_scale, orig_w, orig_h


def connected_components(binary, width, height):
    """Mirrors Craft.connectedComponents() — BFS flood-fill, 4-connectivity"""
    labels = np.zeros(width * height, dtype=np.int32)
    current_label = 0

    for y in range(height):
        for x in range(width):
            idx = y * width + x
            if not binary[idx] or labels[idx] != 0:
                continue

            current_label += 1
            queue = [(x, y)]
            labels[idx] = current_label

            while queue:
                cx, cy = queue.pop(0)
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue
                    nidx = ny * width + nx
                    if not binary[nidx] or labels[nidx] != 0:
                        continue
                    labels[nidx] = current_label
                    queue.append((nx, ny))

    return labels, current_label


def postprocess(region_score, affinity_score, heatmap_h, heatmap_w,
                text_threshold, low_text_threshold, min_area, scale, orig_w, orig_h):
    """Mirrors Craft.postProcess() exactly"""
    # 1. Combine
    combined = np.clip(region_score + affinity_score, 0, 1)

    # 2. Binary threshold
    binary = combined >= low_text_threshold

    print(f"\nHeatmap: {heatmap_w}x{heatmap_h}")
    print(f"Region score range:   [{region_score.min():.4f}, {region_score.max():.4f}]")
    print(f"Affinity score range: [{affinity_score.min():.4f}, {affinity_score.max():.4f}]")
    print(f"Combined range:       [{combined.min():.4f}, {combined.max():.4f}]")
    print(f"Foreground pixels (>= {low_text_threshold}): {binary.sum()} / {len(binary)}")

    # 3. Connected components
    labels, max_label = connected_components(binary.ravel(), heatmap_w, heatmap_h)
    print(f"Connected components: {max_label}")

    if max_label == 0:
        return []

    # 4. Per-component stats
    regions = []
    for label_id in range(1, max_label + 1):
        mask = labels == label_id
        area = mask.sum()
        if area < min_area:
            continue

        ys, xs = np.where(mask.reshape(heatmap_h, heatmap_w))
        mean_score = region_score.reshape(heatmap_h, heatmap_w)[ys, xs].mean()
        if mean_score < text_threshold:
            continue

        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()

        # Scale to original image coords
        x1 = max(0, min(min_x * 2 / scale, orig_w))
        y1 = max(0, min(min_y * 2 / scale, orig_h))
        x2 = max(0, min((max_x + 1) * 2 / scale, orig_w))
        y2 = max(0, min((max_y + 1) * 2 / scale, orig_h))

        regions.append({
            "box": (x1, y1, x2, y2),
            "confidence": float(mean_score),
            "area": int(area),
        })

    regions.sort(key=lambda r: r["confidence"], reverse=True)
    return regions


def run(image_path, model_path):
    # Preprocess
    tensor, scale, orig_w, orig_h = preprocess(image_path, TARGET_SIZE)

    # Run inference
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    print(f"\nModel input:  {input_name}")
    print(f"Model outputs: {output_names}")

    results = session.run(None, {input_name: tensor})

    for i, (name, result) in enumerate(zip(output_names, results)):
        print(f"  {name}: shape={result.shape}, range=[{result.min():.4f}, {result.max():.4f}]")

    # Find score_map — the tensor with last dim = 2
    score_map = None
    for i, name in enumerate(output_names):
        if name == "score_map" or results[i].shape[-1] == 2:
            score_map = results[i]
            break
    if score_map is None:
        score_map = results[0]

    heatmap_h, heatmap_w = score_map.shape[1], score_map.shape[2]
    region_score = score_map[0, :, :, 0].ravel()
    affinity_score = score_map[0, :, :, 1].ravel()

    # Postprocess
    regions = postprocess(region_score, affinity_score, heatmap_h, heatmap_w,
                          TEXT_THRESHOLD, LOW_TEXT_THRESHOLD, MIN_COMPONENT_AREA,
                          scale, orig_w, orig_h)

    print(f"\nDetected {len(regions)} text regions:")
    for i, r in enumerate(regions):
        x1, y1, x2, y2 = r["box"]
        print(f"  {i+1}. [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] "
              f"(confidence={r['confidence']:.4f}, area={r['area']})")

    # Save annotated image
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for r in regions:
        x1, y1, x2, y2 = r["box"]
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)

    # Build output path: same dir, _annotated suffix
    from pathlib import Path
    p = Path(image_path)
    out_path = p.parent / f"{p.stem}_annotated_python{p.suffix}"
    img.save(out_path)
    print(f"\nAnnotated image saved to {out_path}")

    return regions


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <image_path> [model_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "output/model.onnx"

    run(image_path, model_path)
