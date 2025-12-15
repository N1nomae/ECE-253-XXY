import os

def polygon_to_bbox(xs, ys):
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    return xmin, ymin, xmax, ymax

def bbox_to_yolo(xmin, ymin, xmax, ymax):
    # Inputs are normalized coordinates in [0, 1], so the center/size are also normalized.
    xc = (xmin + xmax) / 2
    yc = (ymin + ymax) / 2
    w = (xmax - xmin)
    h = (ymax - ymin)
    return xc, yc, w, h

def convert_polygon_txt_to_yolo(input_txt_path, output_txt_path):
    with open(input_txt_path, "r") as f:
        lines = f.read().strip().splitlines()

    skipped_count = 0
    with open(output_txt_path, "w") as out:
        for line_idx, line in enumerate(lines, 1):
            parts = line.strip().split()
            if len(parts) < 3:
                skipped_count += 1
                continue

            class_id = parts[0]
            coords = list(map(float, parts[1:]))

            # Coordinates must come in (x, y) pairs.
            if len(coords) % 2 != 0:
                skipped_count += 1
                continue

            # Split the flat list into x and y arrays.
            xs = coords[0::2]
            ys = coords[1::2]

            # Need at least 3 points (a valid polygon).
            if len(xs) < 3:
                skipped_count += 1
                continue

            # Expect normalized polygon coordinates in [0, 1].
            if any(x < 0 or x > 1 for x in xs) or any(y < 0 or y > 1 for y in ys):
                skipped_count += 1
                continue

            
            # Compute the tight axis-aligned bounding box.
            xmin, ymin, xmax, ymax = polygon_to_bbox(xs, ys)

            # Convert to YOLO bbox format: (xc, yc, w, h), all normalized.
            xc, yc, w, h = bbox_to_yolo(xmin, ymin, xmax, ymax)


            # Filter overly large boxes (often annotation errors).
            if w > 0.95 or h > 0.95:
                skipped_count += 1
                continue

            # Filter tiny boxes (likely noise).
            if w < 0.01 or h < 0.01:
                skipped_count += 1
                continue

            out.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    if skipped_count > 0:
        print(f"生成 YOLO 标签: {output_txt_path} (跳过了 {skipped_count} 个无效标注)")
    else:
        print(f"生成 YOLO 标签: {output_txt_path}")

if __name__ == "__main__":
    # Example: batch-convert polygon annotations to YOLO bbox labels.
    input_folder  = "polygon_txts"
    output_folder = "labels_yolo"

    os.makedirs(output_folder, exist_ok=True)

    for fname in os.listdir(input_folder):
        if not fname.endswith(".txt"):
            continue
        input_path  = os.path.join(input_folder, fname)
        output_path = os.path.join(output_folder, fname)
        convert_polygon_txt_to_yolo(input_path, output_path)
