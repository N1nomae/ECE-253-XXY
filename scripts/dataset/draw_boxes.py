import cv2
import os

IMG_DIR = "images"         # Folder containing the original images
LABEL_DIR = "labels_yolo"  # Folder containing YOLO bbox label .txt files
OUT_DIR = "vis"            # Output folder for visualizations

os.makedirs(OUT_DIR, exist_ok=True)

for img_name in os.listdir(IMG_DIR):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMG_DIR, img_name)
    label_path = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Read YOLO bbox labels (if present) and draw them on the image
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.read().splitlines()
        for line in lines:
            class_id, xc, yc, bw, bh = line.split()
            class_id = int(class_id)
            xc, yc, bw, bh = map(float, (xc, yc, bw, bh))

            # Convert normalized YOLO coords back to pixel coordinates
            xmin = int((xc - bw/2) * w)
            ymin = int((yc - bh/2) * h)
            xmax = int((xc + bw/2) * w)
            ymax = int((yc + bh/2) * h)

            # Draw bounding box and class id
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(img, str(class_id), (xmin, ymin-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Save visualization image
    out_path = os.path.join(OUT_DIR, img_name)
    cv2.imwrite(out_path, img)
    print(f"saved {out_path}")
