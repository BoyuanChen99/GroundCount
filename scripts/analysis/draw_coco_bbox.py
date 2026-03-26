import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def main():
    ### Step 0: Initialize file paths
    coco_dir = "../../../data/coco"
    annotations_dir = os.path.join(coco_dir, "annotations")
    image_file = "COCO_train2014_000000576290.jpg"
    
    ### Step 1: Load annotations
    ann_file = os.path.join(annotations_dir, "instances_train2014.json")
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    ### Step 2: Extract image ID from filename
    image_id = int(image_file.split('_')[-1].replace('.jpg', ''))
    
    ### Step 3: Build category id to name mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    ### Step 4: Get all annotations for this image
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
    
    ### Step 5: Load and display image
    image_path = os.path.join(coco_dir, "train2014", image_file)
    img = Image.open(image_path)
    img_w, img_h = img.size
    
    fig, ax = plt.subplots(1, figsize=(36, 30))
    ax.imshow(img)
    
    ### Step 6: Draw all bounding boxes
    colors = plt.cm.tab20.colors
    for i, ann in enumerate(annotations):
        x, y, w, h = ann['bbox']
        color = colors[i % len(colors)]
        rect = patches.Rectangle((x, y), w, h, linewidth=5, edgecolor=color, facecolor='none', alpha=0.5)
        ax.add_patch(rect)
        cat_name = cat_id_to_name[ann['category_id']]
        ax.text(x, y - 5, cat_name, color=color, fontsize=50, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    ax.set_xlim(-10, img_w + 10)
    ax.set_ylim(img_h + 20, -50)
    
    ax.set_title(f"{image_file} - {len(annotations)} objects", fontsize=50)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("bbox_visualization.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Found {len(annotations)} bounding boxes")


if __name__ == "__main__":
    main()