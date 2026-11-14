from pathlib import Path
from collections import Counter, defaultdict
from statistics import median
from PIL import Image

root = Path("OPIXray")
CLASSES = ["Folding_Knife", "Multi-tool_Knife", "Scissor", "Straight_Knife", "Utility_Knife"]

def read_list(p):
    return [Path(l.strip()).stem for l in open(p, encoding="utf-8") if l.strip()]

def parse_ann(path):
    boxes = []
    for line in open(path, encoding="utf-8"):
        t = line.split()
        cls = [x for x in t if x in CLASSES]
        nums = [x for x in t if x.lstrip("-").isdigit()]
        if cls and len(nums) >= 4:
            x1, y1, x2, y2 = map(int, nums[-4:])
            boxes.append((cls[0], x1, y1, x2, y2))
    return boxes

groups = {
    "train": (read_list(root / "splits/train.txt"), "train"),
    "val":   (read_list(root / "splits/val.txt"), "train"),
    "test":  (read_list(root / "test/test_knife.txt"), "test"),
}
for i in (1, 2, 3):
    groups[f"test-{i}"] = (read_list(root / f"test/test_knife-{i}.txt"), "test")

for gname, (names, split) in groups.items():
    ann_dir = root / split / f"{split}_annotation"
    cls_count, per_img, w_list, h_list, missing, empty = Counter(), [], [], [], 0, 0
    for n in names:
        p = ann_dir / f"{n}.txt"
        if not p.exists():
            missing += 1
            continue
        boxes = parse_ann(p)
        if not boxes:
            empty += 1
            continue
        per_img.append(len(boxes))
        for c, x1, y1, x2, y2 in boxes:
            cls_count[c] += 1
            w_list.append(x2 - x1)
            h_list.append(y2 - y1)
    print(f"\n== {gname}: {len(names)} ảnh | thiếu annotation: {missing} | không đọc được box: {empty}")
    if cls_count:
        print("  số crop:", sum(cls_count.values()), "| tối đa box/ảnh:", max(per_img))
        print("  theo lớp:", {c: cls_count[c] for c in CLASSES})
        print(f"  box trung vị: {median(w_list)}x{median(h_list)} (rộng x cao)")

# kích thước ảnh (lấy mẫu 200 ảnh train)
sizes = Counter()
for n in groups["train"][0][:200]:
    sizes[Image.open(root / "train/train_image" / f"{n}.jpg").size] += 1
print("\nKích thước ảnh (mẫu 200):", sizes.most_common(5))

# in thử annotation đầu tiên để kiểm tra định dạng
first = groups["train"][0][0]
print("\nAnnotation mẫu:", first)
print(open(root / "train/train_annotation" / f"{first}.txt", encoding="utf-8").read()[:300])