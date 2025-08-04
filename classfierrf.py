import os
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import ResNet18Classifier
from tqdm import tqdm

def classify_folder(model_path, image_folder, transform=None, device=None, total_classes=1000, log_interval=5000):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet18Classifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    results = []
    class_count = {}

    with torch.no_grad():
        for idx, img_path in enumerate(tqdm(image_paths, desc="Classifying")):
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)
            outputs = model(image)
            _, pred = torch.max(outputs, 1)
            label = pred.item()
            results.append((os.path.basename(img_path), label))
            class_count[label] = class_count.get(label, 0) + 1

            # 每分类 log_interval 张图像打印一次覆盖情况
            if (idx + 1) % log_interval == 0:
                covered_classes = len(class_count)
                coverage_percent = covered_classes / total_classes * 100
                all_classes = set(range(total_classes))
                found_classes = set(class_count.keys())
                missing_classes = sorted(list(all_classes - found_classes))
                print(f"\n已分类 {idx + 1} 张图像，当前已覆盖类别数：{covered_classes}/{total_classes} ({coverage_percent:.2f}%)")
                print(f"缺失类别数：{len(missing_classes)}")
                print("缺失类别编号：", missing_classes)

    # 最终统计
    total = len(results)
    covered_classes = len(class_count)
    coverage_percent = covered_classes / total_classes * 100
    print(f"\n最终覆盖类别数：{covered_classes} / {total_classes} ({coverage_percent:.2f}%)")
    
    all_classes = set(range(total_classes))
    found_classes = set(class_count.keys())
    missing_classes = sorted(list(all_classes - found_classes))
    print(f"缺失类别数：{len(missing_classes)}")
    print("缺失类别编号：", missing_classes)

if __name__ == "__main__":
    classify_folder(
        model_path="rgb_classifier1.pth",
        image_folder="generated_samples9",
        total_classes=1000,
        log_interval=5000  # 每5000张图像打印一次
    )


