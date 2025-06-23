from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torch

# 1) 전체 데이터셋 로드
full_dataset = datasets.ImageFolder(
    root="./data",
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std =[0.229,0.224,0.225])
    ])
)

# 2) 클래스별 샘플 개수 집계 → 클래스 가중치 계산
counts_per_class = [0] * len(full_dataset.classes)
for _, label in full_dataset.samples:
    counts_per_class[label] += 1
total = float(sum(counts_per_class))
class_weights = [total / c for c in counts_per_class]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# 3) 훈련/검증 분할 (80% : 20%)
total_samples = len(full_dataset)
train_count = int(0.8 * total_samples)
val_count   = total_samples - train_count
train_dataset, val_dataset = random_split(
    full_dataset,
    [train_count, val_count],
    generator=torch.Generator().manual_seed(42)
)

# 4) WeightedRandomSampler → train_loader 생성
weights_per_sample = [class_weights[label] for _, label in full_dataset.samples]
train_indices = train_dataset.indices
train_weights = [weights_per_sample[i] for i in train_indices]
train_sampler = WeightedRandomSampler(
    weights=train_weights,
    num_samples=len(train_indices),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True
)

# 5) val_loader 생성 (shuffle=False)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"전체 샘플: {total_samples}, Train: {len(train_dataset)}, Val: {len(val_dataset)}")
print("클래스별 개수:", dict(zip(full_dataset.classes, counts_per_class)))
print("클래스 가중치:", dict(zip(full_dataset.classes, class_weights)))
