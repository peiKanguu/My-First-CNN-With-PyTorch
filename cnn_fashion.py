import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


# === 配置类 ===
class Config:
    def __init__(self):
        self.ROOT_DATA_DIR = "fashionMNISTDir"
        self.epochs = 10
        self.batch_size = 32
        self.learning_rate = 0.01
        self.image_size = (28, 28)
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using Device: {self.DEVICE}")
        self.SEED = 2022

config = Config()

# === 数据加载与处理 ===
def load_data(config):
    transform = transforms.ToTensor()
    train_data = datasets.FashionMNIST(root=config.ROOT_DATA_DIR, train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root=config.ROOT_DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

    return train_data, test_data, train_loader, test_loader

def get_label_map(train_data):
    given_label_map = train_data.class_to_idx
    label_map = {val: key for key, val in given_label_map.items()}
    return label_map

def view_sample_img(data, index, label_map):
    plt.imshow(data.data[index], cmap="gray")
    plt.title(f"Label: {label_map[data.targets[index].item()]}")
    plt.axis("off")
    plt.show()

# === 模型定义 ===
class CNN(nn.Module):
    def __init__(self, in_, out_):
        super(CNN, self).__init__()

        self.conv_pool_01 = nn.Sequential(
            nn.Conv2d(in_channels=in_, out_channels=8, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_pool_02 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.Flatten = nn.Flatten()
        self.FC_01 = nn.Linear(16 * 4 * 4, 128)
        self.FC_02 = nn.Linear(128, 64)
        self.FC_03 = nn.Linear(64, out_)

    def forward(self, x):
        x = self.conv_pool_01(x)
        x = self.conv_pool_02(x)
        x = self.Flatten(x)
        x = F.relu(self.FC_01(x))
        x = F.relu(self.FC_02(x))
        x = self.FC_03(x)
        return x

# === 单张预测函数 ===
def predict(data, model, label_map, device, idx=0):
    images, labels = data

    img = images[idx].unsqueeze(0).float() / 255.0  # ✅ 保持张量格式 + 转为 float + 归一化
    label = labels[idx]

    # 显示图像
    plt.imshow(img.squeeze(), cmap="gray")

    # 模型预测
    logit = model(img.unsqueeze(0).to(device))  # 再次添加 batch 维度，变为 [1, 1, 28, 28]
    pred_prob = F.softmax(logit, dim=1)
    argmax = torch.argmax(pred_prob).item()

    predicted_label = label_map[argmax]
    actual_label = label_map[label.item()]

    plt.title(f"Actual: {actual_label} | Predicted: {predicted_label}")
    plt.axis("off")
    plt.show()

    return predicted_label, actual_label

# === 主程序开始 ===
def main():
    train_data, test_data, train_loader, test_loader = load_data(config)
    label_map = get_label_map(train_data)

    view_sample_img(train_data, 0, label_map)

    model = CNN(1, 10).to(config.DEVICE)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}") as tqdm_epoch:
            for images, labels in tqdm_epoch:
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                tqdm_epoch.set_postfix(loss=loss.item())
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

    # 保存模型
    os.makedirs("model_dir", exist_ok=True)
    model_path = os.path.join("model_dir", "cnn_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved at: {model_path}")

    # 加载模型
    loaded_model = CNN(1, 10).to(config.DEVICE)
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()
    print("✅ Model loaded.")

    # 模型评估
    pred = np.array([])
    target = np.array([])

    with torch.no_grad():
        for batch in test_loader:
            images = batch[0].to(config.DEVICE)
            labels = batch[1].to(config.DEVICE)

            outputs = loaded_model(images)
            predictions = torch.argmax(outputs, dim=1)

            pred = np.concatenate((pred, predictions.cpu().numpy()))
            target = np.concatenate((target, labels.cpu().numpy()))

    cm = confusion_matrix(target, pred)
    print("📊 Confusion Matrix:\n", cm)

    # 单图预测
    predict((test_data.data, test_data.targets), loaded_model, label_map, config.DEVICE, idx=3)

if __name__ == "__main__":
    main()
