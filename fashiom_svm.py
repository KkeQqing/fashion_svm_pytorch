import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# 定义转换
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# 下载训练数据
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 下载测试数据
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

import numpy as np
from sklearn.svm import SVC

def prepare_data(loader):
    data = []
    labels = []
    for images, image_labels in loader:
        n_samples = images.shape[0]
        images_flat = images.view(n_samples, -1).numpy()
        data.extend(images_flat)
        labels.extend(image_labels.numpy())
    return np.array(data), np.array(labels)

# 准备训练和测试数据
X_train, y_train = prepare_data(trainloader)
X_test, y_test = prepare_data(testloader)

# 创建SVM模型实例
model = SVC(kernel='linear')

# 训练模型
model.fit(X_train, y_train)

# 预测
predictions = model.predict(X_test)

from sklearn.metrics import accuracy_score

print(f'Accuracy: {accuracy_score(y_test, predictions)}')

# 1. 混淆矩阵
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=datasets.FashionMNIST.classes,
            yticklabels=datasets.FashionMNIST.classes)
plt.title('Confusion Matrix - SVM on Fashion-MNIST')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# 2. 分类报告（打印 + 可视化为条形图）
print("\nClassification Report:\n")
report = classification_report(y_test, predictions, target_names=datasets.FashionMNIST.classes, output_dict=True)
print(classification_report(y_test, predictions, target_names=datasets.FashionMNIST.classes))

# 提取每个类别的 F1-score
f1_scores = [report[label]['f1-score'] for label in datasets.FashionMNIST.classes]

plt.figure(figsize=(10, 6))
plt.bar(datasets.FashionMNIST.classes, f1_scores, color='skyblue')
plt.title('F1-Score per Class')
plt.xlabel('Class')
plt.ylabel('F1-Score')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# 3. 可视化部分测试图像及预测结果
# 需要原始图像（未归一化），所以重新加载测试集（不带 Normalize）
transform_raw = transforms.ToTensor()  # 仅转为张量，不归一化
testset_raw = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform_raw)
testloader_raw = torch.utils.data.DataLoader(testset_raw, batch_size=64, shuffle=False)

# 获取前 N 张图像用于展示
N = 25
images_show = []
labels_show = []
for imgs, labs in testloader_raw:
    images_show.append(imgs)
    labels_show.append(labs)
    if len(torch.cat(images_show)) >= N:
        break

images_show = torch.cat(images_show)[:N]
labels_show = torch.cat(labels_show)[:N]


# 对这些图像做与训练时相同的预处理（展平 + 归一化），用于预测
def preprocess_for_svm(img_tensor):
    # img_tensor: [N, 1, 28, 28]
    img_flat = img_tensor.view(img_tensor.size(0), -1).numpy()  # [N, 784]
    img_flat = (img_flat - 0.5) / 0.5  # 应用与训练时相同的 Normalize((0.5,), (0.5,))
    return img_flat


X_show = preprocess_for_svm(images_show)
pred_show = model.predict(X_show)

# 绘图
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
axes = axes.ravel()
for i in range(N):
    img = images_show[i].squeeze().numpy()
    true_label = datasets.FashionMNIST.classes[labels_show[i]]
    pred_label = datasets.FashionMNIST.classes[pred_show[i]]
    color = 'green' if labels_show[i] == pred_show[i] else 'red'

    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
    axes[i].axis('off')
plt.suptitle('Test Image Predictions (Green=Correct, Red=Wrong)', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()