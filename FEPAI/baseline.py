'''
1.赛题名称：语义分割-地表建筑物识别
2.赛题目标：通过本次赛题可以引导大家熟练掌握语义分割任务的定义，具体的解题流程和相应的模型，并掌握语义分割任务的发展。
3.赛题任务：赛题以计算机视觉为背景，要求选手使用给定的航拍图像训练模型并完成地表建筑物识别任务。
'''
import os
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"

import numpy as np
import pandas as pd
import cv2
import time
import torch
import torch.nn as nn
import torch.utils.data as D
import torchvision
from torchvision import transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import albumentations as A

# -------------------------- 1. 配置参数与基础函数 --------------------------
# 数据路径
TRAIN_MASK_CSV = "./data/train_mask.csv"       # 训练集掩码CSV路径
TRAIN_IMG_DIR = "./data/train/"                # 训练集图像文件夹路径
TEST_CSV = "./data/test_a_samplesubmit.csv"       # 测试集提交模板CSV路径
TEST_IMG_DIR = "./data/test_a/"                   # 测试集图像文件夹路径

# 输出路径
BEST_MODEL_SAVE_PATH = "./results/model_best.pth"          # 最优模型保存路径
SUBMISSION_SAVE_PATH = "./results/submission.csv"           # 预测结果提交文件路径
PREDICTION_VIS_PATH = "./results/prediction_sample.png"     # 预测样本可视化保存路径

# 训练配置
EPOCHES = 10
BATCH_SIZE = 64
IMAGE_SIZE = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {DEVICE}")

# 数据增强流程：尺寸调整→水平翻转→垂直翻转→随机旋转
TRAIN_TRFM = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(),
])

# 图像转换流程：格式转换→尺寸调整→张量转换→标准化
TEST_TRFM = T.Compose([
    T.ToPILImage(),
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize([0.625, 0.448, 0.688], [0.131, 0.177, 0.101]),
])


# -------------------------- 2. 数据处理与数据集类 --------------------------
# RLE编码：对掩码图像进行RLE编码
def rle_encode(im):
    # 将二维掩码图像im按列优先 order='F' 展平为一维数组 pixels
    pixels = im.flatten(order='F')
    # 在序列首尾添加0作为哨兵
    pixels = np.concatenate([[0], pixels, [0]])
    # 记录所有像素值发生变化的位置
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # 将 “跳变点位置” 转换为 “起始位置 + 连续长度” 的编码格式
    runs[1::2] -= runs[::2]
    # 将runs数组中的整数转换为字符串，用空格拼接成最终的RLE编码字符串
    return ' '.join(str(x) for x in runs)


# RLE编码：将RLE编码解码为掩码图像
def rle_decode(mask_rle, shape=(512, 512)):
    # 如果输入的是空字符串，表示该图像没有建筑物，直接返回全为0的掩码图像
    if not mask_rle:
        return np.zeros(shape, dtype=np.uint8)
    # 将RLE字符串按空格拆分为列表
    s = mask_rle.split()
    # 将字符串列表转换为整数数组
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    # 调整起始位置并计算结束位置
    starts -= 1
    ends = starts + lengths
    # 生成掩码图像
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    # 重塑为shape对应的二维数组
    return img.reshape(shape, order='F')


class TianChiDataset(D.Dataset):
    def __init__(self, paths, rles=None, transform=None, test_mode=False):
        self.paths = paths
        self.rles = rles if not test_mode else ['' for _ in paths]
        self.transform = transform
        self.test_mode = test_mode

        # 图像转张量：转换为PIL→尺寸调整→转为张量→标准化
        self.to_tensor = T.Compose([
            T.ToPILImage(),
            T.Resize(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize([0.625, 0.448, 0.688], [0.131, 0.177, 0.101]),
        ])

    def __getitem__(self, index):
        # 读取图像并转换为RGB
        img = cv2.imread(self.paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if not self.test_mode:
            # 训练模式：加载掩码并增强
            mask = rle_decode(self.rles[index])
            augments = self.transform(image=img, mask=mask)
            return self.to_tensor(augments['image']), augments['mask'][None]
        else:
            # 测试模式：仅返回图像
            return self.to_tensor(img), ''

    def __len__(self):
        return len(self.paths)


# -------------------------- 3. 模型与损失函数定义 --------------------------
# 构建FCN-ResNet50分割模型
def get_model():
    # 初始化模型
    model = torchvision.models.segmentation.fcn_resnet50(weights=None)

    # 修改输出层为1个通道（二分类）
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
    return model.to(DEVICE)


# Dice损失计算
class SoftDiceLoss(nn.Module):

    def __init__(self, smooth=1., dims=(-2, -1)):
        super().__init__()
        # 平滑项，防止分子 / 分母为 0 导致的计算错误
        self.smooth = smooth
        # 对高度（H）和宽度（W）维度求和，得到每个样本、每个类别的重叠度
        self.dims = dims

    def forward(self, x, y):
        # 计算True Positive：预测和真实掩码都为1的区域
        tp = (x * y).sum(self.dims)
        # 计算False Positive：预测为1但真实为0的区域
        fp = (x * (1 - y)).sum(self.dims)
        # 计算False Negative：预测为0但真实为1的区域
        fn = ((1 - x) * y).sum(self.dims)
        # 计算Dice系数
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        # 返回Dice损失
        return 1 - dc.mean()


# 组合损失函数：80%BCE + 20%Dice
def get_loss_fn():
    # 初始化BCE损失与Dice损失
    bce_fn = nn.BCEWithLogitsLoss()
    dice_fn = SoftDiceLoss()

    # 加权融合两种损失函数
    def loss_fn(y_pred, y_true):
        return 0.8 * bce_fn(y_pred, y_true) + 0.2 * dice_fn(y_pred.sigmoid(), y_true)
    return loss_fn


# -------------------------- 4. 核心功能函数--------------------------
def load_data():
    # 读取训练掩码
    train_mask = pd.read_csv(TRAIN_MASK_CSV, sep='\t', names=['name', 'mask'])
    # 拼接训练图像完整路径
    train_mask['name'] = train_mask['name'].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x))

    # 验证RLE编码一致性
    sample_img_path = train_mask['name'].iloc[0]
    if os.path.exists(sample_img_path):
        sample_mask = rle_decode(train_mask['mask'].iloc[0])
        encode_ok = (rle_encode(sample_mask) == train_mask['mask'].iloc[0])
        print(f"RLE编码验证: {'通过' if encode_ok else '失败'}")
    else:
        print(f"样本图像 {sample_img_path} 不存在，跳过RLE验证")
        encode_ok = False

    # 构建数据集
    full_dataset = TianChiDataset(
        paths=train_mask['name'].values,
        rles=train_mask['mask'].fillna('').values,
        transform=TRAIN_TRFM,
        test_mode=False
    )

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split_point = int(np.floor(0.8 * dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx = indices[:split_point]
    valid_idx = indices[split_point:]
    
    print(f"数据集总数: {dataset_size}, 训练集数量: {len(train_idx)}, 验证集数量: {len(valid_idx)}")
    
    # 构建加载器
    train_loader = D.DataLoader(
        D.Subset(full_dataset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,  # 训练集加载器需要再次打乱
        num_workers=0
    )
    valid_loader = D.DataLoader(
        D.Subset(full_dataset, valid_idx),
        batch_size=BATCH_SIZE,
        shuffle=False, # 验证集加载器不需要打乱
        num_workers=0
    )
    return train_loader, valid_loader, full_dataset


# 模型训练
def train_model(train_loader, valid_loader):
    # 初始化模型、优化器、损失函数
    model = get_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    loss_fn = get_loss_fn()
    best_loss = float('inf')

    # 输出训练日志
    print("\n" + r"Epoch |  Train Loss |  Valid Loss | Time(m)")
    print("-" * 40)

    for epoch in range(1, EPOCHES + 1):
        start_time = time.time()
        model.train()
        train_losses = []

        # 训练轮次
        for img, target in tqdm(train_loader, desc=f"Epoch {epoch}"):
            img, target = img.to(DEVICE), target.float().to(DEVICE)
            optimizer.zero_grad()
            output = model(img)['out']
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # 验证轮次
        model.eval()
        valid_losses = []
        with torch.no_grad():
            for img, target in valid_loader:
                img, target = img.to(DEVICE), target.float().to(DEVICE)
                output = model(img)['out']
                valid_losses.append(loss_fn(output, target).item())

        # 计算指标并打印
        avg_train_loss = np.mean(train_losses)
        avg_valid_loss = np.mean(valid_losses)
        train_time = (time.time() - start_time) / 60

        print(f"{epoch:^5d} |  {avg_train_loss:.4f}   |  {avg_valid_loss:.4f}   | {train_time:.2f}")

        # 保存最优模型
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
            print(f"保存最优模型至 {BEST_MODEL_SAVE_PATH}（验证损失: {best_loss:.4f}）")
    return model


# 模型测试
def predict_test(model):
    """预测测试集并生成提交文件"""
    # 读取测试集信息
    test_mask = pd.read_csv(TEST_CSV, sep='\t', names=['name', 'mask'])

    # 拼接测试图像完整路径
    test_mask['name'] = test_mask['name'].apply(lambda x: os.path.join(TEST_IMG_DIR, x))

    # 构建测试数据集
    test_dataset = TianChiDataset(
        paths=test_mask['name'].values,
        test_mode=True
    )

    # 预测并编码
    model.eval()
    submissions = []
    for idx, (img, _) in tqdm(enumerate(test_dataset), total=len(test_dataset)):
        with torch.no_grad():
            # 模型预测
            img = img.to(DEVICE)[None]
            pred = model(img)['out'][0][0].sigmoid().cpu().numpy()
            # 二值化+恢复尺寸
            pred_mask = (pred > 0.5).astype(np.uint8)
            pred_mask = cv2.resize(pred_mask, (512, 512))
            # RLE编码
            rle = rle_encode(pred_mask)
            # 提取文件名
            img_filename = os.path.basename(test_mask['name'].iloc[idx])
            submissions.append([img_filename, rle])

    # 保存提交文件
    pd.DataFrame(submissions).to_csv(SUBMISSION_SAVE_PATH, index=None, header=None, sep='\t')
    print(f"\n预测结果已保存至: {SUBMISSION_SAVE_PATH}")

    # 可视化第一个样本
    if submissions:
        sample_name, sample_rle = submissions[0]
        sample_img_path = os.path.join(TEST_IMG_DIR, sample_name)
        if os.path.exists(sample_img_path):
            sample_img = cv2.imread(sample_img_path)
            sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
            sample_mask = rle_decode(sample_rle)

            plt.figure(figsize=(12, 6))
            plt.subplot(121), plt.imshow(sample_img), plt.title("测试图像")
            plt.subplot(122), plt.imshow(sample_mask, cmap='gray'), plt.title("预测掩码")
            plt.savefig(PREDICTION_VIS_PATH)
            print(f"预测样本可视化已保存至: {PREDICTION_VIS_PATH}")
        else:
            print(f"测试样本 {sample_img_path} 不存在，跳过可视化")


# -------------------------- 5. 主函数（串联流程） --------------------------
def main():
    # 1. 加载数据
    print("=== 加载数据 ===")
    train_loader, valid_loader, _ = load_data()

    # 2. 训练模型
    print("\n=== 模型训练 ===")
    model = train_model(train_loader, valid_loader)

    # 3. 预测测试集
    print("\n=== 模型预测 ===")
    predict_test(model)


if __name__ == "__main__":
    main()