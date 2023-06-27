import snntorch.functional as SF
import torch
from tqdm import trange

from model.SpikingModel import hybrid_neuron
from spiking.SpikeEncoder import SimpleEncoder

batch_size = 100
data_path = 'data/mnist'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define a transform
transform = transforms.Compose([
    # 像素大小修改
    transforms.Resize((28, 28)),
    # 灰白图像
    transforms.Grayscale(),
    # 张量化
    transforms.ToTensor(),
    # 归一化处理
    transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

net = hybrid_neuron(input_shape=28 * 28,
                    output_shape=10,
                    T=3,
                    batch_size=batch_size).float().to(device)

# 定义了优化器，使用Adam优化算法来更新网络模型的参数。net.parameters() 返回模型中所有需要学习的参数，这些参数将被优化器更新。
optimizer = torch.optim.Adam(
    list(net.parameters()), lr=2e-3, betas=(0.9, 0.999))

# 定义了损失函数，这里使用了自定义的均方误差计数损失函数。损失函数用于度量模型输出与目标值之间的差异。
loss_fn = SF.mse_count_loss(correct_rate=0.9, incorrect_rate=0.1)
# 创建空列表用于存储每个批次的损失值和准确率。
loss_hist = []
acc_hist = []

# 定义训练的总轮数。
num_epochs = 1
# train loop 外层循环遍历每个训练轮数，内层循环遍历训练数据加载器中的每个批次。img 是输入图像的批次，target 是对应的目标值。
for epoch in trange(num_epochs):
    for batch_idx, (img, target) in enumerate(train_loader):
        # 将网络模型设置为训练模式，并将输入图像数据转移到指定的设备上
        net.train()
        img = SimpleEncoder(img_batch=img)
        img, target = img.to(device), target.to(device)
        # 通过网络模型进行前向传播，得到输出结果 spk_rec，然后计算损失值 loss_val，通过调用损失函数 loss_fn 来比较输出和目标值之间的差异。
        spk_rec = net(img)
        loss_val = loss_fn(spk_rec, target)

        # Gradient calculation + weight update
        # 梯度计算和权重更新步骤。首先将优化器的梯度缓存清零，然后通过调用 backward() 方法计算损失相对于模型参数的梯度。最后，调用优化器的 step() 方法来更新模型参数
        optimizer.zero_grad()
        loss_val.backward(retain_graph=True)
        # loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        # 将当前批次的损失值添加到 loss_hist 列表中，并将损失值的缓存清零，以便下一个批次使用。
        loss_hist.append(loss_val.item())
        """
        This step is important.
        """
        loss_val.zero_()

        # print every 25 iterations
        # 每经过25个batch，输出当前epoch和batch的训练损失值，然后计算并输出当前的准确率。
        if batch_idx % 25 == 0:
            # if 1:
            net.eval()
            print(f"Epoch {epoch}, Iteration {batch_idx} \nTrain Loss: {loss_val.item():.2f}")

            # check accuracy on a single batch
            acc = SF.accuracy_rate(spk_rec, target)
            acc_hist.append(acc)
            print(f"Accuracy: {acc * 100:.2f}%\n")
        pass
