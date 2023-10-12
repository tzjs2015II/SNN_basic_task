import torch.autograd
import torch.nn as nn
from snntorch import surrogate

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

"""
定义多层感知机
一个继承自nn.Module的类,表示这是一个神经网络模型
"""


class MLP(nn.Module):
    # 初始化函数
    def __init__(self, input_shape, output_shape):
        # 继承父类的初始化方式
        super(MLP, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        # 创建一个全连接层,设置输入输出的特征维度
        self.fc = nn.Linear(in_features=self.input_shape,
                            out_features=self.output_shape)
        # 创建ReLU的激活函数
        self.ReLU = nn.ReLU()

    # 定义向前传播的过程
    def forward(self, x):
        # 对全连接层进行线性变换
        x = self.fc(x)
        # 对线性变换结果设置激活函数
        x = self.ReLU(x)
        return x


"""
  定义混合神经元模型
"""


class hybrid_neuron(nn.Module):
    def __init__(self, input_shape, output_shape, T, batch_size):
        super(hybrid_neuron, self).__init__()
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_shape = output_shape
        # 这个MLP网络将输入特征映射到输出特征,并将MLP网络的数据类型转换为浮点型，并将其移动到指定的设备上进行计算。
        self.MLP = MLP(input_shape=self.input_shape,
                       output_shape=self.output_shape).float().to(device)
        self.T = T  # num steps
        # 创建了一个快速sigmoid函数的实例，用于模拟神经元的激活函数。
        self.sigmoid = surrogate.fast_sigmoid()

        # 用torch.autograd.Variable创建的Alpha, Eta, Beta张量，可以进行梯度计算和反向传播
        self.alpha = torch.autograd.Variable(torch.tensor(0.),
                                             requires_grad=True).to(device)
        self.eta = torch.autograd.Variable(torch.tensor(0.),
                                           requires_grad=True).to(device)
        self.beta = torch.autograd.Variable(torch.tensor(0.),
                                            requires_grad=True).to(device)
        # 定义了膜电位时间常数和突触常数
        self.tao_u = torch.tensor(5).to(device)  # membrane time constant
        self.tao_w = torch.tensor(1).to(device)  # synaptic constant
        
        # 时间步长
        self.dt = torch.tensor(1).to(device)
        # 状态变量，用于存储神经元的输出状态和膜电位状态。它们被初始化为全零张量
        self.s = torch.zeros(self.output_shape, self.batch_size).to(device)
        self.u = torch.zeros(self.output_shape, self.batch_size).to(device)
        # 定义了一个比例因子k，用于计算膜电位的变化
        self.k = self.dt / self.tao_u
        # 阈值电位，当膜电位超过这个阈值时，神经元会发放脉冲
        self.V_th = torch.tensor(0.2).to(device)
        
        # 是一个可训练的参数矩阵，形状为(self.output_shape, self.input_shape)，
        # 用于模拟突触权重。它被初始化为全零矩阵，并设置为可计算梯度和进行反向传播
        self.P = torch.autograd.Variable(
            torch.zeros((self.output_shape, self.input_shape)),
            requires_grad=True).to(device)

    def forward(self, s_batch):
        # 存储每个时间步骤内的脉冲状态
        spk = []
        # 模型迭代,迭代次数为时间步数
        for m in range(self.T):
            # for s in s_batch:
            # 获取像素强度的batch
            s = s_batch
            # 计算衰减常数，用于模拟突触权重的衰减
            decay_1 = torch.tensor(-m) / self.tao_w
            # 根据膜电位的动力学方程更新膜电位值。通过乘法和加法操作，结合了输入信号、膜电位的历史状态和突触权重。
            self.u = torch.mul((1 - self.s) * (1 - self.k), self.u) + \
                     self.k * ((self.MLP(s) * decay_1.exp()).T + self.alpha * torch.matmul(self.P, s.T))
            # 根据突触权重的动力学方程更新突触权重。通过乘法、加法和矩阵乘法操作，结合了膜电位、输入信号和突触权重的历史状态。
            self.P = self.P * (-self.dt / self.tao_w).exp() + \
                     self.eta * torch.matmul(self.sigmoid(self.u) + self.beta,
                                             s)
            # 根据膜电位和阈值之间的差异确定输出脉冲状态。使用sigmoid函数将膜电位映射到[0, 1]的范围内。
            self.s = self.sigmoid(self.u - self.V_th)
            # 将当前时间步骤的输出脉冲状态添加到列表中
            spk.append(self.s.T)
            # 将当前时间步骤的输出脉冲状态添加到列表中
            self.reset()

            # 实现方法2
        # spk_batch = []
        # for s in s_batch:
        #     spk = []
        #     for m in range(self.T):
        #         decay_1 = torch.tensor(-m) / self.tao_w
        #         self.u = (1 - self.s) * (1 - self.k) * self.u + \
        #                  self.k * (self.MLP(s) * decay_1.exp() + self.alpha * torch.matmul(self.P, s)).sum()
        #         self.P = self.P * (-self.dt / self.tao_w).exp() + \
        #                  self.eta * torch.matmul(torch.stack([self.sigmoid(self.u) + self.beta]).T,
        #                                          torch.stack([s]))
        #         self.s = self.sigmoid(self.u - self.V_th)
        #         spk.append(self.s)
        #     self.reset()
        #     spk = torch.stack(spk)
        #     spk_batch.append(spk)

        # 将所有时间步骤的输出脉冲状态堆叠成一个张量，并作为函数的返回值。
        return torch.stack(spk)

    def reset(self):
        self.s = torch.zeros(self.output_shape, self.batch_size).to(device)
        self.u = torch.zeros(self.output_shape, self.batch_size).to(device)


if __name__ == '__main__':
    # 设置设备为CUDA
    device = 'cuda'
    # 创建一个模拟实例
    h_n = hybrid_neuron(input_shape=10,
                        output_shape=8,
                        T=5,
                        batch_size=2).float().to(device)
    # 创建一个输入特征张量
    input_feature = torch.tensor([[0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
                                  [0, 0, 0, 1, 0, 0, 1, 0, 1, 0]]).float().to(device)
    # 通过混合神经元模型进行前向传播，计算输出脉冲状态
    # 输出形状为(5, 2, 8)，表示5个时间步骤、2个样本、每个样本有8个输出脉冲状态
    output = h_n(input_feature)

    # 对输出进行求和，并进行反向传播计算梯度
    output.sum().backward()
    # time steps; batch_size; class nums
