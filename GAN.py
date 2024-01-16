import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
import os

# 设置随机种子
torch.manual_seed(33)

# 定义生成器模型
class Generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=784):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# 定义判别器模型
class Discriminator(nn.Module):
    def __init__(self, input_dim=784):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 定义训练函数
def train(generator, discriminator, dataloader, datetime_str, num_epochs=100, sample_interval=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    # 定义损失函数和优化器
    loss_fun = nn.BCELoss()  # 二分类的交叉熵
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_loss_list = []
    g_loss_list = []
    dis_real_list = []
    dis_fake_list = []
    for epoch in range(num_epochs):
        d_loss_final = 0
        g_loss_final = 0
        dis_real_final = 0
        dis_fake_final = 0
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.view(-1, 784).to(device)
            batch_size = real_images.shape[0]
            # 创建标签，区分真实图像和生成图像
            real = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)

            # 训练判别器
            optimizer_D.zero_grad()

            # 生成一批噪声样本
            z = torch.randn(batch_size, 100).to(device)
            generated_images = generator(z).detach()

            # 判别器对真实图像的损失
            dis_real = discriminator(real_images)
            dis_real_final += float(dis_real.mean())
            real_loss = loss_fun(dis_real, real)

            # 判别器对生成图像的损失
            dis_fake = discriminator(generated_images)
            dis_fake_final += float(dis_fake.mean())
            fake_loss = loss_fun(dis_fake, fake)

            # 总体判别器损失
            d_loss = (real_loss + fake_loss) / 2
            d_loss_final = d_loss.item()

            d_loss.backward()
            optimizer_D.step()

            # 训练生成器
            optimizer_G.zero_grad()

            # 生成一批新的噪声样本
            z = torch.randn(batch_size, 100).to(device)
            generated_images = generator(z)

            # 生成器的损失
            g_loss = loss_fun(discriminator(generated_images), real)
            g_loss_final = g_loss.item()

            g_loss.backward()
            optimizer_G.step()
        print(f"Epoch [{epoch}/{num_epochs}] Discriminator Loss: {d_loss_final:.4f} Generator Loss: {g_loss_final:.4f}")
        d_loss_list.append(d_loss_final)
        g_loss_list.append(g_loss_final)
        dis_real_final /= len(dataloader)
        dis_fake_final /= len(dataloader)
        dis_real_list.append(dis_real_final)
        dis_fake_list.append(dis_fake_final)

        # 显示生成的图像
        if epoch % sample_interval == 0 or epoch == num_epochs - 1:
            generate_samples(generator, epoch, datetime_str)
    plt.figure('loss')
    plt.plot(d_loss_list, label='discriminator loss')
    plt.plot(g_loss_list, label='generator loss')
    plt.legend()
    plt.figure('discriminate')
    plt.plot(dis_real_list, label='real pic')
    plt.plot(dis_fake_list, label='fake_pic')
    plt.legend()
    plt.show()
    torch.save(generator.state_dict(), 'generator.pth')

# 生成样本
def generate_samples(generator, epoch, datetime_str, num_samples=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    with torch.no_grad():
        z = torch.randn(num_samples, 100).to(device)
        generated_images = generator(z).cpu().view(-1, 28, 28)
        fig, axs = plt.subplots(5, 5)
        count = 0
        for i in range(5):
            for j in range(5):
                axs[i, j].imshow(generated_images[count], cmap='gray')
                axs[i, j].axis('off')
                count += 1
        plt.savefig(f"./{datetime_str}/generated_images_epoch_{epoch}.png")
        plt.close()

if __name__ == '__main__':
    # 加载MNIST数据集
    transform = transforms.Compose([  # 数据预处理
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)  # 如果没有MNIST数据集，则下载
    dataloader = DataLoader(dataset=mnist_dataset, batch_size=128, shuffle=True)

    # 创建生成器和判别器实例
    generator = Generator()
    discriminator = Discriminator()
    # 当前时间
    current_datetime = datetime.now()
    datetime_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    os.mkdir(datetime_str)

    mode = 'train'
    # mode = 'generate'
    # 训练GAN模型
    if mode == 'train':
        train(generator, discriminator, dataloader, datetime_str, num_epochs=100, sample_interval=10)
    elif mode == 'generate':
        generator.load_state_dict(torch.load('generator.pth'))
        generate_samples(generator, 0, datetime_str)
