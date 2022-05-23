# 导入库：
import torch
import torchvision
import torchvision.models
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

# 图像预处理方法：# 这种预处理的地方尽量别修改，修改意味着需要修改网络结构的参数
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(120),#将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪的图像为制定的大小；
                                                                  #该操作的含义在于：即使只是该物体的一部分，也认为这是该类物体；
                                 transforms.RandomHorizontalFlip(),#以给定的概率随机水平旋转给定的PIL的图像，默认为0.5；
                                 transforms.ToTensor(),            #将给定图像转为Tensor
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),# 归一化处理

    "val": transforms.Compose([transforms.Resize((120, 120)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

# 数据导入方法： 导入自己的数据，自己的数据放在跟代码相同的文件夹下新建一个data文件夹，data文件夹
# 里的新建一个train文件夹用于放置训练集的图片。同理新建一个val文件夹用于放置测试集的图片。
train_data = torchvision.datasets.CIFAR10(root = "../data" , train = True ,download = False,transform=data_transform["train"])
#train_data = torchvision.datasets.ImageFolder(root="./data", transform=data_transform["train"])
traindata = DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=0)

test_data = torchvision.datasets.CIFAR10(root = "../data" , train = False ,download = False,transform=data_transform["val"])
#test_data = torchvision.datasets.ImageFolder(root="../data", transform=data_transform["val"])
testdata = DataLoader(dataset=test_data, batch_size=32, shuffle=True,num_workers=0)  # windows系统下，num_workers设置为0，linux系统下可以设置多进程

train_size = len(train_data)  # 求出训练集的长度
test_size = len(test_data)  # 求出测试集的长度
print(train_size)  # 输出训练集的长度
print(test_size)  # 输出测试集的长度

# 设置调用GPU，如果有GPU就调用GPU，如果没有GPU则调用CPU训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))



from torch.hub import load_state_dict_from_url
model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',}#vgg16的模型地址

class VGG(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=False):  # 自己是几种就把这个数改成几，但是如果采用标准的VGG模型必须是1000,否则会不匹配
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))#二维自适应平均池化，输出大小为3*3的张量
        self.classifier = nn.Sequential(
            nn.Linear(512*3*3, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),#元素归零的概率。 默认值：0.5
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()  # 参数初始化

    def forward(self, x):
        # N x 3 x 224 x 224   3*120*120
        x = self.features(x)
        # N x 512 x 7 x 7      512*3*3
        x = self.avgpool(x)    #512*3*3
        x = torch.flatten(x, start_dim=1) #     x = torch.flatten(x, 1)
        # N x 512*7*7           512*3*3
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():  # 遍历各个层进行参数初始化
            if isinstance(m, nn.Conv2d):  # 判断类型函数isinstance() ，如果是卷积层的话 进行下方初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')   #kaiming正态分布初始化
                #nn.init.xavier_uniform_(m.weight)  # 均匀分布初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 如果偏置不是0 将偏置置成0  相当于对偏置进行初始化，
                                                  # 初始化参数使其为常值，即每个参数值都相同。一般是给网络中bias进行初始化。
            elif isinstance(m, nn.Linear):  # 如果是全连接层
                nn.init.xavier_uniform_(m.weight)  # 也进行均匀分布初始化
                # nn.init.normal_(m.weight, 0, 0.01)#torch.init.normal_(tensor,mean=,std=) ，
                # 给tensor初始化，一般是给网络中参数weight初始化，初始化参数值符合正态分布。mean:均值，std:正态分布的标准差
                nn.init.constant_(m.bias, 0)  # 将所有偏执置为0


def make_features(cfg: list,batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]#BatchNorm2d(v)对channel维度就行批归一化
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

#当预训练时，只取提取特征结构层的权重，但是分类结构需要在训练，当不是预训练时，不加载权重
def vgg(pretrained=False, progress=True, model_name="vgg16",  **kwargs):
    #断言，用于判断一个表达式，如果发生异常就说明表达式为假。就会触发异常。
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]
    model = VGG(make_features(cfg), **kwargs)
    if pretrained:#加载这个模型。
        state_dict = load_state_dict_from_url(model_urls['vgg16'], model_dir='./model_data',
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    #采用预训练时，自定义自己的分类结构
    # if num_classes!=1000:
    #     model.classifier =  nn.Sequential(
    #         nn.Linear(512 * 3 * 3, 4096),
    #         nn.ReLU(True),
    #         nn.Dropout(),
    #         nn.Linear(4096, 4096),
    #         nn.ReLU(True),
    #         nn.Dropout(),
    #         nn.Linear(4096, num_classes),)
    return model


VGGnet = vgg(num_classes=10, init_weights=True)  # 将模型命名，是几种就把这个改成几
VGGnet.to(device)
print(VGGnet.to(device))  # 输出模型结构


# 设置训练需要的参数，epoch，学习率learning 优化器。损失函数。
epoch = 1  # 这里是训练的轮数
learning = 0.0001  # 学习率
optimizer = torch.optim.Adam(VGGnet.parameters(), lr=learning)  # 优化器
loss = nn.CrossEntropyLoss()  # 损失函数

#设置四个空数组，用来存放训练集的loss和accuracy    测试集的loss和 accuracy
train_loss_all = []
train_accur_all = []
test_loss_all = []
test_accur_all = []


#开始训练：
for i in range(epoch):  # 开始迭代
    train_loss = 0  # 训练集的损失初始设为0
    train_num = 0.0  #
    train_accuracy = 0.0  # 训练集的准确率初始设为0
    VGGnet.train()  # 启用 BatchNormalization 和 Dropout。 在模型测试阶段使用model.train() 让model变成训练模式，此时 dropout和batch normalization的操作在训练时起到防止网络过拟合的问题。
    train_bar = tqdm(traindata)  # 用于进度条显示，没啥实际用处

    #每次迭代一小批数据，一组是32张图片,  len(train_bar)=1563
    for step, data in enumerate(train_bar):  # 开始迭代跑， enumerate这个函数不懂可以查查，将训练集分为 step是序号，data是数据
        #print(len(train_bar))  # len(train_bar)=1563
        img, target = data  # 将data 分为 img图片，和 target标签
        optimizer.zero_grad()  # 清空历史梯度
        #print(img.shape)       #img.shape = [32, 3, 120, 120])
        #print(target.shape)     #target.shape=32
        outputs = VGGnet(img.to(device))  # 将图片打入网络进行训练,outputs是输出的结果
        loss1 = loss(outputs, target.to(device))  # 计算神经网络输出的结果outputs与图片真实标签target的差别-这就是我们通常情况下称为的损失
        outputs = torch.argmax(outputs, 1)  # 会输出10个值，最大的值就是我们预测的结果 求最大值，
        loss1.backward()  # 神经网络反向传播
        optimizer.step()  # 梯度优化 用上面的adam优化0
        train_loss += abs(loss1.item()) * img.size(0) #将所有损失绝对值求和,当前的损失值乘上本批次大小，相当于本批次32张图片的损失值，直到遍历训练集
        accuracy = torch.sum(outputs == target.to(device))  # outputs == target的 即使预测正确的，统计当前32张图片预测正确的个数,从而计算准确率
        #print( torch.sum( outputs== outputs.to(device)))            #查看本批次训练时预测正确了多少张图片
        train_accuracy = train_accuracy + accuracy  # 求迭代完本轮数据的，总共的训练集的预测正确数
        train_num += img.size(0)  #循环一次为32张图片，共循环1563次，所以每迭代一轮加上32

    print("epoch：{} ， train-Loss：{} , train-accuracy：{}".format(i + 1, train_loss / train_num,  # 输出训练情况
                                                                train_accuracy / train_num))
    train_loss_all.append(train_loss / train_num)  # 将训练的损失放到一个列表里 方便后续画图
    train_accur_all.append(train_accuracy.double().item() / train_num)  # 训练集的准确率


    test_loss = 0  # 同上 测试损失
    test_accuracy = 0.0  # 测试准确率
    test_num = 0
    VGGnet.eval()  # 将模型调整为测试模型，不启用 BatchNormalization 和 Dropout。
    with torch.no_grad():  # 清空历史梯度，进行测试  与训练最大的区别是测试过程中取消了反向传播
        test_bar = tqdm(testdata)
        for data in test_bar:  #len(test_bar)=313   10000/3=313
            img, target = data                # 将data 分为 img图片，和 target标签
            outputs = VGGnet(img.to(device))  #将图片打入网络进行测试,outputs是输出的结果
            loss2 = loss(outputs, target.to(device))  # 计算神经网络输出的结果outputs与图片真实标签target的差别
            outputs = torch.argmax(outputs, 1)        # 会输出10个值，最大的值就是我们预测的结果 求最大值，
            test_loss = test_loss + abs(loss2.item()) * img.size(0)#将所有损失绝对值求和,相当于本批次32张图片的损失值，直到遍历测试集
            accuracy = torch.sum(outputs == target.to(device)) # 即预测正确的，统计当前32张图片预测正确的个数,从而计算准确率
            #print(accuracy)
            test_accuracy = test_accuracy + accuracy   # 求迭代完本轮数据的，总共的训练集的预测正确数
            test_num += img.size(0)                     #循环一次为32张图片，共循环1563次，所以每迭代一轮加上32

    print("test-Loss：{} , test-accuracy：{}".format(test_loss / test_num, test_accuracy / test_num))
    test_loss_all.append(test_loss / test_num) # 将测试的损失放到一个列表里 方便后续画图
    test_accur_all.append(test_accuracy.double().item() / test_num) # 测试集的准确率


# 下面的是画图过程，将上述存放的列表  画出来即可
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(epoch), train_loss_all,
         "ro-", label="Train loss")
plt.plot(range(epoch), test_loss_all,
         "bs-", label="test loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.subplot(1, 2, 2)
plt.plot(range(epoch), train_accur_all,
         "ro-", label="Train accur")
plt.plot(range(epoch), test_accur_all,
         "bs-", label="test accur")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend()
plt.show()

torch.save(VGGnet, "VGG.pth")
print("模型已保存")

