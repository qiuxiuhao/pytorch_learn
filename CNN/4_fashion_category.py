import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader

#数据来源：https://www.kaggle.com/datasets/zalando-research/fashionmnist
fashion_train = pd.read_csv('./data/fashion-mnist_train.csv')
fashion_test = pd.read_csv('./data/fashion-mnist_test.csv')

#����������ȡX��Y��ת����������ʽ
x_train = fashion_train.iloc[:,1:].values
x_train = torch.tensor(x_train,dtype=torch.float32).reshape(-1,1,28,28)
y_train = fashion_train.iloc[:,0].values
y_train = torch.tensor(y_train,dtype=torch.int64)
x_test = fashion_test.iloc[:,1:].values
x_test = torch.tensor(x_test,dtype=torch.float32).reshape(-1,1,28,28)
y_test = fashion_test.iloc[:,0].values
y_test = torch.tensor(y_test,dtype=torch.int64)

#输出其中一张图片及其标签
#plt.imshow(x_train[12345,0,:,:],cmap='gray')
#plt.show()
#print('标签值',y_train[12345])

#构建数据集
train_dataset =  TensorDataset(x_train,y_train)
test_dataset = TensorDataset(x_test,y_test)

#创建模型
model = nn.Sequential(
    nn.Conv2d(1,6,5,stride=1,padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(2,stride=2),

    nn.Conv2d(6,16,5,stride=1,padding=0),
    nn.Sigmoid(),
    nn.AvgPool2d(2,stride=2),

    nn.Flatten(),

    nn.Linear(16*5*5,120),
    nn.Sigmoid(),

    nn.Linear(120,84),
    nn.Sigmoid(),

    nn.Linear(84,10)
)

"""
x = torch.randn(size=(1,1,28,28),dtype=torch.float)

for layer in model:
    x = layer(x)
    print(layer.__class__.__name__,'output shape:\t',x.shape)
"""

#3.模型训练和测试

def train_test(model,train_dataset,test_dataset,learning_rate,
               num_epochs,batch_size,device):
    #定义一个参数初始化函数
    def init_weights(layer):
        if isinstance(layer,nn.Linear) or isinstance(layer,nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
    #初始化相关操作
    model.apply(init_weights)
    model.to(device)
    #定义损失函数
    loss = nn.CrossEntropyLoss()
    #定义优化器
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

    #迭代训练过程
    for epoch in range(num_epochs):
        model.train()
        #创建数据加载器
        train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
        train_loss = 0.0
        train_correct_num = 0
        #按小批量循环迭代
        for batch_idx, (x,y) in enumerate(train_loader):
            x,y = x.to(device),y.to(device)
            #前向计算
            y_pred = model(x)
            #计算损失
            loss_value = loss(y_pred,y)
            #反向传播
            loss_value.backward()
            #更新参数
            optimizer.step()
            #梯度清零
            optimizer.zero_grad()
            #统计训练损失和准确率
            train_loss += loss_value.item()*y.shape[0]
            _,y_pred_label = torch.max(y_pred,dim=1)
            train_correct_num += (y_pred_label == y).sum().item()

            #打印进度条
            print(f"\rEpoch:{epoch+1:0>2}[{'='*int((batch_idx+1)/len(train_loader)*50)}]",end='')

        #计算训练集的平均损失和准确率
        train_loss /= len(train_dataset)
        train_acc = train_correct_num / len(train_dataset)

        #在测试集上评估模型
        model.eval()
        test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
        test_correct_num = 0
        #迭代测试集
        with torch.no_grad():
            for x,y in test_loader:
                x,y = x.to(device),y.to(device)
                y_pred = model(x)
                _,y_pred_label = torch.max(y_pred,dim=1)
                test_correct_num += (y_pred_label == y).sum().item()
            #计算测试集准确率
            test_acc = test_correct_num / len(test_dataset)
            #打印结果
            print(f" Train Loss:{train_loss:.4f} Train Acc:{train_acc:.4f} Test Acc:{test_acc:.4f}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#定义超参数
learning_rate = 0.9
num_epochs = 20
batch_size = 256

#训练和测试模型
train_test(model,train_dataset,test_dataset,learning_rate,
           num_epochs,batch_size,device)

#选取一个数据进行测试对比
plt.imshow(x_test[666,0,:,:],cmap='gray')
plt.show()
print('真实标签值:',y_test[666].item())
#预测标签值:
output = model(x_test[666].unsqueeze(0).to(device))
_,pred_label = torch.max(output,dim=1)

print('预测标签值:',pred_label.item())
