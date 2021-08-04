import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        self._conv1 = nn.Conv2d(3, 16, 5)
        self._pool1 = nn.MaxPool2d(2, 2)
        self._conv2 = nn.Conv2d(16, 32, 5)
        self._pool2 = nn.MaxPool2d(2, 2)
        self._fc1 = nn.Linear(32*5*5, 120)
        self._fc2 = nn.Linear(120, 84)
        self._fc3 = nn.Linear(84, self.num_classes)

    def forward(self, x):
        x = F.relu(self._conv1(x))   # in: (3, 32, 32)        out: (16, 28, 28)
        x = self._pool1(x)           # in: (16, 28, 28)       out: (16, 14, 14)
        x = F.relu(self._conv2(x))   # in: (16, 14, 14)       out: (32, 10, 10)
        x = self._pool2(x)           # in: (32, 10, 10)       out: (32, 5, 5)
        x = x.view(-1, 32*5*5)       # 拉平成一个vector, 如果其中一个维度是-1则表示改为度由其他维度推断得来
        x = F.relu(self._fc1(x))     # in: 32*5*5             out: 120
        x = F.relu(self._fc2(x))     # in: 120                out: 84
        # 在最后一层没有用softmax是因为pytorch的激活函数nn.CrossEntropyLoss中集成了softmax和NLLLoss
        x = self._fc3(x)             # in: 84                 out: num_classes                                
        return x         

    # def forward(self, x):
    #     print("conv1 input_size: " + str(str(x.shape)))
    #     x = F.relu(self._conv1(x))   # in: (3, 32, 32)        out: (16, 28, 28)
    #     print("covv1 output_size: " + str(x.shape) + "\n")
    #     print("pool1 input_size: " + str(x.shape))
    #     x = self._pool1(x)           # in: (16, 28, 28)       out: (16, 14, 14)
    #     print("pool1 output_size: " + str(x.shape) + "\n")
    #     print("conv2 input_size: " + str(x.shape))
    #     x = F.relu(self._conv2(x))   # in: (16, 14, 14)       out: (32, 10, 10)
    #     print("covv2 output_size: " + str(x.shape) + "\n")
    #     print("cpool2 input_size: " + str(x.shape))
    #     x = self._pool2(x)           # in: (32, 10, 10)       out: (32, 5, 5)
    #     print("pool2 output_size: " + str(x.shape) + "\n")
    #     print("before_view input_size: " + str(x.shape))
    #     x = x.view(-1, 32*5*5)       # 拉平成一个vector, 如果其中一个维度是-1则表示改为度由其他维度推断得来
    #     print("after_view output_size: " + str(x.shape) + "\n")
    #     print("fc1 input_size: " + str(x.shape))
    #     x = F.relu(self._fc1(x))     # in: 32*5*5             out: 120
    #     print("fc1 output_size: " + str(x.shape) + "\n")
    #     print("fc2 input_size: " + str(x.shape))
    #     x = F.relu(self._fc2(x))     # in: 120                out: 84
    #     print("fc2 output_size: " + str(x.shape) + "\n")
    #     print("fc3 input_size: " + str(x.shape))
    #     # 在最后一层没有用softmax是因为pytorch的激活函数nn.CrossEntropyLoss中集成了softmax和NLLLoss
    #     x = self._fc3(x)             # in: 84                 out: num_classes   
    #     print("fc3 output_size: " + str(x.shape) + "\n")                             
    #     return x        


if __name__ == "__main__":
    import torch
    from torchsummary import summary

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = LeNet(5).to(device)

    # input = torch.zeros((1, 3, 32, 32))
    # output = net(input)
    summary(net, input_size=(3, 32, 32), batch_size=16)

    print(net.dict)