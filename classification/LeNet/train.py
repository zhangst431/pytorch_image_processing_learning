import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # 50000张训练图片
    train_set = torchvision.datasets.CIFAR10(root='../data/CIFAR10/', train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.Dataloader(train_set, batch_size=36,
                                               shuffle=True, num_workers=3)
    
    # 10000张验证图片
    val_set = torchvision.datasets.CIFAR10(root='../data/CIFAR10/', train=True,
                                             download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=10000,
                                             shuffle=False, num_workers=3)

    # TODO：学习dataloader的访问方式，下面这两行代码现在还不能完全了解
    val_data_iter = iter(val_loader)
    val_image, val_label = val_data_iter.next()

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = LeNet()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        epoch_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            images, labels = data

            optimizer.zero_grad()
            preds = model(images)
            loss = loss_function(preds, images)
            loss.backward()
            optimizer.step()

            # TODO:了解一下loss的组成
            epoch_loss += loss.item()
            if step % 500 == 499:      #每500个mini-batch
                with torch.no_grad():
                    preds = model(val_image)
                    pred_y = torch.max(preds, dim=1)[1]
                    accuracy = torch.eq(pred_y, val_label).sum().item() / val_label.size(0)

                    print('[%d, %d] train_loss: %.3f test_accuracy: %.3f' % 
                          (epoch + 1, step + 1, epoch_loss / 500, accuracy))
                    epoch_loss = 0.0
    
    print("Training Finished!")

    save_path = './Lenet.pth'
    torch.save(model.state_dict(), save_path)





if __name__ == "__main__":
    main()