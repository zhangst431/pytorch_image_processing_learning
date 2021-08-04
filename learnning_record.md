# 学习笔记

## 加速github的clone push
    镜像站：https://github.com.cnpmjs.org/

            https://hub.fastgit.org/

            https://github.wuyanzheshui.workers.dev

## 关注的两个B站up主
* [霹雳吧啦Wz](https://space.bilibili.com/18161609/video), [迪哥有点愁](https://space.bilibili.com/252075192/)

## nn.Conv2D()
`torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)`
* 前面几个参数都比较简单，只记录一下两个不常见的
    * dilation: 空洞卷积
    * groups: 分组卷积（可以减少参数量，目前我还没用到，用到再补充）
    * 详见[这个博客](https://blog.csdn.net/qq_34243930/article/details/107231539)


## nn.MaxPool2D()
`torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)`
* 常用：卷积核尺寸， 步长， padding大小
    * dilation：~~暂时不会，日后补充~~
    * return_indices: 返回最大值出的索引，在MaxUnPool2D操作时会用到
    * ceil_mod: 计算输出shape的时候是向上取整还是向下取整，默认向下取整
    * 详见[这个博客](https://blog.csdn.net/weixin_38481963/article/details/109962715)


## nn.Linear()
`torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)`
* 输入神经元个数，输出神经元个数，偏移，device，dtype


## torch.Tensor.view(*shape)
* 将tensor变成其他shape但里面原有的数据不会发生改变,如果其中有一个维度是-1，则表示该维度由其他维度的信息推断得来
* 详见[这篇博客](https://blog.csdn.net/qimo601/article/details/112139783)


## torch.zeros()
`torch.zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor`
* 详见[这篇博客](https://blog.csdn.net/qq_32806793/article/details/102951466)


## 模型可视化
* [torchsummary方法](https://www.jianshu.com/p/4b7d45bf2eec)
* [其他方法](https://www.jianshu.com/p/43f2893c3ab8)


## torchvision.transforms
* 详见[这篇博客](https://www.jianshu.com/p/1ae863c1e66d)
* 各种数据增强方法详见[这篇博客](https://blog.csdn.net/u011995719/article/details/85107009)


## torch.utils.data.DataLoader()
`DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)`

* ~~过几天学到这里再补充~~
* 可参考下面几篇博客：[pytorch :: Dataloader中的迭代器和生成器应用](https://blog.csdn.net/jx69693678nab/article/details/103819766?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_title~default-0.pc_relevant_baidujshouduan&spm=1001.2101.3001.4242), 
[pytorch-DataLoader(数据迭代器)](https://blog.csdn.net/weixin_42468475/article/details/108714940), 
[聊聊Pytorch中的dataloader---知乎](https://zhuanlan.zhihu.com/p/117270644), 
* 这是一篇比较高质量的博客[系统学习Pytorch笔记三：Pytorch数据读取机制(DataLoader)与图像预处理模块(transforms)](https://blog.csdn.net/wuzhongqiang/article/details/105499476), 这是那篇博客的作者[Miracle8070](https://blog.csdn.net/wuzhongqiang)


## 