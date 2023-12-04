# 我的硕士毕设

这是我为硕士毕设创建的项目，虽然毕设的主要内容之前都已经完成了，但我希望借助这次机会，体验创建项目的整个过程。

我希望实现的目标包括：

- [ ] 使用逻辑清晰、通用性的的项目结构模板来管理脚本

- [ ] 使用yacs、ignite这样的深度学习高级管理工具
- [ ] 使用日志管理库来合理的记录日志

## 内容简介

本项目旨在实现根据轴承振动信号识别故障以及判断故障程度，识别故障的思路流程图如下

![flowchart](.\resource\img\flowchart.png)

详细思路见我的[论文](.\resource\doc\paper.docx)。之后我也会在这里补充项目简介。

## 模板介绍

尽管我保留了模板的README，但在实现中我会有自己的理解与调整，所以在这里写明。

```
├──  config
│    └── defaults.py  - here's the default config file.
|       这部分是通用的参数配置（不管换成什么网络和数据集，都有这些参数）。最终使用的是defaults.py中的类cfg
│
│
├──  configs  
│    └── train_mnist_softmax.yml  - here's the specific config file for specific model or dataset.
|       这部分是特殊的参数配置，只针对特定的网络和数据集。最终使用的是（？）
│ 
│
├──  data  
│    └── datasets  - here's the datasets folder that is responsible for all data handling.
|       该目录存放数据集，理想状态下，其中没有其他脚本。
│    └── transforms  - here's the data preprocess folder that is responsible for all data augmentation.
|       一个目的是给出transform函数的包，我认为可以化简，待议。
│    └── build.py  		   - here's the file to make dataloader.
│    └── collate_batch.py   - here's the file that is responsible for merges a list of samples to form a mini-batch.
│
│
├──  engine
│   ├── trainer.py     - this file contains the train loops.
│   └── inference.py   - this file contains the inference process.
│
│
├── layers              - this folder contains any customed layers of your project.
│   └── conv_layer.py
│
│
├── modeling            - this folder contains any model of your project.
│   └── example_model.py
│
│
├── solver             - this folder contains optimizer of your project.
│   └── build.py
│   └── lr_scheduler.py
│   
│ 
├──  tools                - here's the train/test model of your project.
│    └── train_net.py  - here's an example of train model that is responsible for the whole pipeline.
│ 
│ 
└── utils
│    ├── logger.py
│    └── any_other_utils_you_need
│ 
│ 
└── tests					- this foler contains unit test of your project.
     ├── test_data_sampler.py
```