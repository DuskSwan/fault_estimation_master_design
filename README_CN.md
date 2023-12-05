# 我的硕士毕设

这是我为硕士毕设创建的项目，虽然毕设的主要内容之前都已经完成了，但我希望借助这次机会，体验创建项目的整个过程。

我希望实现的目标包括：

- [ ] 使用逻辑清晰、通用性的的项目结构模板来管理脚本
- [ ] 使用yacs、ignite这样的深度学习高级管理工具
- [ ] 使用日志管理库loguru来合理的记录日志

## 内容简介

本项目旨在实现根据轴承振动信号识别故障以及判断故障程度，识别故障的思路流程图如下

![flowchart](.\resource\img\flowchart.png)

详细思路见我的[论文](.\resource\doc\paper.docx)。之后我也会在这里补充项目简介。

## 模板介绍

尽管我保留了模板的README，但在实现中我会有自己的理解与调整，所以在这里写明。

原本的模板结构如下

```text
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
|       这两个文件分别返回dataloader和mini-batch，我认为放到一个文件里就好了。
│
│
├──  engine
│   ├── trainer.py     - this file contains the train loops.
│   └── inference.py   - this file contains the inference process.
|       我不太懂inference是什么样的过程，查了一下发现有两种解释，一种就是指对训练好的网络进行测试，含义约等于prediction，可以看成深度学习版本的prediction；另一种是“推理”，指研究输入如何影响结果，通常用于生成模型，其背景是在统计中用样本X和目标y去infer参数θ。总的来看，这里的含义应该就是测试/预测。这两个脚本并非训练与测试的主文件，只是提供函数。
│
│
├── layers              - this folder contains any customed layers of your project.
│   └── conv_layer.py
│
│
├── modeling            - this folder contains any model of your project.
│   └── example_model.py
|       这两个文件分别定制模型和层，我认为也应该放到一个文件里。
│
├── solver             - this folder contains optimizer of your project.
│   └── build.py
│   └── lr_scheduler.py
|       定制求解器，包括优化器、学习率变化器等。
│ 
├──  tools                - here's the train/test model of your project.
│    └── train_net.py  - here's an example of train model that is responsible for the whole pipeline.
|       这个工具是项目层面的，实际训练和测试网络的脚本写在这里。但我在想，这部分好像也可以直接挪去主目录下。或者为了容易看懂改名为start。
│ 
└── utils
│    ├── logger.py
│    └── any_other_utils_you_need
|       算法层面的实用工具，里面装了很多过程中需要的函数。
│ 
└── tests					- this foler contains unit test of your project.
     ├── test_data_sampler.py
     |   测试其他文件的debug文件。可以用于流程单元的测试。
```