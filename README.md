# 项目结构  
```
├── config/
│   └── config.json       
├── Remote/                
├── swanlog/              
├── config.py             
├── dataset.py            
├── main.py                 
├── metrics.py          
├── model.py                
├── train.py               
└── utils.py

# 数据集信息
训练集：3000条
验证集：1000条
测试集：1064条
| 类别ID | 类别名称 |
| ---- | ---- |
| 100 | 新闻故事 |
| 101 | 新闻文化 |
| 102 | 新闻娱乐 |
| 103 | 新闻体育 |
| 104 | 新闻财经 |
| 106 | 新闻房产 |
| 107 | 新闻汽车 |
| 108 | 新闻教育 |
| 109 | 新闻科技 |
| 110 | 新闻军事 |
| 112 | 新闻旅游 |
| 113 | 新闻国际 |
| 114 | 股票 |
| 115 | 新闻农业 |
| 116 | 新闻游戏 |

# 超参数记录
| 实验 | 学习率 | Batch Size | Dropout | 最佳f1| 测试集 Accuracy | recall | f1 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 第一组 | 2e-5 | 4 | 0.2 | 82.35% | 81.86% | 80.84% | 81.41% |
| 第二组 | 2e-5 | 8 | 0.2 | 82.48% | 84.02% | 82.26% | 82.98% |
| 第三组 | 2e-5 | 8 | 0.3 | 82.02% | 84.02% | 82.34% | 83.35% |
| 第四组 | 1e-5 | 8 | 0.3 | 81.34% | 83.55% | 80.97% | 82.04% |
| 第五组 | 1e-5 | 16 | 0.3 | 79.91% | 83.46% | 80.04% | 81.43% |
# 实验结果说明
新增 linear warmup和linear decay后有效缓解过拟合，acc，recall和f1相比无线性预热和衰减时都有提升，第三组的对比情况如下图所示：
新增 linear warmup和linear decay后
<img width="2028" height="1002" alt="image" src="https://github.com/user-attachments/assets/514e959b-4b2d-4df9-b243-ff865c4ec4eb" />
新增 linear warmup和linear decay前
<img width="2037" height="1020" alt="image" src="https://github.com/user-attachments/assets/3bb8b743-8f6e-45cd-b9fe-72d74f12ffed" />
acc，recall，f1验证结果整体在新增 linear warmup和linear decay后有所提高，峰值也更高

# 实验结果
最佳情况下测试结果acc为84.02%达到83%的预计结果，同时recall为82.34%，f1为83.35%。
<img width="1035" height="594" alt="image" src="https://github.com/user-attachments/assets/b9f2a7a0-59c2-44f6-ad2b-015b7af1c7d8" />
<img width="2013" height="1020" alt="image" src="https://github.com/user-attachments/assets/b1fb7bf5-8b1e-4892-bd79-be0ec145da22" />
<img width="2028" height="1023" alt="image" src="https://github.com/user-attachments/assets/d5d13d2f-f127-4973-b10e-ce66067007c4" />


    "dropout_rate": 0.3,
    "batch_size": 8,
    "num_epochs": 6,
    "learning_rate": 2e-5,
    "num_classes": 15,
    "max_seq_len": 128,
    "weight_decay": 0.01
    "seed": 42,
    "warmup_ratio": 0.1,
    "early_stopping_patience": 3
# 实验总结
 Batch Size较低可以提高更新次数但过低会导致稳定性降低，需要适当提升Batch Size提高稳定性，提高dropout可以减缓过拟合现象，但也有可能因为正则化提升导致泛化能力下降，学习率也会影响训练稳定性，过高会导致梯度较大过低可能导致收敛不充分，需要根据验证结果调整学习率和batch大小来提高泛化能力。添加随机种子后确保实验结果可以复现，有利于对照不同参数的结果，线性预热有利于训练过程中梯度稳定，预热完后开始衰减，减少震荡有利于让收敛效果更好。










运行环境  
Python 3.13.13  
torch      2.12.0+cu132  
transformers 5.16.1  
swanlab      0.10.0  
