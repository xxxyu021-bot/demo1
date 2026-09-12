# 项目结构  
├─config.json       
├─config.py         
├─dataset.py               
├─metrics.py        
├─model.py         
├─train.py          
├─utils.py 
└── README.md
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
| 实验 | 学习率 | Batch Size | Dropout | 最佳验证集 Accuracy | 测试集 Accuracy | recall | f1 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 第一组 | 2e-5 | 4 | 0.2 | 80.9% | 80.55% | 76.95% | 77.55% |
| 第二组 | 2e-5 | 8 | 0.2 | 82.6% | 83.08% | 81.64% | 82.22% |
| 第三组 | 2e-5 | 8 | 0.3 | 82.9% | 80.92% | 77.83% | 79.63% |
| 第四组 | 1e-5 | 8 | 0.3 | 82.8% | 83.27% | 80.54% | 81.74% |
| 第五组 | 1e-5 | 16 | 0.3 | 82.7% | 83.46% | 81.33% | 81.95% |
# 实验结果说明
1.第一组在第3epoch后验证准确度开始下降，损失提高，出现明显的过拟合，最终测试准确度较低，第二组将batch调高后准确度提升，但过拟合依然比较明显
第一组：
<img width="2013" height="1074" alt="image" src="https://github.com/user-attachments/assets/9e6600f6-3539-4b8b-bcc1-56671274de39" />
第二组：
<img width="2043" height="1005" alt="image" src="https://github.com/user-attachments/assets/f78a1646-b6f6-4db9-ac7c-4d1415e137b8" />
2.根据前面的结果先选择batchsize为8降低震荡，提高dropout减缓过拟合，第三组过拟合现象有减缓，但准确度下降，recall和f1也比第二组更低
<img width="2055" height="1020" alt="image" src="https://github.com/user-attachments/assets/63df3f76-9018-4313-aff3-e5c41a0cb0bd" />
3.第四组尝试调整学习率，结果准确度提升，说明降低学习率提高训练稳定性在正则化提高后提高了准确度
<img width="2067" height="1014" alt="image" src="https://github.com/user-attachments/assets/a14a968c-a74b-4041-85e3-661fec7e59c2" />
4.第五组提高batchsize后准确度依然达到83%，在较低学习率下适当提高batch大小让训练更平稳可以提高模型泛化能力
<img width="2019" height="1041" alt="image" src="https://github.com/user-attachments/assets/f321becc-a755-414e-8040-46a4066ad228" />
# 实验结果
新增recall和f1可视化
<img width="762" height="594" alt="image" src="https://github.com/user-attachments/assets/6cca7e28-cbbe-49d1-a4ea-8c71a910fc57" />
<img width="2004" height="1013" alt="image" src="https://github.com/user-attachments/assets/cdb3468d-c209-4fee-a607-57b83bfa1fb1" />
<img width="2037" height="1029" alt="image" src="https://github.com/user-attachments/assets/f075fcf6-6a2b-4847-b50d-723eeeafbbb6" />
    "dropout_rate": 0.3,
    "batch_size": 16,
    "num_epochs": 6,
    "learning_rate": 1e-5,
    "num_classes": 15,
    "max_seq_len": 128,
    "weight_decay": 0.01
# 实验总结
 Batch Size较低可以提高更新次数但过低会导致稳定性降低，需要适当提升Batch Size提高稳定性，提高dropout可以减缓过拟合现象，但也有可能因为正则化提升导致泛化能力下降，学习率也会影响训练稳定性，过高会导致梯度较大过低可能导致收敛不充分，需要根据验证结果调整学习率和batch大小来提高泛化能力。










运行环境  
Python 3.13.13  
torch      2.12.0+cu132  
transformers 5.16.1  
swanlab      0.10.0  
