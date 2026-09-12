# 项目结构  
├─config.json       
├─config.py         
├─dataset.py               
├─metrics.py        
├─model.py         
├─train.py          
└─utils.py          
# 超参数分析
| 实验 | 学习率 | Batch Size | Dropout | 最佳验证集 Accuracy | 测试集 Accuracy |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Baseline | 2e-5 | 8 | 0.2 | 82.7% | 83.18% |
| lr-1 | 1e-5 | 8 | 0.2 | 83.3% | 83.27% |
| lr-3 | 3e-5 | 8 | 0.2 | 83% | 83.27% |
| Batch-4 | 2e-5 | 4 | 0.2 | 82.5% | 83.74% |
| Batch-16 | 2e-5 | 16 | 0.2 | 82.5% | 83.46% |
| Dropout-01 | 2e-5 | 8 | 0.1 | 83% | 84.3% |
| Dropout-03 | 2e-5 | 8 | 0.3 | 83.1% | 83.65% |





运行环境  
Python 3.13.13  
torch      2.12.0+cu132  
transformers 5.16.1  
swanlab      0.10.0  
