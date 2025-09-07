# NIPT Optimal Timing & Fetal Abnormality Modeling (CUMCM 2025, Q-C)

> 基于多因素风险评估模型的 NIPT 最优时点选择与胎儿异常判定（国赛C题）。数据处理、建模与可重复实验代码。  
> 对应论文：NIPT 最优检测时点选择与异常判定的多因素模型与稳健性分析。

## 🧭 Project Structure
2025Modeling/

├─ Code_FromMATLAB/

├─ Data_Processing/

├─ Question_1/

├─ Question_2/

├─ Question_3/

└─ Question_4/


## ✨ What’s inside
- **数据预处理**：孕周格式标准化、缺失/异常处理、GC 质控；  
- **问题一**：Y 染色体浓度 ~ 孕周 + BMI，含分段非线性交互模型与诊断；  
- **问题二**：Monte Carlo + 风险函数，得到 **BMI 动态分组**与各组 **最佳检测时点**；  
- **问题三**：π(w,x)（逻辑回归）与 q(w,gc)（分箱 + 核平滑）融合的综合风险 R(w|x)；  
- **问题四**：女胎异常判定（VIF 特征筛选 + 组合采样 + 逻辑回归，ROC/AUC）。  


## 🖥️ Environment
- Python ≥ 3.9（兼容 3.9 写法）；建议 Conda
- 主要依赖：`imbalanced_learn==0.12.4 
joblib==1.5.2 
matplotlib==3.10.6
numpy==2.3.2
pandas==2.3.2
scikit_learn==1.7.1
scipy==1.16.1
statsmodels==0.14.5`

