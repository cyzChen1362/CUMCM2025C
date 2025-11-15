# NIPT Optimal Timing & Fetal Abnormality Modeling (CUMCM 2025, Q-C)

> **基于多因素风险评估模型的 NIPT 最优时点选择与胎儿异常判定**  
> 本仓库为 **2025 全国大学生数学建模竞赛 C 题·国家一等奖** 论文的复现与展示工程，  
> 主要通过 **Jupyter Notebook** 的方式完整呈现数据处理、建模流程与可重复实验。

本工程对应论文：
**《NIPT 最优检测时点选择与异常判定的多因素模型与稳健性分析》**


---

## 📘 Notebook-Based Usage Guide

本仓库主要通过 Notebook 展示建模全过程，阅读路线如下：

### **1）数据预处理与分析（施工中）**

### 路径：`NoteBook/Data_Preprocessing/Data_Preprocessing.ipynb`  
当前进度：完成孕周单位转换、缺失值处理、异常值处理、数据清洗；  
施工中：女胎数据工程、数据初探。

### **2）问题一（挖坑中）**  

### 路径：`NoteBook/Question_1/`（待填）
研究 **Y 染色体浓度 ~ 孕周 + BMI** 的分段非线性交互模型；  

### **3）问题二（挖坑中）**  

### 路径：`NoteBook/Question_2/`（待填）
基于 Monte Carlo + 风险函数，构建 **BMI 分层的最佳检测时点模型**；  

### **4）问题三（挖坑中）**  

### 路径：`NoteBook/Question_3/`（待填）
π(w,x) + q(w,gc) 融合的综合风险 R(w|x) 建模；  

### **5）问题四（挖坑中）**  

### 路径：`NoteBook/Question_4/`（待填）
女胎异常判定（VIF、采样策略、逻辑回归、ROC/AUC）；  

---

## 👥 Contribution Breakdown（论文中工作量说明）

### **🟦 Simon-Tisa（建模手、编程手、论文手）**

* 完全承担 **问题一** 的建模、实验及写作；
* 与 CYZ Chan 合作完成 **问题二、问题四** 的数学建模；
* 与 CYZ Chan 合作完成 **问题二** 的代码及实验；
* 提供部分论文插图；
* 参与论文后期完善与结果呈现。

### **🟩 CYZ Chan（编程手、建模手）**

* 完全承担 **问题三** 的数学建模；
* 完全承担 **问题三、问题四** 的全部代码与实验；
* 与 Simon-Tisa 合作完成 **问题二、问题四** 的数学建模；
* 与 Simon-Tisa 合作完成 **问题二** 的代码与实验；
* 提供部分论文插图；
* 参与论文后期实验结果的修正与完善。

### **🟧 Lsuuu_econ（论文手）**

* 完全承担 **问题二、三、四** 的论文写作；
* 提供部分论文插图；
* 参与论文后期的润色整理与展示。

> ⚙️ 本工程由 **CYZ Chan** 建立并持续维护。

---

## ✨ What’s inside

* **数据预处理**：孕周格式标准化、缺失/异常处理、GC 质控；
* **问题一**：Y 染色体浓度 ~ 孕周 + BMI，含分段非线性交互模型与诊断；
* **问题二**：Monte Carlo + 风险函数，得到 **BMI 动态分组**与各组 **最佳检测时点**；
* **问题三**：π(w,x)（逻辑回归）与 q(w,gc)（分箱 + 核平滑）融合的综合风险 R(w|x)；
* **问题四**：女胎异常判定（VIF 特征筛选 + 组合采样 + 逻辑回归，ROC/AUC）。

---

## 🧭 Project Structure

```
2025Modeling/  
├─ Original_Code/  
│   ├─ Code_FromMATLAB/  
│   ├─ Data_Processing/  
│   ├─ Question_1/  
│   ├─ Question_2/  
│   ├─ Question_3/  
│   └─ Question_4/  
├─ NoteBook/   
│   ├─ Data_Preprocessing/  
│   ├─ Question_1/  
│   ├─ Question_2/  
│   ├─ Question_3/  
│   └─ Question_4/  
│  
├─ requirements.txt  
└─ README.md  
```

---

## 🖥️ Environment

* Python ≥ 3.9（兼容 3.9 写法）
* Conda 推荐
* 主要依赖：

```
imbalanced_learn==0.12.4
joblib==1.5.2
matplotlib==3.10.6
numpy==2.3.2
pandas==2.3.2
scikit_learn==1.7.1
scipy==1.16.1
statsmodels==0.14.5
```

---

## 🚧 当前施工进度

* **数据预处理与分析**：已完成 *孕周单位转换、缺失值处理*；施工中 *异常值处理、数据清洗、女胎数据工程、初步分析*。
* **问题一、二、三、四**：挖坑待填……

---

欢迎在 Issue 中交流复现过程、数据探讨、模型改进思路！
