# -*- coding: utf-8 -*-
'''
Example Bayesian Network
Author: Dr QIU Lu
Commented & modified by: LUO Chensheng 
Time: 25 May 2025
'''
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 创建贝叶斯网络
model = DiscreteBayesianNetwork([
    ('A', 'B'),
    ('A', 'C'),
    ('B', 'C'),
    ('E', 'D'),
    ('C', 'D')
])

# 定义各个节点的条件概率分布
cpd_a = TabularCPD(variable='A', variable_card=2, 
                   values=[[0.3], [0.7]], 
                   state_names={'A': ['a', '~a']})

cpd_b = TabularCPD(variable='B', variable_card=2, 
                   evidence=['A'], evidence_card=[2], 
                   values=[[0.1, 0.6], [0.9, 0.4]], 
                   state_names={'A': ['a', '~a'], 
                                'B': ['b', '~b']})

cpd_e = TabularCPD(variable='E', variable_card=2, 
                   values=[[0.35], [0.65]], 
                   state_names={'E': ['e', '~e']})

cpd_c = TabularCPD(variable='C', variable_card=2, 
                   evidence=['A', 'B'], evidence_card=[2, 2], 
                   values=[
                       [0.05, 0.5, 0.45, 0.6], 
                       [0.95, 0.5, 0.55, 0.4]
                   ], 
                   state_names={'A': ['a', '~a'], 'B': ['b', '~b'], 
                                'E': ['e', '~e'], 'C': ['c', '~c']})

cpd_d = TabularCPD(variable='D', variable_card=2, 
                   evidence=['C', 'E'], evidence_card=[2, 2], 
                   values=[
                       [0.01, 0.5, 0.75, 0.31], 
                       [0.99, 0.5, 0.25, 0.69]
                   ], 
                   state_names={'C': ['c', '~c'], 
                                'E': ['e', '~e'], 'D': ['d', '~d']})

# 将条件概率分布添加到模型中
model.add_cpds(cpd_a, cpd_b, cpd_e, cpd_c, cpd_d)

# 验证模型是否正确
assert model.check_model()


# 创建推理引擎
inference = VariableElimination(model)

# 进行概率查询
# 示例：计算在A=a，B=b的情况下，D的概率
result = inference.query(variables=['D','C'], evidence={'A': 'a', 'B': 'b', 'E': 'e'})
print(result)


# 计算联合概率 P(a, ¬b, c, ¬d, e) 的另一种方法
# 由于不能使用VariableElimination同时指定所有变量作为查询和证据，我们可以直接计算乘积
prob_a = cpd_a.values[0]  # P(a)
prob_b_given_a = cpd_b.values[1][0]  # P(¬b | a)
prob_e = cpd_e.values[0]  # P(e)
prob_c_given_a_b = cpd_c.values[0][0][1]  # P(c | a, ¬b) - 需要根据cpd_c的结构确定索引
prob_d_given_c_e = cpd_d.values[1][0][0]  # P(¬d | c, e) - 需要根据cpd_d的结构确定索引

# 计算联合概率
joint_prob = prob_a * prob_b_given_a * prob_e * prob_c_given_a_b * prob_d_given_c_e

print(f"P(a, ¬b, c, ¬d, e) = {joint_prob}")