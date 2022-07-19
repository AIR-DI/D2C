# AI控制优化算法引擎参数说明

## MORE 模型参数说明
>    https://arxiv.org/abs/2102.11492  
>    **M**odel-based **O**ffline RL with **R**estrictive **E**xploration (**MORE**)  
> 算法简介：是一种基于模型的离线强化学习算法；适用于期望获得最大化长期利益的控制问题；可适用于较大规模的控制问题（控制变量数量可达20~30），使用收集的离线数据进行模型的训练（需基于离线数据进行系统动态模型的训练），训练完成后得到系统的优化控制策略；应用场景举例：火电燃烧优化控制。

| 参数名 | 含义 | 取值类型 | 取值范围 | 默认值 | 说明 |
| - | - | - | - | - | - |
| `n_action_samples` | Q网络计算loss时采样action个数 | int | 1~10 | 1 | 若取值大于1，则在Q网络更新时，通过采样多个action并最大化Q值来计算目标 |
| `sim_ratio` | 模拟样本比例 | float | 0~1 | 0.1 | 强化模型训练过程中，构建真实样本与模拟样本的混合训练样本时，模拟样本所占混合样本数量的比例 |
| `rollout_length` | 模拟轨迹长度 | int | 1~10 | 1 | 进行模拟样本的采样时，模拟操作轨迹的最大长度 |
| `automatic_elbo` | 是否自动计算elbo阈值 | bool | True,False | True | ELBO: the evidence lower bound objective; 利用ELBO值来近似数据分布的概率密度，该算法通过ELBO值来对模拟样本进行划分 |
| `elbo_percentile` | 计算elbo阈值百分位数 | int | 0~100 | 70 | 计算全部历史数据的ELBO值，并根据给定的百分位数来确定ELBO阈值 |
| `elbo_threshold` | elbo阈值 | float | -inf~inf | -0.4 | 若自动计算elbo阈值，则不用填写该值 |
| `penalty_alpha` | 奖励惩罚系数 | float | 0~10 | 1 | 对模拟样本的奖励进行惩罚的系数 |
| `penalty_func` | 奖励惩罚函数 | str | diff,other | diff | 对模拟样本的奖励进行惩罚时，选择的惩罚函数类型 |
| `sensitivity_percentile` | 动态敏感性阈值百分位数 | int | 0~100 | 90 | 该算法基于动态模型计算模拟样本的动态敏感性，并对模拟样本进行过滤；计算全部历史数据的动态敏感性值，并根据给定的百分位数来确定动态敏感性阈值 |
| `sigma` | 动态敏感性计算噪声强度 | float | 0~1 | 0.1 | 计算模拟样本动态敏感性值时，添加的噪声强度 |
| `alpha_entropy` | 策略网络损失函数中entropy系数 | float | 0~1 | 0.0 | - | 
| `train_alpha_entropy` | 是否训练 alpha_entropy | bool | True,False | False | - |
| - | q 网络 | list | - | [400,400] | - |
| - | q 网络数量 | int | 1~3 | 2 | - |
| - | policy 网络 | list | - | [300,300] | - |
| - | vae 网络 | list | - | [750,750] | 该输入为 encoder 的结构，decoder 与 encoder 结构相同 |
| - | q 网络学习率 | float | 0~1 | 1e-3 | - |
| - | policy 网络学习率 | float | 0~1 | 1e-5 | - |
| - | vae 网络学习率 | float | 0~1 | 1e-4 | - |

## SABER 模型参数说明
> **S**afe **A**ctor-Critic with **Be**havior **R**egularization (SABER) (原 VAE_MMD)  
> 算法简介：是一种免模型的离线强化学习算法；适用于期望获得最大化长期利益的控制问题；可适用于较大规模的控制问题（控制变量数量可达20~30），使用收集的离线数据进行模型的训练（不需要进行系统动态模型的训练），训练完成后得到系统的优化控制策略；应用场景举例：火电燃烧优化控制。

| 参数名 | 含义 | 取值类型 | 取值范围 | 默认值 | 说明 |
| - | - | - | - | - | - |
| `n_action_samples` | Q网络计算loss时采样action个数 | int | 1~10 | 1 | 若取值大于1，则在Q网络更新时，通过采样多个action并最大化Q值来计算目标 |
| `alpha` | 策略分布差异系数 | float | 0~1 | 0.2 | 策略网络的损失函数中，学习的策略与数据中的行为策略之间分布差异的系数 |
| `train_alpha` | 是否训练alpha | bool | True,False | False | - |
| `beta` | 策略网络损失函数中cost的系数 | float | 0~10 | 1.0 | 策略网络损失函数中，cost网络值的系数 |
| `train_beta` | 是否训练beta | bool | True,False | False | - |
| `alpha_entropy` | 策略网络损失函数中entropy系数 | float | 0~1 | 0.0 | - |
| `train_alpha_entropy` | 是否训练 alpha_entropy | bool | True,False | False | - |
| `policy_type` | 策略网络类型 | str | det,sto | det | det, deterministic, 确定性策略；sto, stochastic, 随机性策略 |
| `mmd_epsilon` | 策略分布差异目标值 | float | 0~1 | 0.05 | - |
| `cost_epsilon` | cost 目标值 | float | 0~1 | 0.05 | - |
| - | q 网络 | list | - | [400, 400] | - |
| - | q 网络数量 | int | 1~3 | 2 | - |
| - | policy 网络 | list | - | [300,300] | - |
| - | behavior 网络 | list | - | [750,750] | 数据中行为策略网络；这里使用 vae 网络进行学习，该输入为 encoder 的结构，decoder 与 encoder 结构相同 |
| - | q 网络学习率 | float | 0~1 | 1e-3 | - |
| - | policy 网络学习率 | float | 0~1 | 1e-5 | - |
| - | behavior 网络学习率 | float | 0~1 | 1e-4 | - |

## MOPP 模型参数说明
> **M**odel-Based **O**ffline **P**lanning with Trajectory **P**running (**MOPP**)  
> 算法简介：是一种轻量级的基于模型的规划算法；适用于期望获得最大化长期利益的控制问题和使系统最优地达到预期目标的最优控制问题；要求控制问题的规模较小（控制变量数量在10个左右），使用收集的离线数据进行模型的训练（需基于离线数据进行系统动态模型的训练），在线规划求解给出系统的优化控制策略；应用场景举例：火电直接空冷系统控制优化、火电锅炉汽温控制优化。

| 参数名 | 含义 | 取值类型| 取值范围 | 默认值 | 说明 |
| - | - | - | - | - | - |
| `use_value_fn` | 是否使用Q-value | bool | True,False | False | 在规划求解过程中，是否在每条模拟轨迹的累积奖励上加上基于轨迹末端状态预测的 Q 值 |
| `pred_len` | 规划轨迹长度 | int | 1~50 | 10 | - |
| `beta` | 轨迹加权系数 | float | 0~1 | 0.6 | 在求解过程中，将前一次求解得到的控制序列与当前采样的控制序列进行加权，此系数为前一次求解的控制序列的加权系数 |
| `pop_size` | 采样轨迹条数 | int | 1~10000 | 1000 | - |
| `kappa` | 奖励加权因子 | float | 0~10 | 0.9 | - |
| `noise_sigma` | 策略分布标准差缩放参数 | float | 0~1 | 0.6 | 在基于行为策略分布进行动作采样时，对行为策略分布的标准差进行缩放的目标值 |
| `uncertainty_percentile` | 动态不确定性阈值计算百分位数 | int | 0~100 | 90 | 该算法基于动态模型计算模拟样本的动态不确定性，通过与动态不确定性阈值进行比较来对模拟样本进行过滤；计算全部历史数据的动态不确定性值，并根据给定的百分位数来确定动态不确定性阈值 |
| `maxq` | 是否使用Max-Q采样 | bool | True,False | True | - |
| `b_alpha_entropy` | 行为策略网络损失函数中entropy系数 | float | 0~1 | 0.0 | - |
| `train_b_alpha_entropy` | 是否训练 b_alpha_entropy | bool | True,False | False | - |
| - | q 网络 | list | - | [400, 400] | - |
| - | q 网络数量 | int | 1~3 | 2 | - |
| - | q 网络学习率 | float | 0~1 | 1e-3 | - |
| - | behavior 网络 | list | - | [500,200,100] | - |
| - | behavior 网络学习率 | float | 0~1 | 1e-3 | - |


---
## 通用参数

* ### Train 参数说明
| 参数名 | 含义 | 取值类型| 取值范围 | 默认值 |
| - | - | - | - | - |
| `train_test_ratio` | 训练集比例 | float | 0~1 | 0.99 |
| `batch_size` | 每个batch的大小 | int | 1~256 | 64 |
| `weight_decays` | 网络L2正则化系数 | float | 0~1 | 1e-5 |
| `update_freq` | 更新target网络参数频率 | int | 1~5 | 1 |
| `update_rate` | 更新target网络参数的比率 | float | 0~1 | 0.005 |
| `discount` | 奖励折扣因子 | float | 0~1 | 0.99 |
| `total_train_steps` | 模型训练步数 | int | 0~5e+6 | 500000 |

* ### Dynamics 模型参数
| 参数名 | 含义 | 取值类型| 取值范围 | 默认值 |
| - | - | - | - | - |
| `dynamic_module_type` | 动态模型类型 | str | `dnn` | dnn |
| - | 网络结构 | - | - | [200, 200] |
| - | 网络数量 | int | 1~5 | 3 |
| - | 学习率 | float | 0~1 | 1e-3 |

* ### 模型评价参数
| 参数名 | 含义 | 取值类型| 取值范围 | 默认值 | 说明 |
| - | - | - | - | - | - |
| `episode_step` | 预测步数 | int | 1~20 | 10 | 测试中，采用推荐的优化动作后，预测未来状态变化时预测的最大步数 |
| `start` | 测试数据起点百分比 | float | 0~1 | 0.9 | - |
| `steps` | 测试步数 | int | 1~10000 | 1000 | 从测试数据起点开始，测试的数据点的总数 |

* ### 数据参数
| 参数名 | 含义 | 取值类型| 取值范围 | 默认值 | 说明 |
| - | - | - | - | - | - |
| - | state 索引起点| int | - | - | 数据中状态特征的起点索引 |
| - | state 索引终点| int | - | - | 数据中状态特征的终点索引 |
| - | state 上界| int | - | 1 | state 取值上界 |
| - | state 下界| int | - | 0 | state 取值下界 |
| - | action 索引起点| int | - | - | 数据中动作特征的起点索引 |
| - | action 索引终点| int | - | - | 数据中动作特征的终点索引 |
| - | action 上界| int | - | 1 | action 取值上界 |
| - | action 下界| int | - | 0 | action 取值下界 |
