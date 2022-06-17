# PPO-GAIl

## 各个文件用途

| 文件名                     | 用途                                            |
| -------------------------- | ----------------------------------------------- |
| cluster.py                 | 包含cluster与machine类的定义                    |
| expert_function.py         | 包含不同启发式方法的优先级计算函数              |
| HPCSimpickjobs_gail.py     | 包含gym环境代码                                 |
| test_different_schedule.py | 测试不同调度方式性能                            |
| test_my_schedule.py        | 测试PPO-GAIL调度方式性能                        |
| job.py                     | 包含job类与load类，负责日志数据读取生成对应load |
| ppo-gail.py                | 主程序入口                                      |

## 使用

python ppo-gail.py --workload "数据集名1 数据集名2 数据集名3" --exp_name 实验名 --backfill 0 /1 --epochs 训练总轮次 --trajs 单次采样轨迹数 

## 验证

python test_different_schedule.py --workload "数据集名"

python test_my_schedule.py --workload "数据集名" --pre_trained 1 --trained_model "模型路径"
