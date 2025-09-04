# GRAFF 虎口朝上抓取训练指南

## 当前状态分析

根据测试结果，你的模型已经有了基础能力：
- ✅ 成功抓取率：20% (1/5)
- ✅ 成功举起率：20% (1/5) 
- ⚠️ 虎口朝上得分：0.324/1.0 (较差)
- ⚠️ 平均奖励：-1169.88 (负数，还在学习中)

**问题诊断：** 1000步训练太短，模型还没有充分学习。

## 训练策略建议

### 方案1：继续训练 (推荐)
从现有模型继续训练，增加虎口朝上奖励权重：

```bash
# 给脚本执行权限
chmod +x ./scripts/train/continue_training.sh

# 开始继续训练 (100,000步，约需要几小时)
./scripts/train/continue_training.sh
```

**改进点：**
- 增加训练步数：1,000 → 100,000
- 提高虎口朝上奖励权重：0.3 → 2.0
- 降低学习率：5e-5 → 1e-4 (更稳定的学习)
- 从第9个模型继续训练

### 方案2：长期训练 (如果有充足时间)
完全重新训练，使用更好的超参数：

```bash
# 给脚本执行权限  
chmod +x ./scripts/train/graff_palm_long.sh

# 开始长期训练 (50,000,000步，可能需要几天)
./scripts/train/graff_palm_long.sh
```

**改进点：**
- 大幅增加训练步数：1,000 → 50,000,000
- 增加并行环境：1 → 8 (加速训练)
- 优化超参数配置
- 虎口朝上奖励权重：1.0

## 训练监控

### 1. 实时监控训练进度
```bash
# 查看训练进程
ps aux | grep python | grep train

# 如果想要保存日志，重新启动训练时加上：
nohup ./scripts/train/continue_training.sh > training.log 2>&1 &

# 实时查看日志
tail -f training.log
```

### 2. 使用TensorBoard可视化
```bash
# 启动TensorBoard
tensorboard --logdir=./expts/graff_palm_continue/logs --port=6006

# 浏览器访问: http://localhost:6006
```

### 3. 定期测试模型
```bash
# 测试最新的best模型
python test_palm_orientation.py --model ./expts/graff_palm_continue/models/best.pt --episodes 10

# 测试特定编号的模型
python test_palm_orientation.py --model ./expts/graff_palm_continue/models/500.pt --episodes 10
```

## 期望的训练进度

### 短期目标 (10,000-20,000步)
- 成功抓取率：40-60%
- 虎口朝上得分：0.5-0.6
- 平均奖励：-500 到 -200

### 中期目标 (50,000-100,000步)  
- 成功抓取率：70-80%
- 虎口朝上得分：0.7-0.8
- 平均奖励：-100 到 100

### 长期目标 (500,000+步)
- 成功抓取率：85-95%
- 虎口朝上得分：0.8-0.9
- 平均奖励：100-500

## 故障排除

### 如果训练卡住或崩溃
```bash
# 检查GPU内存
nvidia-smi

# 检查磁盘空间
df -h

# 重启训练（从最新保存的模型）
# 修改continue_training.sh中的--load_model参数为最新的模型编号
```

### 如果奖励不提升
1. 检查虎口朝上奖励权重是否足够高
2. 考虑调整其他奖励权重的平衡
3. 检查环境是否正常工作

### 如果想要调整训练参数
编辑 `./scripts/train/continue_training.sh` 文件：
- `--rewards grasp:1 aff:1 palm_orientation:X` - 调整奖励权重
- `--lr X` - 调整学习率
- `--num-env-steps X` - 调整总训练步数

## 下一步行动

**立即执行：**
```bash
# 1. 开始继续训练
chmod +x ./scripts/train/continue_training.sh
./scripts/train/continue_training.sh

# 2. 在另一个终端监控进度
watch -n 30 'ls -lt ./expts/graff_palm_continue/models/ | head -5'

# 3. 每隔一段时间测试模型
# (建议每训练1-2小时测试一次)
python test_palm_orientation.py --model ./expts/graff_palm_continue/models/best.pt --episodes 5
```

训练需要耐心，强化学习通常需要大量的试错才能学会复杂的技能！
