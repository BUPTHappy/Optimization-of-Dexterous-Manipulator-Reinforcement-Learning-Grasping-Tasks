"""
Script to train the PPO policy for dexterous grasping
Check scripts/train.sh for training scripts
"""
import os
from a2c_ppo_acktr.arguments import get_args

args = get_args()

import time
import shutil
import os.path as osp
from collections import deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage


def main():
    #设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    #确定性设置，保证试验的可重现（结果一致）
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False #cuDNN使用默认算法，不进行性能优化选择
        torch.backends.cudnn.deterministic = True #强制cuDNN只使用确定性算法，保证相同输入产生完全相同的输出

    exp_dir = args.exp
    log_dir = osp.join(exp_dir, 'monitor')#monitor/ 训练监控
    tb_dir = osp.join(exp_dir, 'logs')  #logs/  TensorBoard日志  
    save_dir = osp.join(exp_dir, 'models') #models/ 训练好的模型
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    utils.cleanup_log_dir(log_dir)
    writer = SummaryWriter(tb_dir)
    
    # copy important codes into exp dir 
    code_dir = osp.join(exp_dir, 'codes') #codes/  得到这组结果当时使用的代码组
    os.makedirs(code_dir, exist_ok=True)
    with open(osp.join(code_dir, 'args.txt'), 'w') as f:
        f.write(str(args)+'\n')
    curr_dir = osp.dirname(osp.abspath(__file__))
    shutil.copy(osp.join(curr_dir, 'train.py'), code_dir)
    shutil.copy(osp.join(curr_dir, 'a2c_ppo_acktr/arguments.py'), code_dir)
    shutil.copy(osp.join(curr_dir, 'a2c_ppo_acktr/envs.py'), code_dir)
    shutil.copy(osp.join(curr_dir, 'a2c_ppo_acktr/model.py'), code_dir) 
    shutil.copy(osp.join(curr_dir, 'envs/mj_envs/dex_manip/graff.py'), code_dir)

    torch.set_num_threads(1) #每个进程只用1个线程（避免CPU多线程竞争）
    device = torch.device("cuda" if args.cuda else "cpu")
    
    grasp_attrs_dict = {'dataset': args.dataset,
                        'obj': args.obj,
                        'policy': args.policy, #cnn-mlp
                        'cnn_arch': args.cnn_arch,
                        'noise': args.noise,
                        'inputs': args.inputs, #proprio loc rgb depth aff
                        'cameras': args.cameras,
                        'img_res': args.img_res,
                        'rewards': args.rewards, #grasp:1 aff:1 
                        'reward_dst_thr': args.reward_dst_thr,
                        'obj_mass': args.obj_mass,
                        'obj_rot': args.obj_rot,
                        'obj_tr': args.obj_tr,
                        'gravity': args.gravity,
                        'debug': args.debug}

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, log_dir, device, int(args.gpu_env), True, dataset=args.dataset, 
                         object=args.obj, grasp_attrs_dict=grasp_attrs_dict)

    if args.load_model == 0: #从头开始训练 
        # 创建全新的策略网络
        actor_critic = Policy(
            envs.observation_space, # 观察空间（输入维度）
            envs.action_space,      # 动作空间（输出维度）
            policy=args.policy,     
            cnn_args={'arch': args.cnn_arch,
                      'pretrained': args.cnn_pretrained,
                      'cameras': args.cameras},
            base_kwargs={'recurrent': args.recurrent_policy})
        start_update_num = 0  #从第0次更新开始
    else:
        from a2c_ppo_acktr.utils import get_vec_normalize
        # 加载已保存的模型和观察统计信息
        actor_critic, ob_rms = \
            torch.load(os.path.join(args.exp, 'models', str(args.load_model) + ".pt"))

        start_update_num = args.load_model + 1 #从上次停止的地方继续

        #恢复观察归一化的统计信息
        vec_norm = get_vec_normalize(envs)
        if vec_norm is not None:
            vec_norm.eval() # 设置为评估模式
            vec_norm.ob_rms = ob_rms # 恢复之前的统计信息

    actor_critic.to(device)

    #创建PPO优化器
    agent = algo.PPO(
        actor_critic, # 要训练的策略网络
        args.clip_param, # PPO裁剪参数：限制策略更新幅度
        args.ppo_epoch, # 每次更新的训练轮数：每批数据训练几次
        args.num_mini_batch, # 小批量数量：分成多少个小批量
        args.value_loss_coef, # 价值函数损失权重
        args.entropy_coef, # 熵正则化权重：鼓励探索
        lr=args.lr, # 学习率
        eps=args.eps, # Adam优化器的epsilon
        max_grad_norm=args.max_grad_norm # 梯度裁剪阈值：防止梯度爆炸
    )

    #创建经验存储
    rollouts = RolloutStorage(
        args.num_steps,  # 每次收集多少步
        args.num_processes, # 并行环境数量  
        envs.observation_space,  # 观察空间大小
        envs.action_space, # 动作空间大小
        actor_critic.recurrent_hidden_state_size  # RNN隐藏状态大小
    )

    obs = envs.reset() # 重置所有环境，获取初始观察
    rollouts.obs.copy_(0, obs) # 把初始观察存入第0步
    rollouts.to(device)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    train_rewards = deque(maxlen=10)  # 最近10个episode的奖励
    episode_successes = deque(maxlen=10)  # 最近10个episode的成功率(举起物体)  
    episode_successes_orig = deque(maxlen=10)  # 最近10个episode的成功率(抓住物体)  
    best_ep_rews = -np.inf  # 记录最佳奖励

    #PPO强化学习的主训练循环：
    for j in range(start_update_num, num_updates): #外层：大循环，每次更新网络（用经验更新大脑）

        #第一步：学习率调整，线性递减
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates, args.lr)

        #第二步：收集经验
        for step in range(args.num_steps):  #内层：小循环，收集数据（经验）
            
            #【智能体的决策】
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], # 当前观察（机器人看到什么）
                    rollouts.recurrent_hidden_states[step], # 记忆状态
                    rollouts.masks[step] # 哪些环境还在运行
                )

            #【环境的反馈】
            obs, reward, done, infos = envs.step(action)

            #【记录成功失败】
            for info in infos:
                if 'episode' in info.keys():
                    train_rewards.append(info['episode']['r']) #这次得了多少reward
                    episode_successes.append(info['episode']['obj_lift']) #成功举起物体了吗？(obj_lift: True/False)  
                    episode_successes_orig.append(info['episode']['obj_grab']) #成功抓住物体了吗？(obj_grab: True/False)

            # If done then clean the history of observations
            # mask告诉系统：哪些经验是连续的，哪些需要重新开始
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            #标记哪些是"坏"的结束（比如超时，不是真正失败）
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            #【存储经验】
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        #第三步：计算未来价值
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1],  # 最后一步的观察
                rollouts.recurrent_hidden_states[-1], # 最后的记忆状态  
                rollouts.masks[-1] # 最后的mask
            ).detach()

        #第四步：计算回报（考虑了未来奖励）
        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        #第五步：更新神经网络
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        ##返回三个损失：
            #value_loss: 价值预测有多准？
            #action_loss: 动作选择有多好？  
            #dist_entropy: 探索够不够多？
        
        #第六步：清理准备下次
        rollouts.after_update()

        #第七步：监控和记录
        if j % args.log_interval == 0 and len(train_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(train_rewards), np.mean(train_rewards),
                            np.median(train_rewards), np.min(train_rewards),
                            np.max(train_rewards), dist_entropy, value_loss,
                            action_loss))
            writer.add_scalar('Loss/total', value_loss * args.value_loss_coef + action_loss -
                              dist_entropy * args.entropy_coef, total_num_steps)
            writer.add_scalar('Loss/value', value_loss, total_num_steps)
            writer.add_scalar('Loss/action', action_loss, total_num_steps)
            writer.add_scalar('Loss/entropy', dist_entropy, total_num_steps)
            writer.add_scalar('Rewards/mean', np.mean(train_rewards), total_num_steps)
            writer.add_scalar('Rewards/median', np.median(train_rewards), total_num_steps)
            writer.add_scalar('Rewards/max', np.max(train_rewards), total_num_steps)
            writer.add_scalar('Rewards/min', np.min(train_rewards), total_num_steps)
            writer.add_scalar('Success_rate/hand-obj-notable', np.mean(episode_successes), total_num_steps)
            writer.add_scalar('Success_rate/obj-notable', np.mean(episode_successes_orig), total_num_steps)

        # save model for every interval-th episode or for the last epoch
        #第八步：保存模型
        if ((j + 1) % args.save_interval == 0 or j == num_updates - 1):
            # save model
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_dir, str(j + 1) + ".pt"))

            # save best model if available
            if np.mean(train_rewards) > best_ep_rews:
                print('Best model found. Saving.')
                best_ep_rews = np.mean(train_rewards)
                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], os.path.join(save_dir, "best.pt"))


if __name__ == "__main__":
    main()
