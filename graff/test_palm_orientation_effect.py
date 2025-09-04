#!/usr/bin/env python3
"""
测试虎口朝上效果的专用脚本
比较有无虎口朝上约束的模型表现
"""

import argparse
import os
import sys
import numpy as np
import torch
from collections import defaultdict

sys.path.append('.')
from envs.mj_envs.dex_manip.graff import GraffV0
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.utils import get_vec_normalize

class PalmOrientationTester:
    def __init__(self, exp_path, model_name='best'):
        self.exp_path = exp_path
        self.model_name = model_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 加载模型
        model_path = os.path.join(exp_path, 'models', f'{model_name}.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        self.actor_critic, self.ob_rms = torch.load(model_path, map_location=self.device)
        print(f"成功加载模型: {model_path}")
        
    def create_env(self, with_palm_constraint=True):
        """创建环境，可选择是否包含虎口朝上约束"""
        if with_palm_constraint:
            rewards = {'grasp': 1.0, 'aff': 1.0, 'palm_orientation': 2.0}
        else:
            rewards = {'grasp': 1.0, 'aff': 1.0}
            
        grasp_attrs_dict = {
            'dataset': 'contactdb',
            'obj': 'pan',
            'policy': 'cnn-mlp',
            'cnn_arch': 'custom',
            'noise': False,
            'inputs': ['proprio', 'loc', 'rgb', 'depth', 'aff'],
            'cameras': ['egocentric'],
            'img_res': 128,
            'rewards': rewards,
            'reward_dst_thr': 0.05,
            'obj_mass': 1.0,
            'obj_rot': True,
            'obj_tr': False,
            'gravity': -9.81,
            'debug': False
        }
        
        env = GraffV0(object='pan', device_id=0, process_id=0, grasp_attrs_dict=grasp_attrs_dict)
        return env
    
    def test_palm_orientation_metrics(self, num_episodes=20):
        """测试虎口朝上的具体指标"""
        env = self.create_env(with_palm_constraint=True)
        
        orientation_scores = []
        grasp_successes = []
        episode_rewards = []
        
        print(f"开始测试虎口朝上效果，共{num_episodes}个episode...")
        
        for episode in range(num_episodes):
            obs = env.reset_model()
            episode_reward = 0
            orientation_rewards = []
            done = False
            step = 0
            
            while not done and step < 200:
                # 获取动作
                with torch.no_grad():
                    if isinstance(obs, dict):
                        obs_tensor = {}
                        for key, value in obs.items():
                            obs_tensor[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
                    else:
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    
                    _, action, _, _ = self.actor_critic.act(obs_tensor, None, None)
                    action = action.cpu().numpy().flatten()
                
                # 执行动作
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                # 计算虎口朝上奖励
                palm_orientation_reward = env.get_palm_orientation_reward()
                orientation_rewards.append(palm_orientation_reward)
                
                step += 1
            
            # 记录结果
            avg_orientation = np.mean(orientation_rewards)
            orientation_scores.append(avg_orientation)
            grasp_successes.append(info.get('obj_grab', False))
            episode_rewards.append(episode_reward)
            
            print(f"Episode {episode+1}: 平均虎口朝上得分={avg_orientation:.3f}, "
                  f"抓取成功={info.get('obj_grab', False)}, 总奖励={episode_reward:.2f}")
        
        # 统计结果
        results = {
            'avg_orientation_score': np.mean(orientation_scores),
            'std_orientation_score': np.std(orientation_scores),
            'grasp_success_rate': np.mean(grasp_successes),
            'avg_episode_reward': np.mean(episode_rewards),
            'orientation_scores': orientation_scores,
            'grasp_successes': grasp_successes,
            'episode_rewards': episode_rewards
        }
        
        return results
    
    def visualize_palm_orientation(self, num_episodes=5):
        """可视化虎口朝上的效果（无渲染版本）"""
        print("注意：服务器环境不支持图形渲染，使用数值显示模式")
        print("=" * 50)
        
        # 直接调用数值测试，但显示更详细的信息
        env = self.create_env(with_palm_constraint=True)
        
        print(f"开始详细测试虎口朝上效果，共{num_episodes}个episode...")
        
        for episode in range(num_episodes):
            print(f"\n=== Episode {episode+1} ===")
            obs = env.reset_model()
            done = False
            step = 0
            episode_reward = 0
            orientation_rewards = []
            
            while not done and step < 200:
                # 获取动作
                with torch.no_grad():
                    if isinstance(obs, dict):
                        obs_tensor = {}
                        for key, value in obs.items():
                            obs_tensor[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
                    else:
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    
                    _, action, _, _ = self.actor_critic.act(obs_tensor, None, None)
                    action = action.cpu().numpy().flatten()
                
                # 执行动作
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                # 计算虎口朝上指标
                palm_orientation_reward = env.get_palm_orientation_reward()
                orientation_rewards.append(palm_orientation_reward)
                
                # 获取手掌姿态信息
                palm_pos = env.data.site_xpos[env.S_grasp_sid].ravel()
                palm_rotmat = env.data.site_xmat[env.S_grasp_sid].reshape(3, 3)
                palm_z = palm_rotmat[:, 2]  # 手掌z轴
                world_up = np.array([0, 0, 1])
                up_alignment = np.dot(palm_z, world_up)
                
                # 每20步显示一次详细信息
                if step % 20 == 0:
                    print(f"  Step {step:3d}: 虎口朝上={palm_orientation_reward:.3f}, "
                          f"向上对齐={up_alignment:.3f}, 奖励={reward:.2f}")
                
                step += 1
            
            avg_orientation = np.mean(orientation_rewards)
            print(f"Episode {episode+1} 结束:")
            print(f"  抓取成功: {info.get('obj_grab', False)}")
            print(f"  平均虎口朝上得分: {avg_orientation:.3f}")
            print(f"  总奖励: {episode_reward:.2f}")
            print(f"  总步数: {step}")
    
    def compare_with_baseline(self, baseline_exp_path, num_episodes=10):
        """与没有虎口朝上约束的基线模型对比"""
        print("正在与基线模型对比...")
        
        # 测试当前模型（有虎口朝上约束）
        print("测试当前模型（有虎口朝上约束）...")
        current_results = self.test_palm_orientation_metrics(num_episodes)
        
        # 如果有基线模型，也测试基线
        if os.path.exists(baseline_exp_path):
            print("测试基线模型（无虎口朝上约束）...")
            baseline_tester = PalmOrientationTester(baseline_exp_path, self.model_name)
            baseline_results = baseline_tester.test_palm_orientation_metrics(num_episodes)
            
            # 对比结果
            print("\n=== 对比结果 ===")
            print(f"当前模型（有约束）:")
            print(f"  平均虎口朝上得分: {current_results['avg_orientation_score']:.3f} ± {current_results['std_orientation_score']:.3f}")
            print(f"  抓取成功率: {current_results['grasp_success_rate']:.3f}")
            print(f"  平均episode奖励: {current_results['avg_episode_reward']:.2f}")
            
            print(f"基线模型（无约束）:")
            print(f"  平均虎口朝上得分: {baseline_results['avg_orientation_score']:.3f} ± {baseline_results['std_orientation_score']:.3f}")
            print(f"  抓取成功率: {baseline_results['grasp_success_rate']:.3f}")
            print(f"  平均episode奖励: {baseline_results['avg_episode_reward']:.2f}")
            
            # 计算改进
            orientation_improvement = current_results['avg_orientation_score'] - baseline_results['avg_orientation_score']
            success_improvement = current_results['grasp_success_rate'] - baseline_results['grasp_success_rate']
            
            print(f"\n改进效果:")
            print(f"  虎口朝上得分提升: {orientation_improvement:+.3f}")
            print(f"  抓取成功率变化: {success_improvement:+.3f}")
            
        else:
            print(f"基线模型路径不存在: {baseline_exp_path}")
            print("只显示当前模型结果:")
            print(f"  平均虎口朝上得分: {current_results['avg_orientation_score']:.3f} ± {current_results['std_orientation_score']:.3f}")
            print(f"  抓取成功率: {current_results['grasp_success_rate']:.3f}")
            print(f"  平均episode奖励: {current_results['avg_episode_reward']:.2f}")

def main():
    parser = argparse.ArgumentParser(description='测试虎口朝上效果')
    parser.add_argument('--exp', type=str, required=True, help='实验路径')
    parser.add_argument('--model', type=str, default='best', help='模型名称')
    parser.add_argument('--baseline', type=str, help='基线模型路径（可选）')
    parser.add_argument('--mode', type=str, choices=['test', 'visualize', 'compare'], 
                       default='test', help='测试模式')
    parser.add_argument('--episodes', type=int, default=20, help='测试episode数量')
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = PalmOrientationTester(args.exp, args.model)
    
    if args.mode == 'test':
        print("=== 测试虎口朝上效果 ===")
        results = tester.test_palm_orientation_metrics(args.episodes)
        print(f"\n最终结果:")
        print(f"平均虎口朝上得分: {results['avg_orientation_score']:.3f} ± {results['std_orientation_score']:.3f}")
        print(f"抓取成功率: {results['grasp_success_rate']:.3f}")
        print(f"平均episode奖励: {results['avg_episode_reward']:.2f}")
        
    elif args.mode == 'visualize':
        print("=== 可视化虎口朝上效果 ===")
        tester.visualize_palm_orientation(args.episodes)
        
    elif args.mode == 'compare':
        print("=== 对比虎口朝上效果 ===")
        baseline_path = args.baseline or './expts/graff_trained'
        tester.compare_with_baseline(baseline_path, args.episodes)

if __name__ == '__main__':
    main()
