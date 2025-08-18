"""
测试训练好的模型是否学会了虎口朝上抓取
"""
import os
import numpy as np
import torch
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.utils import get_vec_normalize
import time

def test_palm_orientation(model_path, num_episodes=10):
    """
    测试模型的虎口朝上抓取能力
    """
    print(f"加载模型: {model_path}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        return
    
    # 加载模型
    try:
        actor_critic, ob_rms = torch.load(model_path, map_location='cpu')
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 设置环境参数
    grasp_attrs_dict = {
        'dataset': 'contactdb',
        'obj': 'pan',
        'policy': 'cnn-mlp',
        'cnn_arch': 'custom',
        'noise': False,  # 测试时不加噪声
        'inputs': ['proprio', 'loc', 'rgb', 'depth', 'aff'],
        'cameras': ['egocentric'],
        'img_res': 128,
        'rewards': {'grasp': 1, 'aff': 1, 'palm_orientation': 0.3},
        'reward_dst_thr': 0.5,
        'obj_mass': 1,
        'obj_rot': True,
        'obj_tr': False,
        'gravity': 9.8,
        'debug': False
    }
    
    # 创建环境
    device = torch.device('cpu')
    envs = make_vec_envs('graff-v0', 1, 1, None, None, device, 0, True, 
                        dataset='contactdb', object='pan', 
                        grasp_attrs_dict=grasp_attrs_dict)
    
    # 恢复观察归一化
    vec_norm = get_vec_normalize(envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms
    
    actor_critic.eval()
    
    # 测试统计
    total_episodes = 0
    successful_grasps = 0
    successful_lifts = 0
    palm_orientation_scores = []
    episode_rewards = []
    
    print(f"\n开始测试 {num_episodes} 个episodes...")
    print("=" * 60)
    
    for episode in range(num_episodes):
        obs = envs.reset()
        episode_reward = 0
        episode_palm_scores = []
        episode_grasp = False
        episode_lift = False
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        for step in range(200):  # 最大200步
            with torch.no_grad():
                _, action, _, _ = actor_critic.act(obs, None, None, deterministic=True)
            
            obs, reward, done, infos = envs.step(action)
            episode_reward += reward.item()
            
            # 获取环境信息
            if len(infos) > 0 and 'obj_grab' in infos[0]:
                if infos[0]['obj_grab']:
                    episode_grasp = True
                if infos[0]['obj_lift']:
                    episode_lift = True
            
            # 计算虎口朝上得分（需要访问环境内部）
            try:
                env = envs.venv.envs[0].env  # 获取原始环境
                while hasattr(env, 'env'):
                    env = env.env
                palm_score = env.get_palm_orientation_reward()
                episode_palm_scores.append(palm_score)
            except:
                pass
            
            if done:
                break
        
        total_episodes += 1
        if episode_grasp:
            successful_grasps += 1
        if episode_lift:
            successful_lifts += 1
        
        episode_rewards.append(episode_reward)
        if episode_palm_scores:
            avg_palm_score = np.mean(episode_palm_scores)
            palm_orientation_scores.append(avg_palm_score)
            print(f"  奖励: {episode_reward:.2f}, 抓取: {'✓' if episode_grasp else '✗'}, "
                  f"举起: {'✓' if episode_lift else '✗'}, 虎口朝上得分: {avg_palm_score:.3f}")
        else:
            print(f"  奖励: {episode_reward:.2f}, 抓取: {'✓' if episode_grasp else '✗'}, "
                  f"举起: {'✓' if episode_lift else '✗'}")
    
    # 打印测试结果
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print(f"总episodes: {total_episodes}")
    print(f"成功抓取率: {successful_grasps}/{total_episodes} ({100*successful_grasps/total_episodes:.1f}%)")
    print(f"成功举起率: {successful_lifts}/{total_episodes} ({100*successful_lifts/total_episodes:.1f}%)")
    print(f"平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    
    if palm_orientation_scores:
        avg_palm_score = np.mean(palm_orientation_scores)
        print(f"平均虎口朝上得分: {avg_palm_score:.3f} ± {np.std(palm_orientation_scores):.3f}")
        print(f"虎口朝上表现: {'优秀' if avg_palm_score > 0.8 else '良好' if avg_palm_score > 0.6 else '一般' if avg_palm_score > 0.4 else '较差'}")
    
    # 评估建议
    print("\n评估建议:")
    if successful_grasps == 0:
        print("❌ 模型还没有学会抓取，需要继续训练")
    elif successful_grasps < total_episodes * 0.3:
        print("⚠️  模型偶尔能抓取，但成功率较低，建议继续训练")
    elif successful_grasps < total_episodes * 0.7:
        print("✅ 模型已经学会基本抓取，可以考虑调整奖励权重优化虎口朝上")
    else:
        print("🎉 模型抓取能力很好！")
    
    if palm_orientation_scores and avg_palm_score < 0.5:
        print("📝 虎口朝上约束还需要加强，建议增加palm_orientation奖励权重")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./expts/graff_palm_seed1/models/best.pt',
                       help='模型文件路径')
    parser.add_argument('--episodes', type=int, default=10,
                       help='测试episodes数量')
    args = parser.parse_args()
    
    test_palm_orientation(args.model, args.episodes)
