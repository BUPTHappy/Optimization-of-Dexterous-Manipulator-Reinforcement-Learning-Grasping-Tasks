#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

import numpy as np
from envs.mj_envs.dex_manip.graff import GraffV0

def test_enhanced_palm_orientation():
    """
    测试增强的手掌朝向奖励函数
    """
    print("=== 测试增强的手掌朝向奖励函数 ===")
    
    # 创建环境
    grasp_attrs_dict = {
        'dataset': 'contactdb',
        'policy': 'mlp',
        'noise': False,
        'cameras': ['egocentric'],
        'rewards': {'grasp': 1, 'aff': 1, 'palm_orientation': 2.0},
        'reward_dst_thr': 0.5,
        'obj_mass': 0.8,
        'obj_rot': True,
        'obj_tr': False,
        'gravity': 9.8,
        'debug': False,
        'inputs': ['proprio', 'loc', 'rgb', 'depth', 'aff'],
        'img_res': 84
    }
    
    # 测试不同物体
    objects = ['hammer', 'pan', 'knife']
    
    for obj_name in objects:
        print(f"\n--- 测试物体: {obj_name} ---")
        
        try:
            env = GraffV0(object=obj_name, grasp_attrs_dict=grasp_attrs_dict)
            
            # 重置环境
            obs = env.reset()
            print(f"环境重置成功，观察空间: {type(obs)}")
            
            # 测试多个步骤
            total_palm_reward = 0
            num_steps = 10
            
            for step in range(num_steps):
                # 随机动作
                action = np.random.uniform(-0.5, 0.5, env.action_space.shape)
                
                # 执行动作
                obs, reward, done, info = env.step(action)
                
                # 获取手掌朝向奖励
                palm_reward = env.get_palm_orientation_reward()
                total_palm_reward += palm_reward
                
                # 获取手掌和物体信息
                palm_pos = env.data.site_xpos[env.S_grasp_sid].ravel()
                palm_rotmat = env.data.site_xmat[env.S_grasp_sid].reshape(3, 3)
                obj_pos = env.data.body_xpos[env.obj_bid].ravel()
                obj_rotmat = env.data.body_xmat[env.obj_bid].reshape(3, 3)
                
                palm_z = palm_rotmat[:, 2]
                distance = np.linalg.norm(obj_pos - palm_pos)
                
                print(f"步骤 {step+1}:")
                print(f"  手掌朝向奖励: {palm_reward:.3f}")
                print(f"  总奖励: {reward:.3f}")
                print(f"  手掌Z轴Z分量: {palm_z[2]:.3f}")
                print(f"  手掌-物体距离: {distance:.3f}")
                print(f"  抓取状态: {info.get('obj_grab', False)}")
                
                if done:
                    print(f"  Episode结束")
                    break
            
            avg_palm_reward = total_palm_reward / num_steps
            print(f"\n{obj_name} 平均手掌朝向奖励: {avg_palm_reward:.3f}")
            
            # 测试特定姿态的奖励
            print(f"\n--- 测试 {obj_name} 的特定姿态奖励 ---")
            
            # 重置环境
            env.reset()
            
            # 测试不同的手掌朝向
            test_orientations = [
                "正常朝向",
                "轻微倒置", 
                "严重倒置"
            ]
            
            for i, orientation in enumerate(test_orientations):
                # 模拟不同的手掌朝向（通过修改Z轴分量）
                palm_rotmat = env.data.site_xmat[env.S_grasp_sid].reshape(3, 3)
                
                if i == 1:  # 轻微倒置
                    palm_rotmat[:, 2] = np.array([0, 0, -0.3])
                elif i == 2:  # 严重倒置
                    palm_rotmat[:, 2] = np.array([0, 0, -0.7])
                
                # 重新计算奖励
                reward = env.get_palm_orientation_reward()
                print(f"  {orientation}: {reward:.3f}")
            
        except Exception as e:
            print(f"测试 {obj_name} 时出错: {e}")
            continue
    
    print("\n=== 测试完成 ===")

def test_reward_components():
    """
    测试奖励函数的各个组件
    """
    print("\n=== 测试奖励组件 ===")
    
    grasp_attrs_dict = {
        'dataset': 'contactdb',
        'policy': 'mlp',
        'noise': False,
        'cameras': ['egocentric'],
        'rewards': {'grasp': 1, 'aff': 1, 'palm_orientation': 2.0},
        'reward_dst_thr': 0.5,
        'obj_mass': 0.8,
        'obj_rot': True,
        'obj_tr': False,
        'gravity': 9.8,
        'debug': False,
        'inputs': ['proprio', 'loc', 'rgb', 'depth', 'aff'],
        'img_res': 84
    }
    
    try:
        env = GraffV0(object='hammer', grasp_attrs_dict=grasp_attrs_dict)
        env.reset()
        
        # 测试距离对奖励的影响
        print("测试距离对奖励的影响:")
        
        palm_pos = env.data.site_xpos[env.S_grasp_sid].ravel()
        obj_pos = env.data.body_xpos[env.obj_bid].ravel()
        
        distances = [0.01, 0.05, 0.1, 0.2, 0.5]
        
        for dist in distances:
            # 模拟不同距离
            original_obj_pos = obj_pos.copy()
            env.data.body_xpos[env.obj_bid] = palm_pos + np.array([dist, 0, 0])
            env.sim.forward()
            
            reward = env.get_palm_orientation_reward()
            print(f"  距离 {dist:.2f}m: 奖励 {reward:.3f}")
            
            # 恢复原始位置
            env.data.body_xpos[env.obj_bid] = original_obj_pos
            env.sim.forward()
        
        print("\n测试完成!")
        
    except Exception as e:
        print(f"测试奖励组件时出错: {e}")

if __name__ == "__main__":
    test_enhanced_palm_orientation()
    test_reward_components()
