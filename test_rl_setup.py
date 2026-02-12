"""
Test script to verify RL training setup is working correctly.

This script performs basic checks:
1. Import all required modules
2. Initialize models
3. Test forward pass
4. Verify reward computation
"""

import os
import sys
import torch
import numpy as np

# Ensure workspace root is on PYTHONPATH
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        from trainer.train_sre_rl import SREActorCritic, PPOMemory, RLEnvironmentWrapper
        from utils.rl_rewards import compute_episode_reward, compute_target_distance
        from policy.sre_model import SpatialEncoder
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_model_initialization():
    """Test that models can be initialized."""
    print("\nTesting model initialization...")
    try:
        from trainer.train_sre_rl import SREActorCritic
        
        args = type('Args', (), {
            'device': torch.device('cpu'),
            'num_patches': 10,
            'patch_size': 64
        })()
        
        model = SREActorCritic(args)
        print(f"✓ Model initialized successfully")
        print(f"  - Actor parameters: {sum(p.numel() for p in model.actor.parameters()):,}")
        print(f"  - Critic parameters: {sum(p.numel() for p in model.critic.parameters()):,}")
        return True
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False


def test_forward_pass():
    """Test forward pass through the model."""
    print("\nTesting forward pass...")
    try:
        from trainer.train_sre_rl import SREActorCritic
        
        args = type('Args', (), {
            'device': torch.device('cpu'),
            'num_patches': 10,
            'patch_size': 64
        })()
        
        model = SREActorCritic(args)
        
        # Create dummy inputs
        batch_size = 2
        scene_image = torch.randn(batch_size, 1, 224, 224)
        target_mask = torch.randn(batch_size, 1, 224, 224)
        object_masks = torch.randn(batch_size, 10, 1, 224, 224)
        bboxes = torch.randn(batch_size, 10, 4)
        
        # Forward pass
        with torch.no_grad():
            action_logits, value, valid_mask = model(scene_image, target_mask, object_masks, bboxes)
        
        print(f"✓ Forward pass successful")
        print(f"  - Action logits shape: {action_logits.shape}")
        print(f"  - Value shape: {value.shape}")
        print(f"  - Valid mask shape: {valid_mask.shape}")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_sampling():
    """Test action sampling from the policy."""
    print("\nTesting action sampling...")
    try:
        from trainer.train_sre_rl import SREActorCritic
        
        args = type('Args', (), {
            'device': torch.device('cpu'),
            'num_patches': 10,
            'patch_size': 64
        })()
        
        model = SREActorCritic(args)
        
        # Create dummy inputs
        scene_image = torch.randn(1, 1, 224, 224)
        target_mask = torch.randn(1, 1, 224, 224)
        object_masks = torch.randn(1, 10, 1, 224, 224)
        bboxes = torch.randn(1, 10, 4)
        
        # Sample action
        with torch.no_grad():
            action, action_logprob, value = model.act(
                scene_image, target_mask, object_masks, bboxes
            )
        
        print(f"✓ Action sampling successful")
        print(f"  - Sampled action: {action.item()}")
        print(f"  - Log probability: {action_logprob.item():.4f}")
        print(f"  - State value: {value.item():.4f}")
        return True
    except Exception as e:
        print(f"✗ Action sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_computation():
    """Test reward function computation."""
    print("\nTesting reward computation...")
    try:
        from utils.rl_rewards import compute_episode_reward, compute_target_distance
        
        # Create dummy data
        target_mask = np.ones((224, 224)) * 0
        target_mask[100:120, 100:120] = 1  # Small square
        
        object_masks = np.zeros((5, 224, 224))
        object_masks[0, 90:110, 90:110] = 1   # Close to target
        object_masks[1, 50:70, 50:70] = 1     # Far from target
        object_masks[2, 110:130, 110:130] = 1 # Overlapping target
        
        # Test distance computation
        dist_close = compute_target_distance(target_mask, object_masks, 0)
        dist_far = compute_target_distance(target_mask, object_masks, 1)
        
        print(f"✓ Distance computation successful")
        print(f"  - Distance to close object: {dist_close:.4f}")
        print(f"  - Distance to far object: {dist_far:.4f}")
        
        # Test full reward computation
        reward, components = compute_episode_reward(
            target_mask=target_mask,
            object_masks=object_masks,
            removed_idx=0,
            obs_before=None,
            obs_after=None,
            grasp_success=True,
            collision=False,
            step_num=1,
            max_steps=10,
            target_reached=False
        )
        
        print(f"✓ Reward computation successful")
        print(f"  - Total reward: {reward:.4f}")
        print(f"  - Components: {list(components.keys())}")
        return True
    except Exception as e:
        print(f"✗ Reward computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ppo_memory():
    """Test PPO memory buffer."""
    print("\nTesting PPO memory...")
    try:
        from trainer.train_sre_rl import PPOMemory
        
        memory = PPOMemory()
        
        # Add some transitions
        for i in range(5):
            memory.states.append({'dummy': i})
            memory.actions.append(torch.tensor([i]))
            memory.logprobs.append(torch.tensor([0.1]))
            memory.rewards.append(1.0)
            memory.is_terminals.append(False)
            memory.values.append(torch.tensor([0.5]))
        
        print(f"✓ PPO memory works")
        print(f"  - Memory length: {len(memory)}")
        
        memory.clear()
        print(f"  - After clear: {len(memory)}")
        
        return True
    except Exception as e:
        print(f"✗ PPO memory failed: {e}")
        return False


def main():
    print("=" * 60)
    print("RL Training Setup Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_model_initialization,
        test_forward_pass,
        test_action_sampling,
        test_reward_computation,
        test_ppo_memory,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! RL training setup is ready.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
