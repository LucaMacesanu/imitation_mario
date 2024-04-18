import numpy as np
import gymnasium as gym
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.data import rollout
from imitation.data.types import Transitions

from imitation.algorithms import bc

#our version

#create our environment
env = make_vec_env(
    "SuperMarioBros-v3",
    rng = np.random.default_rng(),
    post_wrappers = [
        lambda env, _: RolloutInfoWrapper(env)
    ],
)
'''
#expert policy
expert = load_policy(
    "ppo-huggingface",
    organization = "HumanCompatibleAI",
    env_name = "SuperMarioBros-v3",
    venv = env,
)






reward, _ = evaluate_policy(expert, env, 10)
print(reward)


rng = np.random.default_rng()
rollouts = rollout.rollout(
    expert,
    env,
    rollout.make_sample_until(min_timesteps=None, min_episodes=50),
    rng=rng,
)
transitions = rollout.flatten_trajectories(rollouts)



print(
    f"""The `rollout` function generated a list of {len(rollouts)} {type(rollouts[0])}.
After flattening, this list is turned into a {type(transitions)} object containing {len(transitions)} transitions.
The transitions object contains arrays for: {', '.join(transitions.__dict__.keys())}."
"""
)
'''
#instead of using expert policy we use imitation data

demonstrations = Transitions(
    obs = observations, #observation e.g. frame data maybe reward
    acts = actions, #action state
    next_obs = np.roll(observations, -1, axis = 0)
    dones = dones, #finished or not (on flag)
    infos = []
)
rng = np.random.default_rng()
#train w BC

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=demonstrations,
    rng=rng,
)

reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
print(f"Reward before training: {reward_before_training}")

bc_trainer.train(n_epochs=1)
reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
print(f"Reward after training: {reward_after_training}")
