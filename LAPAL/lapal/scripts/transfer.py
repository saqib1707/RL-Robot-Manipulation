import pathlib
import argparse
import time
from ruamel.yaml import YAML

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.logger import configure

from lapal.models.sac_ae import SacAeAgent, AdaptAgent
from lapal.utils import utils
from lapal.utils.replay_buffer import ReplayBuffer

input_obs_keys = [
    'robot0_joint_pos_cos',
    'robot0_joint_pos_sin',
    'target_to_robot0_eef_pos',
    ]


def evaluate(env, agent, num_episodes, L, step):

    for i in range(num_episodes):

        true_lat_obs = []
        pred_lat_obs = []
        obs = env.reset()

        done = False
        episode_reward = 0
        while not done:
            obs = np.concatenate([obs[k] for k in input_obs_keys])
            with torch.no_grad():
                lat_obs = agent.ob_encoder(torch.from_numpy(obs).float().to('cuda').unsqueeze(0))
            lat_obs = lat_obs.squeeze(0).cpu().numpy()

            action = agent.sample_action(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward   

        L.record("eval/reward", episode_reward)
    L.dump(step=step)

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--config', help='train config file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    yaml = YAML(typ='safe')
    params = yaml.load(open(args.config))
    data_path = pathlib.Path(__file__).parent.parent.parent / 'data' / time.strftime("%m.%d.%Y")
    logdir = time.strftime("%H-%M-%S") + '_' + params['suffix']
    logdir = data_path / logdir
    params['logdir'] = str(logdir)
    print(params)

    logdir.mkdir(parents=True, exist_ok=True)
    import yaml
    with open(logdir / 'params.yml', 'w') as fp:
        yaml.safe_dump(params, fp, sort_keys=False)

    logger = configure(params['logdir'], ["stdout", "tensorboard"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    src_env = utils.make_robosuite_env(
        params['src_env']['env_name'], 
        robots=params['src_env']['robots'], 
        controller_type=params['src_env']['controller_type']
    )
    src_eval_env = utils.make_robosuite_env(
        params['src_env']['env_name'], 
        robots=params['src_env']['robots'], 
        controller_type=params['src_env']['controller_type'],
    )
    tgt_env = utils.make_robosuite_env(
        params['tgt_env']['env_name'], 
        robots=params['tgt_env']['robots'], 
        controller_type=params['tgt_env']['controller_type']
    )
    tgt_eval_env = utils.make_robosuite_env(
        params['tgt_env']['env_name'], 
        robots=params['tgt_env']['robots'], 
        controller_type=params['tgt_env']['controller_type'],
    )


    ########################################
    # Source domain
    ########################################
    obs = src_env.reset()
    obs_dim = np.concatenate([obs[k] for k in input_obs_keys], axis=-1).shape[0]
    action_dim = src_env.action_dim
    params['src_env']['obs_dim'] = obs_dim
    params['action_dim'] = action_dim
    # Fill source replay buffer
    src_replay_buffer = ReplayBuffer(
        obs_shape=(obs_dim,),
        action_shape=(action_dim,),
        capacity=100000,
        batch_size=256,
        device=device,
    )
    demo_dir = pathlib.Path(params['src_env']['demo_dir'])
    demo_paths = utils.load_episodes(demo_dir, input_obs_keys)
    src_replay_buffer.add_rollouts(demo_paths)

    obs_shape = (obs_dim,)
    action_shape = (action_dim,)
    params['ob_dim'] = obs_shape[0]
    params['ac_dim'] = action_shape[0]
    ob_encoder_params = utils.AttrDict(
        type=params['ob_ae']['type'],
        input_dim=obs_shape[0],
        feature_dim=params['ob_ae']['feature_dim'],
        n_layers=3,
        hidden_dim=512,
        activation='relu',
        output_activation='identity',
        lr=params['ob_ae']['lr'],
        update_freq=params['ob_ae']['update_freq'],
    )
    actor_critic_params = utils.AttrDict(
        lr=params['actor_critic']['lr'], 
        n_layers=params['actor_critic']['n_layers'], 
        hidden_dim=params['actor_critic']['hidden_dim'],
        update_freq=params['actor_critic']['update_freq'],
        tau=0.005,
    )
    disc_params = utils.AttrDict(
        reward_type=params['discriminator']['reward_type'], 
        n_layers=params['discriminator']['n_layers'],
        hidden_dim=params['discriminator']['hidden_dim'],
        activation=params['discriminator']['activation'],
        spectral_norm=params['discriminator']['spectral_norm'],
        lr=params['discriminator']['lr'],
        update_freq=params['discriminator']['update_freq'],
    )
    
    src_agent = SacAeAgent(
        obs_shape,
        action_shape,
        device=device,
        ob_encoder_params=ob_encoder_params,
        actor_critic_params=actor_critic_params,
        disc_params=disc_params,
    )
    src_agent.load(params['src_env']['trained_model'])

    evaluate(src_eval_env, src_agent, 1, logger, 0)


    ########################################
    # Adaptation in target domain
    ########################################

    obs = tgt_env.reset()
    obs_dim = np.concatenate([obs[k] for k in input_obs_keys], axis=-1).shape[0]
    params['tgt_env']['obs_dim'] = obs_dim
    params['action_dim'] = action_dim
    # Fill source replay buffer
    tgt_replay_buffer = ReplayBuffer(
        obs_shape=(obs_dim,),
        action_shape=(action_dim,),
        capacity=100000,
        batch_size=256,
        device=device,
    )
    demo_dir = pathlib.Path(params['tgt_env']['demo_dir'])
    tgt_replay_buffer.add_rollouts(demo_paths)

    # demo_dir = pathlib.Path(f"human_demonstrations/{params['tgt_env']['env_name']}/{params['tgt_env']['robots']}/{params['tgt_env']['controller_type']}_random")
    # demo_paths = utils.load_episodes(demo_dir, input_obs_keys)
    # tgt_replay_buffer.add_rollouts(demo_paths)

    import ipdb; ipdb.set_trace()

    tgt_agent = AdaptAgent(
        (obs_dim,),
        (action_dim,),
        device=device,
        ob_encoder_params=ob_encoder_params,
        actor_critic_params=actor_critic_params,
        disc_params=disc_params,
    )
    tgt_agent.load(params['src_env']['trained_model'], exclude_ob_encoder=True)
    # train_ob_encoder_online(tgt_env, tgt_eval_env, tgt_agent, tgt_replay_buffer)
    # evaluate(tgt_eval_env, tgt_agent, 1, logger, 0)

    # tgt_agent.ob_encoder.apply(utils.weight_init)
    # evaluate(tgt_eval_env, tgt_agent, 1, logger, 0)


    for step in range(10000):
        tgt_agent.update_transfer(tgt_replay_buffer, src_agent, src_replay_buffer, logger)
        
        if (step+1) % 100 == 0:
            evaluate(tgt_eval_env, tgt_agent, 1, logger, step)
        logger.dump(step=step)

    tgt_agent.save("")


if __name__ == '__main__':
    main()