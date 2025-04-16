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
from lapal.utils.replay_buffer import LatReplayBuffer

input_obs_keys = [
    'robot0_joint_pos_cos',
    'robot0_joint_pos_sin',
    'target_to_robot0_eef_pos',
    ]


latent_obs_keys = [
    'robot0_eef_pos',
    'target_to_robot0_eef_pos'
]


def evaluate(env, agent, num_episodes, L, step):

    for i in range(num_episodes):

        true_lat_obs = []
        pred_lat_obs = []
        obs = env.reset()

        done = False
        episode_reward = 0
        while not done:
            true_lat_obs.append(np.concatenate([obs[k] for k in latent_obs_keys], axis=-1))
            
            obs = np.concatenate([obs[k] for k in input_obs_keys])
            with torch.no_grad():
                lat_obs = agent.ob_encoder(torch.from_numpy(obs).float().to('cuda').unsqueeze(0))
            lat_obs = lat_obs.squeeze(0).cpu().numpy()
            pred_lat_obs.append(lat_obs)

            action = agent.sample_action(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward   


        true_lat_obs = np.array(true_lat_obs)
        pred_lat_obs = np.array(pred_lat_obs)

        L.record("eval/reward", episode_reward)
        L.record("eval/lat_diff_mse", np.mean((true_lat_obs - pred_lat_obs) ** 2))

    L.dump(step=step)

def train_ob_encoder_online(env, eval_env, agent, replay_buffer):

    ob_encoder_opt = torch.optim.Adam(agent.ob_encoder.parameters(), lr=1e-5)
    
    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    for step in range(20000):
        if done:       

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1   
            # logger.record('train/episode', episode)

        # run training update
        # agent_obs, _, _, _, _, true_lat_obs = replay_buffer.sample() 
        # lat_obs = agent.ob_encoder(agent_obs)
        # loss = F.mse_loss(lat_obs, true_lat_obs)

        # ob_encoder_opt.zero_grad()
        # loss.backward()
        # ob_encoder_opt.step()

        # sample action for data collection        
        agent_obs = np.concatenate([obs[k] for k in input_obs_keys])
        lat_obs = np.concatenate([obs[k] for k in latent_obs_keys])
        # action = agent.sample_action(agent_obs)
        action = np.random.uniform(low=-1, high=1, size=7)
        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 
        episode_reward += reward

        agent_next_obs = np.concatenate([next_obs[k] for k in input_obs_keys])
        lat_next_obs = np.concatenate([next_obs[k] for k in latent_obs_keys])
        replay_buffer.add(agent_obs, action, reward, agent_next_obs, done_bool, lat_obs, lat_next_obs) 

        obs = next_obs
        episode_step += 1


def update_idm(src_agent, tgt_agent, src_replay_buffer, tgt_replay_buffer, 
    disc, disc_opt, ob_encoder_opt, L, step):
    
    src_obs, _, _, _, _, _ = src_replay_buffer.sample()
    tgt_obs, tgt_action, _, tgt_next_obs, _, tgt_true_lat_obs = tgt_replay_buffer.sample()

    # Update discriminator
    with torch.no_grad():
        src_lat_obs = src_agent.ob_encoder(src_obs)
        tgt_lat_obs = tgt_agent.ob_encoder(tgt_obs)
    src_logit = disc(src_lat_obs)
    src_prob = torch.sigmoid(src_logit)
    tgt_logit = disc(tgt_lat_obs)
    tgt_prob = torch.sigmoid(tgt_logit)

    src_loss = F.binary_cross_entropy(src_prob, torch.ones_like(src_prob))
    tgt_loss = F.binary_cross_entropy(tgt_prob, torch.zeros_like(tgt_prob))
    disc_loss = src_loss + tgt_loss
    # disc_loss = (tgt_logit - src_logit).mean()
    L.record("disc/src_loss", src_loss.item())
    L.record("disc/tgt_loss", tgt_loss.item())
    L.record("disc/disc_loss", disc_loss.item())

    L.record("disc/lat_diff", F.mse_loss(tgt_lat_obs, tgt_true_lat_obs).item())

    # alpha = torch.rand(src_lat_obs.shape[0], 1, device=device)
    # alpha = alpha.expand(src_lat_obs.size())    

    # # Need to set requires_grad to True to run autograd.grad
    # interpolates = (alpha * src_lat_obs + (1 - alpha) * tgt_lat_obs).detach().requires_grad_(True)  

    # # Calculate gradient penalty
    # discr_interpolates = disc(interpolates)   

    # gradients = torch.autograd.grad(
    #     outputs=discr_interpolates, inputs=interpolates,
    #     grad_outputs=torch.ones(discr_interpolates.size(), device=device),
    #     create_graph=True, retain_graph=True, only_inputs=True)[0]
    # gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    # disc_loss += gradient_penalty * 1000

    disc_opt.zero_grad()
    disc_loss.backward()
    disc_opt.step()

    # L.record("gradient_penalty", gradient_penalty.item(), exclude=exclude)
    L.record("disc/src_acc", (src_prob > 0.5).float().mean().item())
    L.record("disc/tgt_acc", (tgt_prob < 0.5).float().mean().item())
    L.record("disc/src_logit", src_logit.mean().item())
    L.record("disc/tgt_logit", tgt_logit.mean().item())


    # Update generator (ob_encoder)

    # NOTE: Minimize the loss that is against the favor of discriminator!
    # gen_loss = -disc(tgt_agent.ob_encoder(tgt_obs)).mean() 
    tgt_lat_obs = tgt_agent.ob_encoder(tgt_obs)
    tgt_logit = disc(tgt_lat_obs)
    tgt_prob = torch.sigmoid(tgt_logit)
    gen_loss = F.binary_cross_entropy(tgt_prob, torch.ones_like(tgt_prob))
    
    ob_encoder_opt.zero_grad()
    gen_loss.backward()
    ob_encoder_opt.step()
    L.record("gen/gen_loss", gen_loss.item())

    # Inverse dynamics loss
    lat_obs = tgt_agent.ob_encoder(tgt_obs)
    lat_next_obs = tgt_agent.ob_encoder(tgt_next_obs)
    pred_action = tgt_agent.inv_dyn(torch.cat([lat_obs, lat_next_obs], dim=-1))
    inv_loss = F.mse_loss(pred_action, tgt_action)

    # ob_encoder_opt.zero_grad()
    # inv_loss.backward()
    # ob_encoder_opt.step()

    L.record("gen/inv_loss", inv_loss.item())
    # L.dump(step=step)

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
        offscreen_render=True,
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
        offscreen_render=True,
    )


    ########################################
    # Source domain
    ########################################
    obs = src_env.reset()
    obs_dim = np.concatenate([obs[k] for k in input_obs_keys], axis=-1).shape[0]
    lat_obs_dim = np.concatenate([obs[k] for k in latent_obs_keys], axis=-1).shape[0]
    action_dim = src_env.action_dim
    params['src_env']['obs_dim'] = obs_dim
    params['src_env']['lat_obs_dim'] = lat_obs_dim
    params['action_dim'] = action_dim
    # Fill source replay buffer
    src_replay_buffer = LatReplayBuffer(
        obs_shape=(obs_dim,),
        action_shape=(action_dim,),
        capacity=100000,
        batch_size=256,
        device=device,
        lat_obs_shape=(lat_obs_dim,)
    )
    demo_dir = pathlib.Path(f"demonstrations/{params['src_env']['env_name']}/{params['src_env']['robots']}/{params['src_env']['controller_type']}")
    demo_paths = utils.load_episodes(demo_dir, input_obs_keys, latent_obs_keys)
    src_replay_buffer.add_rollouts(demo_paths)

    # demo_dir = pathlib.Path(f"human_demonstrations/{params['src_env']['env_name']}/{params['src_env']['robots']}/{params['src_env']['controller_type']}_random")
    # demo_paths = utils.load_episodes(demo_dir, input_obs_keys, latent_obs_keys)
    # src_replay_buffer.add_rollouts(demo_paths)

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
    model_dir = "./data/05.01.2023/14-42-59_Reach_OSC_POSE_SAC_OSC/models/step_0050000"
    src_agent.load(model_dir, exclude_ob_encoder=True)

    # train_ob_encoder_online(src_env, src_eval_env, src_agent, src_replay_buffer)
    evaluate(src_eval_env, src_agent, 1, logger, 0)


    ########################################
    # Adaptation in target domain
    ########################################

    obs = tgt_env.reset()
    obs_dim = np.concatenate([obs[k] for k in input_obs_keys], axis=-1).shape[0]
    lat_obs_dim = np.concatenate([obs[k] for k in latent_obs_keys], axis=-1).shape[0]
    action_dim = src_env.action_dim
    params['tgt_env']['obs_dim'] = obs_dim
    params['tgt_env']['lat_obs_dim'] = lat_obs_dim
    params['action_dim'] = action_dim
    # Fill source replay buffer
    tgt_replay_buffer = LatReplayBuffer(
        obs_shape=(obs_dim,),
        action_shape=(action_dim,),
        capacity=100000,
        batch_size=256,
        device=device,
        lat_obs_shape=(lat_obs_dim,)
    )
    # demo_dir = pathlib.Path(f"demonstrations/{params['tgt_env']['env_name']}/{params['tgt_env']['robots']}/{params['tgt_env']['controller_type']}")
    # demo_paths = utils.load_episodes(demo_dir, input_obs_keys, latent_obs_keys)
    # tgt_replay_buffer.add_rollouts(demo_paths)

    demo_dir = pathlib.Path(f"human_demonstrations/{params['tgt_env']['env_name']}/{params['tgt_env']['robots']}/{params['tgt_env']['controller_type']}_random")
    demo_paths = utils.load_episodes(demo_dir, input_obs_keys, latent_obs_keys)
    tgt_replay_buffer.add_rollouts(demo_paths)


    tgt_agent = AdaptAgent(
        (obs_dim,),
        (action_dim,),
        device=device,
        ob_encoder_params=ob_encoder_params,
        actor_critic_params=actor_critic_params,
        disc_params=disc_params,
    )
    tgt_agent.load(model_dir, exclude_ob_encoder=True)
    # train_ob_encoder_online(tgt_env, tgt_eval_env, tgt_agent, tgt_replay_buffer)
    # evaluate(tgt_eval_env, tgt_agent, 1, logger, 0)

    # tgt_agent.ob_encoder.apply(utils.weight_init)
    # evaluate(tgt_eval_env, tgt_agent, 1, logger, 0)

    for step in range(10000):
        tgt_agent.update_inv_dyn(tgt_replay_buffer, logger)
        logger.dump(step=step)

    for step in range(10000):
        tgt_agent.update_transfer(tgt_replay_buffer, src_replay_buffer, logger)
        
        if (step+1) % 100 == 0:
            evaluate(tgt_eval_env, tgt_agent, 1, logger, step)
        logger.dump(step=step)

    tgt_agent.save("")

    # import ipdb; ipdb.set_trace()

    # Set up adaptation
    # L = logger
    agent = tgt_agent
    env = tgt_env
    eval_env = tgt_eval_env
    agent_replay_buffer = tgt_replay_buffer


    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    for step in range(params['total_timesteps']):
        if done:
            if step > 0:
                logger.record('train/duration', time.time() - start_time)
                start_time = time.time()
                logger.dump(step=step) 
            if step % params['evaluation']['interval'] == 0:
                print(f"Evaluating at step {step}")
                logger.record('eval/episode', episode)
                evaluate(eval_env, agent, 1, logger, step) 

            # print(agent_replay_buffer.idx, src_replay_buffer.idx)
            logger.record('train/episode_reward', episode_reward)               

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1   
            logger.record('train/episode', episode) 

        # run training update
        agent.update_transfer(agent_replay_buffer, src_replay_buffer, logger)

        # sample action for data collection       
        agent_obs = np.concatenate([obs[k] for k in input_obs_keys]) 
        action = agent.sample_action(agent_obs)
        next_obs, reward, done, _ = env.step(action)    

        # allow infinit bootstrap
        done_bool = 0 
        episode_reward += reward    

        # Fill replay buffer
        agent_next_obs = np.concatenate([next_obs[k] for k in input_obs_keys]) 
        lat_obs = np.concatenate([obs[k] for k in latent_obs_keys])
        lat_next_obs = np.concatenate([next_obs[k] for k in latent_obs_keys])
        agent_replay_buffer.add(agent_obs, action, reward, agent_next_obs, done_bool, lat_obs, lat_next_obs)   

        obs = next_obs
        episode_step += 1



    # for i in range(params['target_train_steps']):

    #     src_obs, src_action, _, src_next_obs, _ = src_replay_buffer.sample()

    #     with torch.no_grad():
    #         src_lat_obs = src_agent.critic.ob_encoder(src_obs)
    #         src_lat_next_obs = src_agent.critic.ob_encoder(src_next_obs)
        
    #     src_logit, src_prob = tgt_agent.disc(src_obs, src_action)
    #     tgt_obs, tgt_action, _, tgt_next_obs, _ = tgt_replay_buffer.sample()
    #     tgt_logit, tgt_prob = tgt_agent.disc(tgt_obs, tgt_action)
    #     disc_loss = (src_logit - tgt_logit).mean()
    #     disc_optimizer.zero_grad()
    #     disc_loss.backward()
    #     disc_optimizer.step()


    #     # tgt_obs, tgt_action, _, tgt_next_obs, _ = tgt_replay_buffer.sample()
    #     tgt_logit, tgt_prob = tgt_agent.disc(tgt_obs, tgt_action)
        
    #     tgt_lat_obs = tgt_agent.critic.ob_encoder(tgt_obs)
    #     tgt_lat_next_obs = tgt_agent.critic.ob_encoder(tgt_next_obs)
    #     pred_action = tgt_agent.inv_dyn(torch.cat([tgt_lat_obs, tgt_lat_next_obs], dim=-1))
    #     inv_dyn_loss = F.mse_loss(pred_action, tgt_action)

    #     loss = tgt_logit.mean() + inv_dyn_loss

    #     ob_encoder_optimizer.zero_grad()
    #     loss.backward()
    #     ob_encoder_optimizer.step()

    #     exclude = None if (i+1) % 1000 == 0 else 'stdout'
    #     logger.record(f'target_train/disc_loss', disc_loss.item(), exclude=exclude)
    #     logger.record(f'target_train/inv_loss', inv_dyn_loss.item(), exclude=exclude)
    #     logger.record(f'target_train/src_acc', torch.mean((src_prob < 0.5).float()).item(), exclude=exclude)
    #     logger.record(f'target_train/tgt_acc', torch.mean((tgt_prob > 0.5).float()).item(), exclude=exclude)
    #     logger.record(f'target_train/tgt_logit', tgt_logit.mean().item(), exclude=exclude)
    #     logger.dump(step=i) 


    # for i in range(params['target_train_steps']):
    #     _, src_action, _, _, _, src_lat_obs, _ = src_replay_buffer.sample(
    #         return_latent_obs=True)

    #     tgt_obs, tgt_action, _, tgt_next_obs, _ = tgt_replay_buffer.sample()


    #     # Discriminator update
    #     tgt_lat_obs = tgt_policy.critic.ob_encoder(tgt_obs)
    #     src_labels = torch.zeros((src_lat_obs.shape[0],1), device=ptu.device)
    #     tgt_labels = torch.ones((tgt_lat_obs.shape[0],1), device=ptu.device)
    #     labels = torch.cat([src_labels, tgt_labels], dim=0)

    #     logit = disc(torch.cat([src_lat_obs, tgt_lat_obs], dim=0))
    #     prob = torch.sigmoid(logit)
    #     disc_loss = F.binary_cross_entropy(prob, labels)

    #     disc_optimizer.zero_grad()
    #     disc_loss.backward()
    #     disc_optimizer.step()

    #     src_logit, tgt_logit = torch.chunk(logit, 2, dim=0)
    #     src_prob, tgt_prob = torch.chunk(prob, 2, dim=0)
    #     src_acc = torch.mean((src_prob < 0.5).float())
    #     tgt_acc = torch.mean((tgt_prob > 0.5).float())
    #     metrics = {
    #         'disc_loss': disc_loss.item(),
    #         'src_acc': src_acc.item(),
    #         'tgt_acc': tgt_acc.item(),
    #         'src_logit': src_logit.mean().item(),
    #         'tgt_logit': tgt_logit.mean().item(),
    #     }

    #     # Dynamics consistency update
    #     # Forward dynamics loss
    #     tgt_lat_obs = tgt_policy.critic.ob_encoder(tgt_obs)
    #     tgt_lat_next_obs = tgt_policy.critic.ob_encoder(tgt_next_obs)
    #     dyn_metrics = train_dyn(tgt_lat_obs, tgt_action, tgt_lat_next_obs, 
    #         forward_dyn, inverse_dyn, dyn_optimizer)

    #     exclude = None if (i+1) % 1000 == 0 else 'stdout'
    #     metrics.update(dyn_metrics)
    #     for k, v in metrics.items():
    #         logger.record(f'target_train/{k}', v, exclude=exclude)
    #     logger.dump(step=i) 

    # evaluate(tgt_env, tgt_agent, input_obs_keys, 10)


if __name__ == '__main__':
    main()