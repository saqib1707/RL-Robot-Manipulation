import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lapal.models.encoder import make_encoder
from lapal.models.decoder import make_decoder
from lapal.models.actor_critic import Actor, Critic
from lapal.models.discriminator import Discriminator
from lapal.utils import utils

class SacAeAgent:
    """SAC+AE algorithm."""
    def __init__(
        self,
        obs_shape,
        act_shape,
        device,
        discount=0.99,
        init_temperature=0.1,
        obs_encoder_params=None,
        act_encoder_decoder_params=None,
        actor_critic_params=None,
        disc_params=None,
    ):
        self.device = device
        self.discount = discount
        self.update_freq = actor_critic_params.update_freq
        self.obs_encoder_update_freq = obs_encoder_params.update_freq 
        self.act_encoder_update_freq = act_encoder_decoder_params.update_freq
        self.act_enc_dec_kl_coef = act_encoder_decoder_params.kl_coef
        self.disc_update_freq = disc_params.update_freq

        ## Observation encoder
        self.obs_encoder = make_encoder(obs_encoder_params).to(device)

        # Action encoder and decoder
        self.act_encoder = make_encoder(act_encoder_decoder_params).to(device)
        self.act_decoder = make_decoder(act_encoder_decoder_params).to(device)

        # Use latent observation/action dim for actor and critic
        obs_feat_dim = self.obs_encoder.feature_dim
        act_feat_dim = self.act_encoder.feature_dim

        self.inv_dyn = utils.build_mlp(
            self.obs_encoder.feature_dim*2, 
            act_shape[0], 
            obs_encoder_params.n_layers, 
            obs_encoder_params.hidden_dim,
            activation=obs_encoder_params['activation'],
            output_activation=obs_encoder_params['output_activation'],
        ).to(device)
        self.fwd_dyn = utils.build_mlp(
            self.obs_encoder.feature_dim+act_shape[0], 
            self.obs_encoder.feature_dim, 
            obs_encoder_params.n_layers, 
            obs_encoder_params.hidden_dim,
            activation=obs_encoder_params['activation'],
            output_activation='identity',
        ).to(device)
        if self.obs_encoder_update_freq > 0:
            self.obs_encoder_opt = torch.optim.Adam(
                list(self.obs_encoder.parameters())+list(self.inv_dyn.parameters())+list(self.fwd_dyn.parameters()),
                lr=obs_encoder_params.lr
            )
        if self.act_encoder_update_freq > 0:
            self.act_enc_dec_opt = torch.optim.Adam(
                list(self.act_encoder.parameters()) + list(self.act_decoder.parameters()),
                lr=act_encoder_decoder_params.lr,
            )

        # modules = [self.obs_encoder, self.act_encoder, self.act_decoder, self.inv_dyn]
        # p_list = [list(m.parameters()) for m in modules]
        # params = []
        # for p in p_list:
        #     params += p
        # self.dyn_cons_opt = torch.optim.Adam(params, lr=obs_encoder_params.lr)


        # Actor-critic
        self.actor = Actor(obs_feat_dim, act_feat_dim, actor_critic_params).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_critic_params.lr)
        
        self.critic = Critic(obs_feat_dim, act_feat_dim, actor_critic_params).to(device)
        # Optimize ob_encoder parameters with critic loss
        self.critic_opt = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.obs_encoder.parameters()) + list(self.act_encoder.parameters()), 
            lr=actor_critic_params.lr
        )

        self.critic_tau = actor_critic_params.tau
        self.critic_target = Critic(obs_feat_dim, act_feat_dim, actor_critic_params).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.disc = Discriminator(obs_feat_dim, act_feat_dim, disc_params).to(device)
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=disc_params.lr)

        self.log_alpha = torch.tensor(np.log(init_temperature)).float().to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=actor_critic_params.lr)

        # set target entropy to negative latent action dimension
        self.target_entropy = -float(act_feat_dim)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def sample_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().to(self.device)
            obs = obs.unsqueeze(0)
        
            lat_obs = self.obs_encoder(obs)
            outputs = self.actor(lat_obs)
        
        lat_act = outputs['mu'] if deterministic else outputs['pi']
        act = self.act_decoder(lat_act, cond=obs)
        return act.cpu().data.numpy().flatten()

    def update_actor_critic(self, obs, act, reward, next_obs, not_done, L):

        # Optimize log_alpha
        with torch.no_grad():
            lat_obs = self.obs_encoder(obs)
            outputs = self.actor(lat_obs, compute_log_pi=True)
        log_pi = outputs['log_pi']
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        L.record('train_alpha/alpha_loss', alpha_loss.item())
        L.record('train_alpha/alpha_value', self.alpha.item())
        L.record('train_alpha/log_pi', log_pi.mean().item())
        self.log_alpha_opt.zero_grad()
        alpha_loss.backward()
        self.log_alpha_opt.step()

        # Optimize critic
        with torch.no_grad():
            lat_next_obs = self.obs_encoder(next_obs)
            outputs = self.actor(lat_next_obs, compute_log_pi=True)
            pi, log_pi = outputs['pi'], outputs['log_pi']
            target_Q1, target_Q2 = self.critic_target(lat_next_obs, pi)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        lat_obs = self.obs_encoder(obs)
        lat_act = self.act_encoder(act, cond=obs)
        current_Q1, current_Q2 = self.critic(lat_obs, lat_act)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        L.record('train_critic/critic_loss', critic_loss.item())
        L.record('train_critic/Q1', current_Q1.mean().item())
        L.record('train_critic/target_Q1', target_Q1.mean().item())
        L.record('train_critic/target_V', target_V.mean().item())
        L.record('train_critic/target_Q', target_Q.mean().item())
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Optimize actor
        # Do not compute gradient through observation encoder with actor loss
        with torch.no_grad():
            lat_obs = self.obs_encoder(obs)
        outputs = self.actor(lat_obs, compute_log_pi=True)
        pi, log_pi, log_std = outputs['pi'], outputs['log_pi'], outputs['log_std']
        actor_Q1, actor_Q2 = self.critic(lat_obs, pi)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        L.record('train_actor/actor_loss', actor_loss.item())
        L.record('train_actor/target_entropy', self.target_entropy)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        L.record('train_actor/entropy', entropy.mean().item())
        L.record('train_actor/actor_log_pi', log_pi.mean().item())
        L.record('train_actor/actor_log_std', log_std.mean().item())
        L.record('train_actor/pi_norm', (pi**2).mean().item())
        L.record('train_actor/mu_norm', (outputs['mu']**2).mean().item())
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)


    def update_obs_encoder(self, obs, act, next_obs, L):
        """
        Dynamics consistency loss from https://arxiv.org/pdf/2112.08526.pdf
        """

        lat_obs = self.obs_encoder(obs)
        lat_next_obs = self.obs_encoder(next_obs)
        pred_act = self.inv_dyn(torch.cat([lat_obs, lat_next_obs], dim=-1))
        inv_loss = F.mse_loss(pred_act, act)

        pred_lat_next_obs = self.fwd_dyn(torch.cat([lat_obs, act], dim=-1))
        fwd_loss = F.mse_loss(pred_lat_next_obs, lat_next_obs)

        loss = fwd_loss + inv_loss
        self.obs_encoder_opt.zero_grad()
        loss.backward()
        self.obs_encoder_opt.step()

        L.record('train_obs/obs_loss', loss.item())
        L.record('train_obs/fwd_loss', fwd_loss.item())
        L.record('train_obs/inv_loss', inv_loss.item())

    def update_act_encoder_decoder(self, obs, act, L):
        """
        CVAE loss for training action encoder and decoder
        """
        outputs = self.act_encoder.compute_latent_dist_and_sample(act, cond=obs)
        z, mu, std = outputs['z'], outputs['mu'], outputs['std']
        pred_act = self.act_decoder(z, cond=obs)

        recon_loss = F.mse_loss(pred_act, act)
        # kld_loss = -0.5 * torch.mean(torch.sum(1 + torch.log(std**2) - mu**2 - std**2, dim=1))
        kld_loss = -0.5 * (1 + torch.log(std.pow(2)) - mu.pow(2) - std.pow(2)).mean()
        loss = recon_loss + self.act_enc_dec_kl_coef * kld_loss

        self.act_enc_dec_opt.zero_grad()
        loss.backward()
        self.act_enc_dec_opt.step()

        L.record('train_act/act_loss', loss.item())
        L.record('train_act/recon', recon_loss.item())
        L.record('train_act/kl_div', kld_loss.item())
        L.record('train_act/act_z_norm', (z**2).mean().item())
        L.record('train_act/act_mu_norm', (mu**2).mean().item())
        L.record('train_act/act_std', std.mean().item())


    def update_dyn_cons(self, obs, act, next_obs, L):
        """
        Dynamics consistency loss from https://arxiv.org/pdf/2112.08526.pdf
        """

        lat_obs = self.obs_encoder(obs)
        lat_next_obs = self.obs_encoder(next_obs)
        pred_lat_act = self.inv_dyn(torch.cat([lat_obs, lat_next_obs], dim=-1))
        pred_act = self.act_decoder(pred_lat_act)
        inv_loss = F.mse_loss(pred_act, act)
        # import ipdb; ipdb.set_trace()

        # lat_act = self.act_encoder(act)        
        # pred_lat_next_obs = lat_obs + lat_act
        # fwd_loss = F.mse_loss(pred_lat_next_obs, lat_next_obs)

        # loss = fwd_loss + inv_loss
        self.dyn_cons_opt.zero_grad()
        inv_loss.backward()
        self.dyn_cons_opt.step()

        # L.record('train_dyn_cons/dyn_loss', loss.item())
        # L.record('train_dyn_cons/fwd_loss', fwd_loss.item())
        L.record('train_dyn_cons/inv_loss', inv_loss.item())

    def update_discriminator(self, obs, act, demo_obs, demo_act, L):
        """
        Do not compute gradient through observation and action encoders
        with discriminator loss
        """
        with torch.no_grad():
            lat_obs = self.obs_encoder(obs)
            demo_lat_obs = self.obs_encoder(demo_obs)
            lat_act = self.act_encoder(act, cond=obs)
            demo_lat_act = self.act_encoder(demo_act, cond=demo_obs)

        agent_logit, agent_prob = self.disc(lat_obs, lat_act)
        demo_logit, demo_prob = self.disc(demo_lat_obs, demo_lat_act)

        agent_loss = F.binary_cross_entropy(agent_prob, torch.ones_like(agent_prob))
        demo_loss = F.binary_cross_entropy(demo_prob, torch.zeros_like(demo_prob))
        loss = agent_loss + demo_loss

        self.disc_opt.zero_grad()
        loss.backward()
        self.disc_opt.step()

        L.record('train_disc/disc_loss', loss.item())
        L.record('train_disc/expert_acc', (demo_prob < 0.5).float().mean().item())
        L.record('train_disc/agent_acc', (agent_prob > 0.5).float().mean().item())
        L.record('train_disc/expert_logit', demo_logit.mean().item())
        L.record('train_disc/agent_logit', agent_logit.mean().item())

    def update(self, replay_buffer, demo_buffer, L, step):
        obs, act, rew, next_obs, not_done = replay_buffer.sample()
        L.record('train/batch_reward', rew.mean().item())

        if self.obs_encoder_update_freq > 0 and step % self.obs_encoder_update_freq == 0:
            self.update_obs_encoder(obs, act, next_obs, L)
            # self.update_dyn_cons(obs, act, next_obs, L)

        if self.act_encoder_update_freq > 0 and step % self.act_encoder_update_freq == 0:
            self.update_act_encoder_decoder(obs, act, L)

        if self.disc.reward_type is not None and step % self.disc_update_freq == 0:
            demo_obs, demo_act, _, _, _ = demo_buffer.sample()
            self.update_discriminator(obs, act, demo_obs, demo_act, L)

        if step % self.update_freq == 0:
            if self.disc.reward_type is not None:
                with torch.no_grad():
                    lat_obs = self.obs_encoder(obs)
                    lat_act = self.act_encoder(act, cond=obs)
                    rew = self.disc.reward(lat_obs, lat_act)
                L.record('train/learned_reward', rew.mean().item())
            self.update_actor_critic(obs, act, rew, next_obs, not_done, L)


    def save(self, model_dir):
        # torch.save(self.obs_encoder.state_dict(), f'{model_dir}/obs_encoder.pt')
        # torch.save(self.inv_dyn.state_dict(), f'{model_dir}/inv_dyn.pt')
        # torch.save(self.critic.state_dict(), f'{model_dir}/critic.pt')
        # torch.save(self.actor.state_dict(), f'{model_dir}/actor.pt')
        # torch.save(self.disc.state_dict(), f'{model_dir}/disc.pt')
        pass

    def load(self, model_dir, exclude_obs_encoder=False):
        """
        Observation encoder should not be loaded 
        if adapting agent is loading from source target
        """
        if not exclude_obs_encoder:
            self.obs_encoder.load_state_dict(torch.load(f'{model_dir}/obs_encoder.pt'))
        self.inv_dyn.load_state_dict(torch.load(f'{model_dir}/inv_dyn.pt'))
        self.critic.load_state_dict(torch.load(f'{model_dir}/critic.pt'))
        self.critic_target.load_state_dict(torch.load(f'{model_dir}/critic.pt'))
        self.actor.load_state_dict(torch.load(f'{model_dir}/actor.pt'))
        self.disc.load_state_dict(torch.load(f'{model_dir}/disc.pt'))


class AdaptAgent(SacAeAgent):
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        obs_encoder_params=None,
        actor_critic_params=None,
        disc_params=None,
    ):

        super().__init__(
            obs_shape,
            action_shape,
            device,
            obs_encoder_params=obs_encoder_params,
            actor_critic_params=actor_critic_params,
            disc_params=disc_params,
        )

        self.obs_encoder_opt = torch.optim.Adam(
            self.obs_encoder.parameters(), lr=1e-5)

        self.adapt_disc = utils.build_mlp(
            self.obs_encoder.feature_dim + action_shape[0], 
            1, 
            3,
            512,
            activation='tanh',
            output_activation='identity',
            spectral_norm=True,
        ).to(device)

        self.adapt_disc_opt = torch.optim.Adam(
            self.adapt_disc.parameters(), lr=3e-4)


        self.inv_dyn = utils.build_mlp(
            self.obs_encoder.feature_dim*2, 
            action_shape[0], 
            3, 
            512,
            activation='leaky_relu',
            output_activation='tanh',
        ).to(device)

        self.inv_dyn_opt = torch.optim.RMSprop(self.inv_dyn.parameters(), lr=3e-4)


    # def update_adapt_disc(self, lat_obs, src_lat_obs, L):
    #     """
    #     Discriminator tries to separate source and target latent states 
    #     """

    #     agent_logit = self.adapt_disc(lat_obs)
    #     src_logit = self.adapt_disc(src_lat_obs)

    #     agent_prob = torch.sigmoid(agent_logit)
    #     src_prob = torch.sigmoid(src_logit)

    #     agent_loss = F.binary_cross_entropy(agent_prob, torch.zeros_like(agent_prob))
    #     src_loss = F.binary_cross_entropy(src_prob, torch.ones_like(src_prob))
    #     loss = agent_loss + src_loss

    #     self.adapt_disc_opt.zero_grad()
    #     loss.backward()
    #     self.adapt_disc_opt.step()

    #     L.record('train_disc/disc_loss', loss.item())
    #     L.record('train_disc/src_acc', (src_prob > 0.5).float().mean().item())
    #     L.record('train_disc/agent_acc', (agent_prob < 0.5).float().mean().item())
    #     L.record('train_disc/src_logit', src_logit.mean().item())
    #     L.record('train_disc/agent_logit', agent_logit.mean().item())  

    def update_adapt_disc(self, lat_obs, action, src_lat_obs, src_action, L):
        """
        Discriminator tries to separate source and target latent states 
        """

        agent_logit = self.adapt_disc(torch.cat([lat_obs, action], dim=-1))
        src_logit = self.adapt_disc(torch.cat([src_lat_obs, src_action], dim=-1))
        # agent_logit = self.adapt_disc(lat_obs)
        # src_logit = self.adapt_disc(src_lat_obs)

        agent_prob = torch.sigmoid(agent_logit)
        src_prob = torch.sigmoid(src_logit)

        agent_loss = F.binary_cross_entropy(agent_prob, torch.zeros_like(agent_prob))
        src_loss = F.binary_cross_entropy(src_prob, torch.ones_like(src_prob))
        loss = agent_loss + src_loss

        self.adapt_disc_opt.zero_grad()
        loss.backward()
        self.adapt_disc_opt.step()

        L.record('train_disc/disc_loss', loss.item())
        L.record('train_disc/src_acc', (src_prob > 0.5).float().mean().item())
        L.record('train_disc/agent_acc', (agent_prob < 0.5).float().mean().item())
        L.record('train_disc/src_logit', src_logit.mean().item())
        L.record('train_disc/agent_logit', agent_logit.mean().item())  

    def update_obs_encoder(self, obs, action, next_obs, L):
        """
        Update observation encoder with generator loss
        """

        for p in self.adapt_disc.parameters():
            p.requires_grad = False

        # logit = self.adapt_disc(self.obs_encoder(obs))
        lat_obs = self.obs_encoder(obs)
        logit = self.adapt_disc(torch.cat([lat_obs, action], dim=-1))
        prob = torch.sigmoid(logit)
        loss = F.binary_cross_entropy(prob, torch.ones_like(prob))
        self.obs_encoder_opt.zero_grad()
        loss.backward()
        self.obs_encoder_opt.step()
        L.record('train_gen/gen_loss', loss.item())

        pred_action = self.inv_dyn(torch.cat([self.obs_encoder(obs), self.obs_encoder(next_obs)], dim=-1))
        inv_loss = F.mse_loss(pred_action, action)
        self.obs_encoder_opt.zero_grad()
        inv_loss.backward()
        self.obs_encoder_opt.step()

        L.record('train_gen/gen_inv_loss', inv_loss.item())

        for p in self.adapt_disc.parameters():
            p.requires_grad = True

    
    def update_inv_dyn(self, replay_buffer, L):
        """
        Train inverse dynamics model
        """

        _, action, _, _, _, lat_obs, lat_next_obs = replay_buffer.sample()
        pred_action = self.inv_dyn(torch.cat([lat_obs, lat_next_obs], dim=-1))
        loss = F.mse_loss(pred_action, action)
        L.record("train_inv/inv_dyn_loss", loss.item())

        self.inv_dyn_opt.zero_grad()
        loss.backward()
        self.inv_dyn_opt.step()


    def update_transfer(self, replay_buffer, src_agent, src_replay_buffer, L):
        """
        Invariance through latent alignment
        """

        obs, action, reward, next_obs, _, true_lat_obs, _ = replay_buffer.sample()
        L.record('train/batch_reward', reward.mean().item())
        src_obs, src_action, _, _, _, true_src_lat_obs, _ = src_replay_buffer.sample()

        # obs, action, reward, next_obs, _ = replay_buffer.sample()
        # L.record('train/batch_reward', reward.mean().item())
        # src_obs, src_action, src_reward, src_next_obs, _ = src_replay_buffer.sample()

        # Update discriminator
        for _ in range(5):
            with torch.no_grad():
                lat_obs = self.obs_encoder(obs)
                # src_lat_obs = src_agent.obs_encoder(src_obs)
            self.update_adapt_disc(lat_obs, action, true_src_lat_obs, src_action, L)

        # Update generator observation encoder
        self.update_obs_encoder(obs, action, next_obs, L)
        
        lat_pred_loss = F.mse_loss(self.obs_encoder(obs), true_lat_obs)
        L.record('valid/lat_pred_loss', lat_pred_loss.item())

        with torch.no_grad():
            outputs = self.actor(self.obs_encoder(obs), compute_log_pi=True)
            act_pred_loss = F.mse_loss(outputs['mu'], action)
        L.record('valid/act_pred_loss', act_pred_loss.item())

# class AdaptAgent(SacAeAgent):
#     def __init__(
#         self,
#         obs_shape,
#         action_shape,
#         device,
#         ob_encoder_params=None,
#         actor_critic_params=None,
#         disc_params=None,
#     ):

#         super().__init__(
#             obs_shape,
#             action_shape,
#             device,
#             ob_encoder_params=ob_encoder_params,
#             actor_critic_params=actor_critic_params,
#             disc_params=disc_params,
#         )

#         self.ob_encoder_opt = torch.optim.RMSprop(
#             self.ob_encoder.parameters(), lr=1e-6)

#         self.adapt_disc = utils.build_mlp(
#             self.ob_encoder.feature_dim, 
#             1, 
#             disc_params.n_layers,
#             disc_params.hidden_dim,
#             activation=disc_params.activation,
#             output_activation='identity',
#             spectral_norm=False,
#         ).to(device)

#         self.adapt_disc_opt = torch.optim.RMSprop(
#             self.adapt_disc.parameters(), lr=1e-5)

#         self.inv_dyn_opt = torch.optim.RMSprop(self.inv_dyn.parameters(), lr=3e-4)


#     def update_adapt_disc(self, lat_obs, src_lat_obs, L):
#         """
#         Discriminator tries to separate source and target latent states 
#         """

#         agent_logit = self.adapt_disc(lat_obs)
#         src_logit = self.adapt_disc(src_lat_obs)        
#         loss = (agent_logit - src_logit).mean()

#         grad_penalty_coef = 100
#         gradient_penalty = self._calculate_gradient_penalty(src_lat_obs, lat_obs)
#         loss += gradient_penalty * grad_penalty_coef

#         self.adapt_disc_opt.zero_grad()
#         loss.backward()
#         self.adapt_disc_opt.step()

#         # for p in self.adapt_disc.parameters():
#         #     p.data.clamp_(-0.1, 0.1)

#         L.record('train_disc/disc_loss', loss.item())
#         L.record('train_disc/src_acc', (src_logit > 0).float().mean().item())
#         L.record('train_disc/agent_acc', (agent_logit < 0).float().mean().item())
#         L.record('train_disc/src_logit', src_logit.mean().item())
#         L.record('train_disc/agent_logit', agent_logit.mean().item())  
#         L.record('train_disc/gradient_penalty', gradient_penalty.item())

#     def _calculate_gradient_penalty(self, original_latent, adapting_latent):
#         # NOTE: Improved training of WGANs
#         # Taken from https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py
#         alpha = torch.rand(original_latent.shape[0], 1, device=self.device)
#         alpha = alpha.expand(original_latent.size())

#         # Need to set requires_grad to True to run autograd.grad
#         interpolates = (alpha * original_latent + (1 - alpha) * adapting_latent).detach().requires_grad_(True)

#         # Calculate gradient penalty
#         discr_interpolates = self.adapt_disc(interpolates)

#         gradients = torch.autograd.grad(outputs=discr_interpolates, inputs=interpolates,
#                                         grad_outputs=torch.ones(discr_interpolates.size(), device=self.device),
#                                         create_graph=True, retain_graph=True, only_inputs=True)[0]
#         gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
#         return gradient_penalty


#     def update_ob_encoder(self, obs, action, next_obs, L):
#         """
#         Update observation encoder with generator loss
#         """

#         for p in self.adapt_disc.parameters():
#             p.requires_grad = False

#         # NOTE: Minimize the loss that is against the favor of discriminator!
#         loss = - self.adapt_disc(self.ob_encoder(obs)).mean()  

#         pred_action = self.inv_dyn(torch.cat([self.ob_encoder(obs), self.ob_encoder(next_obs)], dim=-1))
#         inv_loss = F.mse_loss(pred_action, action)

#         loss += inv_loss

#         self.ob_encoder_opt.zero_grad()
#         loss.backward()
#         self.ob_encoder_opt.step()

#         L.record('train_gen/gen_loss', loss.item())
#         L.record('train_gen/inv_loss', inv_loss.item())

#         for p in self.adapt_disc.parameters():
#             p.requires_grad = True

#     def update_inv_dyn(self, replay_buffer, L):
#         """
#         Train inverse dynamics model
#         """

#         _, action, _, _, _, lat_obs, lat_next_obs = replay_buffer.sample()
#         pred_action = self.inv_dyn(torch.cat([lat_obs, lat_next_obs], dim=-1))
#         loss = F.mse_loss(pred_action, action)
#         L.record("train/inv_dyn_loss", loss.item())

#         self.inv_dyn_opt.zero_grad()
#         loss.backward()
#         self.inv_dyn_opt.step()



#     def update_transfer(self, replay_buffer, src_replay_buffer, L):
#         """
#         Invariance through latent alignment
#         """

#         obs, action, reward, next_obs, _, true_lat_obs, _ = replay_buffer.sample()
#         L.record('train/batch_reward', reward.mean().item())

#         src_obs, src_action, src_reward, src_next_obs, _, true_src_lat_obs, _ = src_replay_buffer.sample()

#         # Update discriminator

#         for _ in range(5):
#             with torch.no_grad():
#                 lat_obs = self.ob_encoder(obs)
#                 # src_lat_obs = src_agent.ob_encoder(src_obs)
#             self.update_adapt_disc(lat_obs, true_src_lat_obs, L)

#         # Update generator observation encoder
#         self.update_ob_encoder(obs, action, next_obs, L)

#         pred_loss = F.mse_loss(self.ob_encoder(obs), true_lat_obs)
#         L.record('eval/pred_loss', pred_loss.item())


        # Optimize inverse
        # h = self.ob_encoder(obs)
        # next_h = self.ob_encoder(next_obs)

        # pred_action = self.inv_dyn(torch.cat([h, next_h], dim=-1))
        # inv_loss = F.mse_loss(pred_action, action)

        # # self.ob_encoder_opt.zero_grad()
        # # inv_loss.backward()
        # # self.ob_encoder_opt.step()
        # L.record('train/inv_loss', inv_loss.item())
