import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
# from tensorflow.keras.mixed_precision import experimental as prec
import tensorflow.keras.mixed_precision as prec

import tools
import pdb


class RSSM(tools.Module):
  def __init__(self, stoch=30, deter=200, hidden=200, act=tf.nn.elu):
    super().__init__()
    self._activation = act   # ELU
    self._stoch_size = stoch  # 30
    self._deter_size = deter  # 200
    self._hidden_size = hidden  # 200
    self._cell = tfkl.GRUCell(self._deter_size)

  def initial(self, batch_size):
    # print("Inside rssm initial", batch_size)
    dtype = prec.global_policy().compute_dtype  # float32
    return dict(
        mean=tf.zeros([batch_size, self._stoch_size], dtype),
        std=tf.zeros([batch_size, self._stoch_size], dtype),
        stoch=tf.zeros([batch_size, self._stoch_size], dtype),
        deter=self._cell.get_initial_state(None, batch_size, dtype))

  @tf.function
  def observe(self, embed, action, state=None):
    '''
      embed: [128,50,1024]
      action: [128,50,7]
    '''
    # print("inside rssm observe", embed.shape, action.shape, state)
    if state is None:
      # initialize state
      state = self.initial(tf.shape(action)[0])   # {mean:[128,30], std:[128,30], stoch:[128,30], deter:[128,200]}
      # print("state is None:", state) 
    embed = tf.transpose(embed, [1, 0, 2])   # [50,128,1024]
    action = tf.transpose(action, [1, 0, 2])  # [50,128,7]
    # print("after shape:", embed.shape, action.shape)
    post, prior = tools.static_scan(lambda prev, inputs: self.obs_step(prev[0], *inputs), (action, embed), (state, state))
    # prior.keys = post.keys = [mean, std, stoch, deter]
    post = {k: tf.transpose(v, [1, 0, 2]) for k, v in post.items()}
    prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
    return post, prior

  @tf.function
  def imagine(self, action, state=None):
    '''
      action: [6,45,7]
    '''
    # print("inside rssm imagine")
    if state is None:
      # initialize state
      state = self.initial(tf.shape(action)[0])
    
    assert isinstance(state, dict), state
    # print("inside imagine:", action.shape, state)
    action = tf.transpose(action, [1, 0, 2])    # [45,6,7]
    prior = tools.static_scan(self.img_step, action, state)  # dict
    prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}  # transpose each value element of prior dictionary without changing keys
    return prior

  def get_feat(self, state):
    '''
      concatenates the stochastic and deterministic part of state
    '''
    return tf.concat([state['stoch'], state['deter']], -1)

  def get_distribution(self, state):
    '''
      state={mean, std, stoch, deter} [128,30] or [128,50,30]
    '''
    # print("inside get_distribution:", state['mean'].shape, state['std'].shape, res)
    return tfd.MultivariateNormalDiag(state['mean'], state['std'])  # tfd object

  @tf.function
  def obs_step(self, prev_state, prev_action, embed):
    # print("inside rssm obs step")
    prior = self.img_step(prev_state, prev_action)
    x = tf.concat([prior['deter'], embed], -1)      # [128,1224=1024+200]

    # usual dense connected NN layer
    x = self.get('obs1', tfkl.Dense, units=self._hidden_size, activation=self._activation)(x)  # [128,200]
    # usual dense connected NN layer
    x = self.get('obs2', tfkl.Dense, units=2 * self._stoch_size, activation=None)(x)  # [128,60]
    mean, std = tf.split(x, num_or_size_splits=2, axis=-1)  # [128,30], [128,30]
    std = tf.nn.softplus(std) + 0.1   # softplus(x) = log(e^x + 1)
    stoch = self.get_distribution({'mean': mean, 'std': std}).sample()  # [128,30]
    post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}
    return post, prior

  @tf.function
  def img_step(self, prev_state, prev_action):
    # print("inside rssm img step")
    x = tf.concat([prev_state['stoch'], prev_action], -1)
    x = self.get('img1', tfkl.Dense, units=self._hidden_size, activation=self._activation)(x)
    x, deter = self._cell(x, [prev_state['deter']])
    deter = deter[0]  # Keras wraps the state in a list.
    x = self.get('img2', tfkl.Dense, units=self._hidden_size, activation=self._activation)(x)
    x = self.get('img3', tfkl.Dense, units=2 * self._stoch_size, activation=None)(x)
    mean, std = tf.split(x, num_or_size_splits=2, axis=-1)
    std = tf.nn.softplus(std) + 0.1   # softplus(x) = log(e^x + 1)
    stoch = self.get_distribution({'mean': mean, 'std': std}).sample()
    prior = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter}
    return prior


class ConvEncoder(tools.Module):
  def __init__(self, depth=32, act=tf.nn.relu, camview_rgb="agentview_image", camview_depth="agentview_depth", use_depth_obs=False):
    self._act = act      # ReLU
    self._depth = depth   # 32
    self._camview_rgb = camview_rgb
    self._camview_depth = camview_depth
    self._use_depth_obs = use_depth_obs
    # print("inside convencoder:", self._act, self._depth, self._camview_rgb)

  def __call__(self, obs):
    # print("stage-0:", obs[self._camview_rgb].shape)     # [128,50,84,84,3]
    # print("stage-1:", tuple(obs[self._camview_rgb].shape[-3:]))   # [84,84,3]
    # print("stage-2:", obs[self._camview_depth].shape)      # [128,50,84,84,1]
  
    rgb = tf.reshape(obs[self._camview_rgb], (-1,) + tuple(obs[self._camview_rgb].shape[-3:]))  # [6400,84,84,3]
    if self._use_depth_obs:
      depth = tf.reshape(obs[self._camview_depth], (-1,) + tuple(obs[self._camview_depth].shape[-3:]))  # [6400,84,84,1]
      x = tf.concat([rgb, depth], axis=-1)      # [6400,84,84,4]
    else:
      x = rgb

    # print("stage0:", x.shape)
    x = self.get('h1', tfkl.Conv2D, filters=1 * self._depth, kernel_size=4, strides=2, activation=self._act)(x)  # [6400,41,41,32]
    # print("stage1:", x.shape)
    x = self.get('h2', tfkl.Conv2D, filters=2 * self._depth, kernel_size=4, strides=2, activation=self._act)(x)  # [6400,19,19,64]
    # print("stage2:", x.shape)
    x = self.get('h3', tfkl.Conv2D, filters=4 * self._depth, kernel_size=4, strides=2, activation=self._act)(x)  # [6400,8,8,128]
    # print("stage3:", x.shape)
    x = self.get('h4', tfkl.Conv2D, filters=8 * self._depth, kernel_size=4, strides=2, activation=self._act)(x)  # [6400,3,3,256]
    # print("stage4:", x.shape)
    x = self.get('h5', tfkl.Conv2D, filters=8 * self._depth, kernel_size=2, strides=1, activation=self._act)(x)   # [6400,2,2,256]
    # print("stage5:", x.shape)
    shape = tf.concat([tf.shape(obs[self._camview_rgb])[:-3], [32 * self._depth]], 0)  # =[128,50,1024]

    # converts each image in a batch to 1024-dim vector
    return tf.reshape(x, shape)  # [128,50,1024]


class ConvDecoder(tools.Module):
  def __init__(self, depth=32, act=tf.nn.relu, shape=(84, 84, 3), use_depth_obs=False):
    self._act = act   # ReLU
    self._depth = depth  # 32
    self._use_depth_obs = use_depth_obs
    self._shape = shape if self._use_depth_obs == False else (84,84,4)  # [84,84,3/4]

  def __call__(self, features):
    # kwargs = dict(strides=2, activation=self._act)
    # print("Stage-1:", features.shape) # [128,50,230]
    x = self.get('h1', tfkl.Dense, units=32 * self._depth, activation=None)(features)   # [128,50,1024]
    # print("Stage1:", x.shape)
    x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])   # [6400,1,1,1024]
    # print("Stage2:", x.shape)
    x = self.get('h2', tfkl.Conv2DTranspose, filters=4 * self._depth, kernel_size=7, strides=2, activation=self._act)(x)  # [6400,7,7,128]
    # print("Stage3:", x.shape)
    x = self.get('h3', tfkl.Conv2DTranspose, filters=2 * self._depth, kernel_size=7, strides=2, activation=self._act)(x)  # [6400,19,19,64]
    # print("Stage4:", x.shape)
    x = self.get('h4', tfkl.Conv2DTranspose, filters=1 * self._depth, kernel_size=4, strides=2, activation=self._act)(x)  # [6400,40,40,32]
    # print("Stage5:", x.shape)
    x = self.get('h5', tfkl.Conv2DTranspose, filters=self._shape[-1], kernel_size=6, strides=2)(x)  # [6400,84,84,4]
    # print("Stage6:", x.shape)
    mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))    # [128,50,84,84,4]
    # print("stage7:", mean.shape)
    return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))


class DenseEncoder(tools.Module):
  def __init__(self, out_units=32, num_layers=2, hidden_units=100, activation=tf.nn.relu, camview_rgb="agentview_image", camview_depth="agentview_depth"):
    self._activation = activation
    self._out_units = out_units
    self._hidden_units = hidden_units
    self._num_layers = num_layers
    self._camview_rgb = camview_rgb
    self._camview_depth = camview_depth

  def __call__(self, obs):
    dtype = prec.global_policy().compute_dtype
    proprio_obs = []
    # for k, v in obs.items():
    #   # obs[k] = tf.cast(v, dtype)
    #   if k not in [self._camview_rgb, self._camview_depth, 'action', 'reward', 'cube_pos', 'cube_quat', 'cube_to_robot0_eef_pos', 'cube_to_robot0_eef_quat', 'robot0_eef_to_cube_yaw']:
    #     # print(v.shape)
    #     proprio_obs.append(tf.cast(v, dtype))
    proprio_obs.append(tf.cast(obs['robot0_proprio-state'], dtype))
    # print("Before:", proprio_obs)
    x = tf.concat(proprio_obs, axis=-1)
    # print("after:", x.shape)

    for i in range(self._num_layers):
      x = self.get(f'h{i}', tfkl.Dense, units=self._hidden_units, activation=self._activation)(x)
    x = self.get(f'hout', tfkl.Dense, units=self._out_units)(x)
    # print("final:", x.shape)

    shape = tf.concat([tf.shape(obs[self._camview_rgb])[:-3], [self._out_units]], 0)  # [128,50,32]
    return tf.reshape(x, shape)


class DenseDecoder(tools.Module):
  def __init__(self, shape, layers, units, distribution='normal', act=tf.nn.elu):
    self._shape = shape  # ()
    self._layers = layers  # 2
    self._units = units    # 400
    self._distribution = distribution  # 'normal' or 'binary'
    self._act = act    # ELU (exponential linear unit)

  def __call__(self, features):
    x = features   # [15,6272,237]
    for index in range(self._layers):
      x = self.get(f'h{index}', tfkl.Dense, units=self._units, activation=self._act)(x)   # [15,6272,400]
      # print("loop:", x.shape)
    x = self.get(f'hout', tfkl.Dense, units=np.prod(self._shape))(x)  # [15,6272,1]
    # print("stage1:", x.shape)
    x = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))  # [15,6272]
    # print("stage2:", x.shape)
    
    if self._distribution == 'normal':
      temp1 = tfd.Normal(loc=x, scale=1)
      temp2 = len(self._shape)
      temp3 = tfd.Independent(temp1, temp2)
      # print("stage3:", temp1, temp2, temp3)
      return temp3
    if self._distribution == 'binary':
      return tfd.Independent(tfd.Bernoulli(logits = x), len(self._shape)), x
    raise NotImplementedError(self._distribution)


class ActionDecoder(tools.Module):
  def __init__(self, size, layers, units, dist='tanh_normal', act=tf.nn.elu, min_std=1e-4, init_std=5, mean_scale=5):
    self._size = size   # 7
    self._layers = layers  # 4
    self._units = units  # 400
    self._dist = dist   # 'tanh_normal'
    self._act = act    # ELU
    self._min_std = min_std    # 1e-4
    self._init_std = init_std   # 5
    self._mean_scale = mean_scale  # 5

  def __call__(self, features):
    raw_init_std = np.log(np.exp(self._init_std) - 1)
    x = features   # [6272 or 128,50, 200(deter)+30(stoch)=230]
    # print("action decoder:", features.shape)
    
    for index in range(self._layers):
      x = self.get(f'h{index}', tfkl.Dense, units=self._units, activation=self._act)(x)
    
    if self._dist == 'tanh_normal':
      # https://www.desmos.com/calculator/rcmcf5jwe7
      x = self.get(f'hout', tfkl.Dense, units=2 * self._size)(x)
      mean, std = tf.split(x, num_or_size_splits=2, axis=-1)
      mean = self._mean_scale * tf.tanh(mean / self._mean_scale)
      std = tf.nn.softplus(std + raw_init_std) + self._min_std
      dist = tfd.Normal(loc=mean, scale=std)
      dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
      dist = tfd.Independent(dist, 1)
      dist = tools.SampleDist(dist)
    elif self._dist == 'onehot':
      x = self.get(f'hout', tfkl.Dense, self._size)(x)
      dist = tools.OneHotDist(x)
    else:
      raise NotImplementedError(dist)
    
    return dist
