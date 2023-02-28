import numpy as np
import optax
from flax.training import train_state
from jax._src.tree_util import tree_flatten
from optax import sgd
from flax.training import checkpoints
from jax import jit, random
import jax.numpy as jnp
import time
from typing import Any


class TrainState(train_state.TrainState):
  variables: Any


def get_num_params(params):
  return int(sum([np.prod(w.shape) for w in tree_flatten(params)[0]]))


def create_train_state(args, model, lr) -> TrainState:
  @jit
  def init(*args):
    return model.init(*args)
  key, input = random.PRNGKey(args.model_seed), jnp.ones((1, *args.image_shape), model.dtype)

  if not hasattr(args, 'nesterov'): args.nesterov = False
  opt = optax.chain(
    optax.add_decayed_weights(args.weight_decay),
    sgd(lr, args.beta, args.nesterov),
  )

  variables, params = init(key, input).pop("params")

  train_state = TrainState.create(apply_fn=model.apply, params=params, tx=opt, variables=variables)
  return train_state


def get_train_state(args, model, lr):
  time_start = time.time()
  print('get train state... ', end='')
  train_state = create_train_state(args, model, lr)
  if args.load_dir:
    print(f'load from {args.load_dir}/ckpts/checkpoint_{args.ckpt}... ', end='')
    train_state = checkpoints.restore_checkpoint(args.load_dir + '/ckpts', train_state, args.ckpt)
  args.num_params = get_num_params(train_state.params)
  print(f'{int(time.time() - time_start)}s')
  return train_state, args
