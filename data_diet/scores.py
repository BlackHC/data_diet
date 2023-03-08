from data_diet.train_state import TrainState
from .data import get_class_balanced_random_subset
from .gradients import compute_mean_logit_gradients, flatten_jacobian, get_mean_logit_gradients_fn
from .metrics import cross_entropy_loss
import flax.linen as nn
from jax import jacrev, jit, vmap
import jax.numpy as jnp
import numpy as np
import scipy.stats as stats

def get_lord_error_fn(fn, state, ord):
  @jit
  def lord_error(X, Y):
    errors = nn.softmax(fn(state, X)) - Y
    scores = jnp.linalg.norm(errors, ord=ord, axis=-1)
    return scores
  np_lord_error = lambda X, Y: np.array(lord_error(X, Y))
  return np_lord_error


def get_margin_error(fn, state, score_type):
  fn_jit = jit(lambda X: fn(state, X))

  def margin_error(X, Y):
    batch_sz = X.shape[0]
    P = np.array(nn.softmax(fn_jit(X)))
    correct_logits = Y.astype(bool)
    margins = P[~correct_logits].reshape(batch_sz, -1) - P[correct_logits].reshape(batch_sz, 1)
    if score_type == 'max':
      scores = np.max(margins, -1)
    elif score_type == 'sum':
      scores = np.sum(margins, -1)
    return scores

  return margin_error


def get_grad_norm_fn(fn, train_state: TrainState):

  @jit
  def score_fn(X, Y):
    def per_sample_loss_fn(params, x, y):
      vs = {**train_state.variables, 'params': params}
      logits = train_state.apply_fn(vs, x, train=False, mutable=False)
      loss = vmap(cross_entropy_loss)(logits, y)
      return loss

    loss_grads = flatten_jacobian(jacrev(per_sample_loss_fn)(train_state.params, X, Y))
    scores = jnp.linalg.norm(loss_grads, axis=-1)
    return scores

  return score_fn


def get_input_variance(fn, state):

  @jit
  def score_fn(X, Y):
    # flatten the last three dimensions (X has shape batch_sz x n_rows x n_cols x n_colors).
    X = X.reshape(X.shape[0], -1)
    # compute the norm of the input samples across the batch
    scores = jnp.linalg.norm(X, axis=-1, ord=2)
    return scores

  return lambda X, Y: np.array(score_fn(X, Y))


def get_noised_input_variance(fn, state, noise_ratio_mean=3.2924, noise_ratio_std=0.2723):
  @jit
  def score_fn(X, Y):
    # flatten the last three dimensions (X has shape batch_sz x n_colors x n_rows x n_cols).
    X = X.reshape(X.shape[0], -1)
    # compute the norm of the input samples across the batch
    scores = jnp.linalg.norm(X, axis=-1, ord=2)

    # sample normal noise and multipy the log with the scores
    noise = stats.norm.rvs(loc=noise_ratio_mean, scale=noise_ratio_std, size=X.shape[0])
    scores *= np.log(noise)
    return scores

  return lambda X, Y: np.array(score_fn(X, Y))


def get_score_fn(fn, state, score_type):
  if score_type == 'l2_error':
    print(f'compute {score_type}...')
    score_fn = get_lord_error_fn(fn, state, 2)
  elif score_type == 'l1_error':
    print(f'compute {score_type}...')
    score_fn = get_lord_error_fn(fn, state, 1)
  elif score_type == 'max_margin':
    print(f'compute {score_type}...')
    score_fn = get_margin_error(fn, state, 'max')
  elif score_type == 'sum_margin':
    print(f'compute {score_type}...')
    score_fn = get_margin_error(fn, state, 'sum')
  elif score_type == 'grad_norm':
    print(f'compute {score_type}...')
    score_fn = get_grad_norm_fn(fn, state)
  elif score_type == 'input_variance':
    print(f'compute {score_type}...')
    score_fn = get_input_variance(fn, state)
  elif score_type == 'noised_input_variance':
    print(f'compute {score_type}...')
    score_fn = get_noised_input_variance(fn, state)
  else:
    raise NotImplementedError
  return score_fn


def compute_scores(fn, state, X, Y, batch_sz, score_type):
  n_batches = X.shape[0] // batch_sz
  Xs, Ys = np.split(X, n_batches), np.split(Y, n_batches)
  score_fn = get_score_fn(fn, state, score_type)
  scores = []
  for i, (X, Y) in enumerate(zip(Xs, Ys)):
    print(f'score batch {i+1} of {n_batches}')
    scores.append(np.array(score_fn(X, Y)))
  scores = np.concatenate(scores)
  return scores


def compute_unclog_scores(fn, params, state, X, Y, cls_smpl_sz, seed, batch_sz_mlgs):
  n_batches = X.shape[0]
  Xs = np.split(X, n_batches)
  X_mlgs, _ = get_class_balanced_random_subset(X, Y, cls_smpl_sz, seed)
  mlgs = compute_mean_logit_gradients(fn, params, state, X_mlgs, batch_sz_mlgs)
  logit_grads_fn = get_mean_logit_gradients_fn(fn, params, state)
  score_fn = jit(lambda X: jnp.linalg.norm((logit_grads_fn(X) - mlgs).sum(0)))
  scores = []
  for i, X in enumerate(Xs):
    if i % 500 == 0: print(f'images {i} of {n_batches}')
    scores.append(score_fn(X).item())
  scores = np.array(scores)
  return scores
