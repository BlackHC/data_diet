from flax.training import checkpoints, lr_schedule
from jax import jit, value_and_grad
import numpy as np
import time
from .data import load_data, train_batches
from .forgetting import init_forget_stats, update_forget_stats, save_forget_scores
from .metrics import accuracy, correct, cross_entropy_loss
from .models import apply_fn_test, get_model
from .recorder import init_recorder, record_ckpt, record_test_stats, record_train_stats, save_recorder
from .test import get_test_step, test
from .train_state import TrainState, get_train_state
from .utils import make_dir, print_args, save_args, set_global_seed


########################################################################################################################
#  Getters
########################################################################################################################


def create_vitaly_learning_rate_schedule():
  def learning_rate(step):
    base_lr, top, total = 0.2, 4680, 31200
    if step <= top:
      lr = base_lr * step / top
    else:
      lr = base_lr - base_lr * (step - top) / (total - top)
    return lr
  return learning_rate


def get_lr_schedule(args):
  if args.lr_vitaly:
    lr = create_vitaly_learning_rate_schedule()
  elif args.decay_steps:
    lr_sched_steps = [[e, args.decay_factor**(i + 1)] for i, e in enumerate(args.decay_steps)]
    lr_ = lr_schedule.create_stepped_learning_rate_schedule(args.lr, 1, lr_sched_steps)
    lr = lambda step: lr_(step)
  else:
    lr = lr_schedule.create_constant_learning_rate_schedule(args.lr, args.steps_per_epoch)
  return lr


########################################################################################################################
#  Bookkeeping
########################################################################################################################

def _log_and_save_args(args):
  print('train args:')
  print_args(args)
  save_args(args, args.save_dir, verbose=True)


def _make_dirs(args):
  make_dir(args.save_dir)
  make_dir(args.save_dir + '/ckpts')
  if args.track_forgetting: make_dir(args.save_dir + '/forget_scores')


def _print_stats(t, T, t_incr, t_tot, lr, train_acc, test_acc, init=False):
  prog = t / T * 100
  lr = '  init' if init else f'{lr:.4f}'
  train_acc = ' init' if init else f'{train_acc:.3f}'
  print(f'{prog:6.2f}% | time: {t_incr:5.1f}s ({t_tot/60:5.1f}m) | step: {t:6d} |',
        f'lr: {lr} | train acc: {train_acc} | test acc: {test_acc:.3f}')


def _record_test(rec, t, T, t_prev, t_start, lr, train_acc, test_acc, test_loss, init=False):
  rec = record_test_stats(rec, t, test_loss, test_acc)
  t_now = time.time()
  t_incr, t_tot = t_now - t_prev, t_now - t_start
  _print_stats(t, T, t_incr, t_tot, lr, train_acc, test_acc, init)
  return rec, t_now


def _save_checkpoint(save_dir, train_state:TrainState, rec, forget_stats=None):
  checkpoints.save_checkpoint(save_dir + '/ckpts', train_state, train_state.step, keep=10000)
  if forget_stats: save_forget_scores(save_dir, train_state.step, forget_stats)
  rec = record_ckpt(rec, train_state.step)
  return rec


########################################################################################################################
#  Train
########################################################################################################################


@jit
def train_step(current_train_state: TrainState, x, y):
  def loss_fn(params):
    vs = {**current_train_state.variables, 'params': params}
    logits, model_state = current_train_state.apply_fn(vs, x, mutable=list(current_train_state.variables.keys()))
    loss = cross_entropy_loss(logits, y)
    acc = accuracy(logits, y)
    return loss, (acc, logits, model_state)

  loss_and_grad_fn = value_and_grad(loss_fn, has_aux=True)

  (loss, (acc, logits, new_model_state)), grads = loss_and_grad_fn(current_train_state.params)
  new_train_state = current_train_state.apply_gradients(
    grads=grads, variables={**current_train_state.variables, **new_model_state})

  return new_train_state, logits, loss, acc


def train(args):
  # setup
  set_global_seed()
  _make_dirs(args)
  I_train, X_train, Y_train, X_test, Y_test, args = load_data(args)
  model = get_model(args)
  lr = get_lr_schedule(args)
  train_state, args = get_train_state(args, model, lr)
  test_step = jit(get_test_step(apply_fn_test))

  rec = init_recorder()
  forget_stats = init_forget_stats(args) if args.track_forgetting else None

  # info
  _log_and_save_args(args)
  time_start = time.time()
  time_now = time_start
  print('train net...')

  # log and save init
  test_loss, test_acc = test(test_step, train_state, X_test, Y_test, args.test_batch_size)
  rec, time_now = _record_test(
      rec, args.ckpt, args.num_steps, time_now, time_start, None, None, test_acc, test_loss, True)

  assert args.ckpt == 0, "train does not support resuming."
  rec = _save_checkpoint(args.save_dir, train_state, rec, forget_stats)

  # NOTE(blackhc): train_batches does not support resuming because it does not rewind the shuffle state
  # to ckpt. This is not a problem because we don't resume.

  # train loop
  for t, idxs, x, y in train_batches(I_train, X_train, Y_train, args):
    assert t == train_state.step + 1

    # train step
    train_state, logits, loss, acc = train_step(train_state, x, y)
    if args.track_forgetting:
      batch_accs = np.array(correct(logits, y).astype(int))
      forget_stats = update_forget_stats(forget_stats, idxs, batch_accs)
    rec = record_train_stats(rec, t-1, loss.item(), acc.item(), lr(t-1))

  #  BOOKKEEPING  #

    # test and log every log_steps
    if t % args.log_steps == 0:
      test_loss, test_acc = test(test_step, train_state, X_test, Y_test, args.test_batch_size)
      rec, time_now = _record_test(rec, t-1, args.num_steps, time_now, time_start, lr(t-1), acc, test_acc, test_loss)

    # every early_save_steps before early_step and save_steps after early_step, and at end of training
    if ((t <= args.early_step and t % args.early_save_steps == 0) or
       (t > args.early_step and t % args.save_steps == 0) or
       (t == args.num_steps)):

      # test and log if not done already
      if t % args.log_steps != 0:
        test_loss, test_acc = test(test_step, train_state, X_test, Y_test, args.test_batch_size)
        rec, time_now = _record_test(rec, t-1, args.num_steps, time_now, time_start, lr(t-1), acc, test_acc, test_loss)

      # save checkpoint
      rec = _save_checkpoint(args.save_dir, train_state, rec, forget_stats)

  # wrap it up
  save_recorder(args.save_dir, rec)
