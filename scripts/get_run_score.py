# python get_run_score.py <ROOT:str> <EXP:str> <RUN:int> <STEP:int> <BATCH_SZ:int> <TYPE:str>

from data_diet.data import load_data
from data_diet.scores import compute_scores
from data_diet.utils import get_fn_params_state, load_args
import sys
import numpy as np
import os

ROOT = sys.argv[1]
EXP = sys.argv[2]
RUN = int(sys.argv[3])
STEP = int(sys.argv[4])
BATCH_SZ = int(sys.argv[5])
TYPE = sys.argv[6]

run_dir = ROOT + f'/exps/{EXP}/run_{RUN}'
args = load_args(run_dir)
args.load_dir = run_dir
args.ckpt = STEP

_, X, Y, _, _, args = load_data(args)
fn, state = get_fn_params_state(args)
scores = compute_scores(fn, state, X, Y, BATCH_SZ, TYPE)

# types: l2_error, grad_norm_scores and input_variance
if TYPE == 'l2_error':
    path_name = 'error_l2_norm_scores'
elif TYPE == 'grad_norm':
    path_name = 'grad_norm_scores'
elif TYPE == 'input_variance':
    path_name = 'input_variance_scores'
elif TYPE == 'noised_input_variance':
    path_name = 'noised_input_variance_scores'
else:
    raise ValueError(f'Invalid TYPE: {TYPE}')

save_dir = run_dir + f'/{path_name}'
save_path = run_dir + f'/{path_name}/ckpt_{STEP}.npy'
if not os.path.exists(save_dir): os.makedirs(save_dir)
np.save(save_path, scores)
