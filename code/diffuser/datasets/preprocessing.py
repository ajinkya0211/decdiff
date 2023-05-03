import gym
import numpy as np
import einops
from scipy.spatial.transform import Rotation as R
import pdb

# from .d4rl import load_environment

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def compose(*fns):
    def _fn(x):
        for fn in fns:
            x = fn(x)
        return x
    return _fn

def get_preprocess_fn(fn_names, env):
    fns = []
    for name in fn_names:
        fns.append(eval(name)(env))
    return compose(*fns)

def get_policy_preprocess_fn(fn_names):
    fns = []
    for name in fn_names:
        fns.append(eval(name))
    return compose(*fns)

#-----------------------------------------------------------------------------#
#-------------------------- preprocessing functions --------------------------#
#-----------------------------------------------------------------------------#

#------------------------ @TODO: remove some of these ------------------------#

def arctanh_actions(*args, **kwargs):
    epsilon = 1e-4

    def _fn(dataset):
        actions = dataset['actions']
        if not (-1 <= actions <= 1).all():
            raise ValueError(f"Actions not in range [-1, 1]: {actions}")
        actions = np.clip(actions, -1 + epsilon, 1 - epsilon)
        dataset['actions'] = np.arctanh(actions)
        return dataset

    return _fn

def add_deltas(env):

    def _fn(dataset):
        deltas = dataset['next_observations'] - dataset['observations']
        dataset['deltas'] = deltas
        return dataset

    return _fn


def maze2d_set_terminals(env):
    if isinstance(env, str):
        env = load_environment(env)
    goal = np.array(env._target)
    threshold = 0.5

    def _fn(dataset):
        xy = dataset['observations'][:, :2]
        distances = np.linalg.norm(xy - goal, axis=-1)
        at_goal = distances < threshold
        timeouts = np.zeros_like(dataset['timeouts'])

        ## timeout at time t iff
        ##      at goal at time t and
        ##      not at goal at time t + 1
        timeouts[:-1] = at_goal[:-1] * ~at_goal[1:]

        timeout_steps = np.where(timeouts)[0]
        path_lengths = timeout_steps[1:] - timeout_steps[:-1]

        print(
            f'[ utils/preprocessing ] Segmented {env.name} | {len(path_lengths)} paths | '
            f'min length: {path_lengths.min()} | max length: {path_lengths.max()}'
        )

        dataset['timeouts'] = timeouts
        return dataset

    return _fn



#-------------------------- block-stacking --------------------------#

import numpy as np
from scipy.spatial.transform import Rotation as R

def blocks_quat_to_euler(observations):
    '''
        input : [ N x robot_dim + n_blocks * 8 ] = [ N x 39 ]
            xyz: 3
            quat: 4
            contact: 1

        returns : [ N x robot_dim + n_blocks * 10] = [ N x 47 ]
            xyz: 3
            sin: 3
            cos: 3
            contact: 1
    '''
    robot_dim = 7
    block_dim = 8
    n_blocks = 4
    assert observations.shape[-1] == robot_dim + n_blocks * block_dim

    X = observations[:, :robot_dim]

    for i in range(n_blocks):
        start = robot_dim + i * block_dim
        end = start + block_dim

        block_info = observations[:, start:end]

        xpos = block_info[:, :3]
        quat = block_info[:, 3:-1]
        contact = block_info[:, -1:]

        euler = R.from_quat(quat).as_euler('xyz')
        sin = np.sin(euler)
        cos = np.cos(euler)

        X = np.concatenate([
            X,
            xpos,
            sin,
            cos,
            contact,
        ], axis=-1)

    return X

def blocks_euler_to_quat_2d(observations):
    robot_dim = 7
    block_dim = 10
    n_blocks = 4

    assert observations.shape[-1] == robot_dim + n_blocks * block_dim

    X = observations[:, :robot_dim]

    for i in range(n_blocks):
        start = robot_dim + i * block_dim
        end = start + block_dim

        block_info = observations[:, start:end]

        xpos = block_info[:, :3]
        sin = block_info[:, 3:6]
        cos = block_info[:, 6:9]
        contact = block_info[:, 9:]

        euler = np.arctan2(sin, cos)
        quat = R.from_euler('xyz', euler, degrees=False).as_quat()

        X = np.concatenate([
            X,
            xpos,
            quat,
            contact,
        ], axis=-1)

    return X

import numpy as np
from scipy.spatial.transform import Rotation as R
import einops

def blocks_euler_to_quat(paths):
    quats = []
    for path in paths:
        quat = blocks_euler_to_quat_2d(path)
        quats.append(quat)
    return np.stack(quats, axis=0)

def blocks_process_cubes(env):
    def process_dataset(dataset):
        for key in ['observations', 'next_observations']:
            dataset[key] = blocks_quat_to_euler(dataset[key])
        return dataset
    return process_dataset

def blocks_remove_kuka(env):
    def remove_kuka(dataset):
        for key in ['observations', 'next_observations']:
            dataset[key] = dataset[key][:, 7:]
        return dataset
    return remove_kuka

def blocks_add_kuka(observations):
    '''
        observations : [ batch_size x horizon x 32 ]
    '''
    robot_dim = 7
    batch_size, horizon, _ = observations.shape
    zeros = np.zeros((batch_size, horizon, 7))
    observations = np.concatenate([zeros, observations], axis=-1)
    return observations

def blocks_cumsum_quat(deltas):
    '''
        deltas : [ batch_size x horizon x transition_dim ]
    '''
    robot_dim = 7
    block_dim = 8
    n_blocks = 4
    assert deltas.shape[-1] == robot_dim + n_blocks * block_dim

    batch_size, horizon, _ = deltas.shape
    cumsum = deltas.cumsum(axis=1)
    for i in range(n_blocks):
        start = robot_dim + i * block_dim + 3
        end = start + 4
        quat = deltas[:, :, start:end].copy()
        quat = einops.rearrange(quat, 'b h q -> (b h) q')
        euler = R.from_quat(quat).as_euler('xyz')
        euler = einops.rearrange(euler, '(b h) e -> b h e', b=batch_size)
        cumsum_euler = euler.cumsum(axis=1)
        cumsum_euler = einops.rearrange(cumsum_euler, 'b h e -> (b h) e')
        cumsum_quat = R.from_euler('xyz', cumsum_euler).as_quat()
        cumsum_quat = einops.rearrange(cumsum_quat, '(b h) q -> b h q', b=batch_size)
        cumsum[:, :, start:end] = cumsum_quat.copy()

    return cumsum

def blocks_delta_quat_helper(observations, next_observations):
    '''
        input : [ N x robot_dim + n_blocks * 8 ] = [ N x 39 ]
            xyz: 3
            quat: 4
            contact: 1
    '''
    robot_dim = 7
    block_dim = 8
    n_blocks = 4
    
    assert observations.shape[-1] == next_observations.shape[-1] == robot_dim + n_blocks * block_dim
    
    deltas = (next_observations - observations)[:, :robot_dim]

    for i in range(n_blocks):
        start = robot_dim + i * block_dim
        end = start + block_dim
        
        block_info = observations[:, start:end]
        next_block_info = next_observations[:, start:end]
        
        xpos = block_info[:, :3]
        next_xpos = next_block_info[:, :3]
        
        quat = block_info[:, 3:7]
        next_quat = next_block_info[:, 3:7]
        
        contact = block_info[:, 7]
        next_contact = next_block_info[:, 7]
        
        delta_xpos = next_xpos - xpos
        delta_contact = next_contact - contact
        
        rot = R.from_quat(quat)
        next_rot = R.from_quat(next_quat)
        
        delta_quat = (next_rot * rot.inv()).as_quat()
        w = delta_quat[:, -1:]
        
        delta_quat = np.multiply(delta_quat, np.sign(w))
        
        next_euler = next_rot.as_euler('xyz')
        next_euler_check = (R.from_quat(delta_quat) * rot).as_euler('xyz')
        assert np.allclose(next_euler, next_euler_check)
        
        deltas = np.concatenate([deltas, delta_xpos, delta_quat, delta_contact], axis=-1)

    return deltas

def blocks_add_deltas(env):
    def _fn(dataset):
        deltas = blocks_delta_quat_helper(dataset['observations'], dataset['next_observations'])
        dataset['deltas'] = deltas
        return dataset
    return _fn
