from multiprocessing import Process, Manager

from pyrep.const import RenderMode

from rlbench import ObservationConfig
from rlbench.action_modes import ActionMode
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
import rlbench.backend.task as task
import gym
import rlbench.gym
import torch

import os
import pickle
from PIL import Image
from rlbench.backend import utils
from rlbench.backend.const import *
import numpy as np

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('save_path',
                    '/tmp/rlbench_data/',
                    'Where to save the demos.')
flags.DEFINE_list('task', [],
                  'The task to collect.')
flags.DEFINE_integer('processes', 1,
                     'The number of parallel processes during collection.')
flags.DEFINE_integer('episodes_per_task', 10,
                     'The number of episodes to collect per task.')


def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def run(i, results, env):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation."""
    np.random.seed(i)
    env = gym.make(env)
    #env.seed(i)

    # Initialise each thread with different seed
    tasks_with_problems = ''
    #np.random.seed(i)

    print('Process', i, 'started collecting for task', env.task.get_name())

    task_path = os.path.join(FLAGS.save_path, env.task.get_name())
    check_and_make(task_path)

    for ex_idx in range(FLAGS.episodes_per_task):
        attempts = 10
        abort_variation = False
        while attempts > 0:
            try:
                # TODO: for now we do the explicit looping.
                #env.seed(0)
                #seed = torch.randint(0, 1, (1,)).item()
                #np.random.seed(seed)
                demo, = env.task.get_demos(
                    amount=1,
                    live_demos=True)
            except Exception as e:
                attempts -= 1
                if attempts > 0:
                    continue
                problem = (
                    'Process %d failed collecting task %s (example: %d). Skipping this task.\n%s\n' % (
                        i, env.task.get_name(), ex_idx, str(e))
                )
                print(problem)
                tasks_with_problems += problem
                abort_variation = True
                break
            episode_path = os.path.join(task_path, str(i + FLAGS.processes*ex_idx))
            check_and_make(episode_path)
            for j, obs in enumerate(demo):
                with open(os.path.join(episode_path, str(j) + '.pkl'), 'wb') as f:
                    obs.left_shoulder_rgb = None
                    obs.left_shoulder_depth = None
                    obs.left_shoulder_mask = None
                    obs.right_shoulder_rgb = None
                    obs.right_shoulder_depth = None
                    obs.right_shoulder_mask = None
                    obs.wrist_rgb = None
                    obs.wrist_depth = None
                    obs.wrist_mask = None
                    pickle.dump(obs, f)
                    #pickle.dump(env._extract_obs(obs), f)
            break
        if abort_variation:
            break

    results[i] = tasks_with_problems


def main(argv):
    manager = Manager()
    result_dict = manager.dict()

    check_and_make(FLAGS.save_path)

    processes = [Process(
        target=run, args=(
            i, result_dict, FLAGS.task[0]))
        for i in range(FLAGS.processes)]
    [t.start() for t in processes]
    [t.join() for t in processes]

    print('Data collection done!')
    for i in range(FLAGS.processes):
        print(result_dict[i])


if __name__ == '__main__':
  app.run(main)
