# # Unity ML Agents
# ## Proximal Policy Optimization (PPO)
# Contains an implementation of PPO as described [here](https://arxiv.org/abs/1707.06347).

from docopt import docopt

import os
import re
from ppo.models import *
from ppo.trainer import Trainer
from unityagents import UnityEnvironment

_USAGE = '''
Usage:
  ppo (<env>) [options] 

Options:
  --help                     Show this message.
  --batch-size=<n>           How many experiences per gradient descent update step [default: 64].
  --beta=<n>                 Strength of entropy regularization [default: 2.5e-3].
  --buffer-size=<n>          How large the experience buffer should be before gradient descent [default: 2048].
  --curriculum=<file>        Curriculum json file for environment [default: None].
  --epsilon=<n>              Acceptable threshold around ratio of old and new policy probabilities [default: 0.2].
  --gamma=<n>                Reward discount rate [default: 0.99].
  --hidden-units=<n>         Number of units in hidden layer [default: 64].
  --keep-checkpoints=<n>     How many model checkpoints to keep [default: 5].
  --lambd=<n>                Lambda parameter for GAE [default: 0.95].
  --learning-rate=<rate>     Model learning rate [default: 3e-4].
  --load                     Whether to load the model or randomly initialize [default: False].
  --max-steps=<n>            Maximum number of steps to run environment [default: 1e6].
  --normalize                Whether to normalize the state input using running statistics [default: False].
  --num-epoch=<n>            Number of gradient descent steps per batch of experiences [default: 5].
  --num-layers=<n>           Number of hidden layers between state/observation and outputs [default: 2].
  --run-path=<path>          The sub-directory name for model and summary statistics [default: ppo].
  --save-freq=<n>            Frequency at which to save model [default: 50000].
  --summary-freq=<n>         Frequency at which to save training statistics [default: 10000].
  --time-horizon=<n>         How many steps to collect per agent before adding to buffer [default: 2048].
  --train                    Whether to train model, or only run inference [default: False].
  --worker-id=<n>            Number to add to communication port (5005). Used for multi-environment [default: 0].
'''

options = docopt(_USAGE)
print(options)

# General parameters
max_steps = float(options['--max-steps'])
model_path = './models/{}'.format(str(options['--run-path']))
run_path = str(options['--run-path'])
# summary_path = './summaries/{}'.format(str(options['--run-path']))
load_model = options['--load']
train_model = options['--train']
summary_freq = int(options['--summary-freq'])
save_freq = int(options['--save-freq'])
env_name = options['<env>']
keep_checkpoints = int(options['--keep-checkpoints'])
worker_id = int(options['--worker-id'])
curriculum_file = str(options['--curriculum'])
if curriculum_file == "None":
    curriculum_file = None

# Algorithm-specific parameters for tuning
gamma = float(options['--gamma'])
lambd = float(options['--lambd'])
time_horizon = int(options['--time-horizon'])
beta = float(options['--beta'])
num_epoch = int(options['--num-epoch'])
num_layers = int(options['--num-layers'])
epsilon = float(options['--epsilon'])
buffer_size = int(options['--buffer-size'])
learning_rate = float(options['--learning-rate'])
hidden_units = int(options['--hidden-units'])
batch_size = int(options['--batch-size'])
normalize = options['--normalize']

trainer_parameters = {'max_steps':max_steps, 'run_path':run_path, 'env_name':env_name,
    'curriculum_file':curriculum_file, 'gamma':gamma, 'lambd':lambd, 'time_horizon':time_horizon,
    'beta':beta, 'num_epoch':num_epoch, 'epsilon':epsilon, 'buffer_size':buffer_size,
    'learning_rate':learning_rate, 'hidden_units':hidden_units, 'batch_size':batch_size,
    'normalize':normalize, 'summary_freq':summary_freq, 'num_layers':num_layers}
# trainer parameters could contain a string for the type of trainer to use and a string for the 
# type of model to use ?

env = UnityEnvironment(file_name=env_name, worker_id=worker_id, curriculum=curriculum_file)
print(str(env))
# brain_name = env.external_brain_names[0]

tf.reset_default_graph()

# Create the Tensorflow model graph
# ppo_model = create_agent_model(env, lr=learning_rate,
#                                h_size=hidden_units, epsilon=epsilon,
#                                beta=beta, max_step=max_steps,
#                                normalize=normalize, num_layers=num_layers)

# is_continuous = (env.brains[brain_name].action_space_type == "continuous")
# use_observations = (env.brains[brain_name].number_observations > 0)
# use_states = (env.brains[brain_name].state_space_size > 0)

if not os.path.exists(model_path):
    os.makedirs(model_path)

# if not os.path.exists(summary_path):
#     os.makedirs(summary_path)
# summary_paths = {}
# for brain in env.external_brain_names:
#     summary_paths[brain] = './summaries/{}'.format(run_path+'_'+brain)
#     if not os.path.exists(summary_paths[brain]):
#         os.makedirs(summary_paths[brain])



# Need to specify steps and last_reward here : Could be a sum of rewards or the min step ?
def get_progress():
    if curriculum_file is not None:
        if env._curriculum.measure_type == "progress":
            return steps / max_steps
        elif env._curriculum.measure_type == "reward":
            return last_reward
        else:
            return None
    else:
        return None

with tf.Session() as sess:
    trainers = {}
    for brain in env.external_brain_names:
        trainers[brain] = Trainer(sess, env, brain, trainer_parameters, train_model)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=keep_checkpoints)
    # Instantiate model parameters
    if load_model:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt == None:
          print('The model {0} could not be found. Make sure you specified the right '
            '--run-path'.format(model_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(init)

    # steps, last_reward = sess.run([ppo_model.global_step, ppo_model.last_reward])
    # put summay writer into trainer <-- AND SUMMARY PATH
    # summary_writers = {}
    # for brain in env.external_brain_names:
    #     summary_writers[brain] = tf.summary.FileWriter(summary_paths[brain])
    # summary_writer = tf.summary.FileWriter(summary_path)
    info = env.reset(train_mode=train_model, progress=get_progress())
    # trainer = Trainer(ppo_model, sess, info, is_continuous, use_observations, use_states, train_model)
    if train_model:
        for brain_name, trainer in trainers.iteritems():
            trainer.write_text('Hyperparameters', trainer_parameters) #will need one trainer_parameters per trainer
    # Need at least one trainer with steps left 
    # while steps <= max_steps or not train_model:
    global_step = 0 # This is only for saving the model
    while any([t.get_step() < t.max_step for k, t in trainers.iteritems()]) :
        if env.global_done:
            info = env.reset(train_mode=train_model, progress=get_progress())
            # trainer.reset_buffers(info, total=True)
            for brain_name, trainer in trainers.iteritems():
                trainer.reset_buffers()
        # Decide and take an action
        take_action_actions = {}
        take_action_memories = {}
        take_action_values = {}
        take_action_outputs = {}
        for brain_name, trainer in trainers.iteritems():
            (take_action_actions[brain_name],
            take_action_memories[brain_name],
            take_action_values[brain_name], 
            take_action_outputs[brain_name]) = trainer.take_action(
                info[brain], env, brain_name)
        # new_info = trainer.take_action(info, env, brain_name, steps, normalize)
        new_info = env.step(action = take_action_actions, memory = take_action_memories, value = take_action_values)
        for brain_name, trainer in trainers.iteritems():
            trainer.add_experiences(info[brain_name], new_info[brain_name], take_action_outputs[brain_name])
        info = new_info
        for brain_name, trainer in trainers.iteritems():
            trainer.process_experiences(info[brain_name], time_horizon, gamma, lambd)
            # if len(trainer.training_buffer['actions']) > buffer_size and train_model:
            if trainer.is_ready_update() and train_model:
                # Perform gradient descent with experience buffer
                trainer.update_model()
            # if steps % summary_freq == 0 and steps != 0 and train_model:
            #     # Write training statistics to tensorboard.
            trainer.write_summary(env._curriculum.lesson_number)
            if train_model:
                global_step += 1
                trainer.increment_step()
                trainer.update_last_reward()
                # if len(trainer.stats['cumulative_reward']) > 0:
                #     mean_reward = np.mean(trainer.stats['cumulative_reward'])
                #     sess.run(ppo_model.update_reward, feed_dict={ppo_model.new_reward: mean_reward})
                #     last_reward = sess.run(ppo_model.last_reward)
        if global_step % save_freq == 0 and global_step != 0 and train_model:
            # Save Tensorflow model
            save_model(sess, model_path=model_path, steps=global_step, saver=saver)

            graph_name = (env_name.strip()
                  .replace('.app', '').replace('.exe', '').replace('.x86_64', '').replace('.x86', ''))
            graph_name = os.path.basename(os.path.normpath(graph_name))
            nodes = []
            for brain in trainers.keys():
                # The scope should be a Trainer property ?
                scope = (re.sub('[^0-9a-zA-Z]+', '-', brain)) + '/'
                nodes +=[scope + x for x in ["action","value_estimate","action_probs"]]
            export_graph(model_path, graph_name, target_nodes=','.join(nodes))

    # Final save Tensorflow model
    if global_step != 0 and train_model:
        save_model(sess, model_path=model_path, steps=global_step, saver=saver)
env.close()

graph_name = (env_name.strip()
      .replace('.app', '').replace('.exe', '').replace('.x86_64', '').replace('.x86', ''))
graph_name = os.path.basename(os.path.normpath(graph_name))
nodes = []
for brain in trainers.keys():
    scope = (re.sub('[^0-9a-zA-Z]+', '-', brain)) + '/'
    nodes +=[scope + x for x in ["action","value_estimate","action_probs"]]   
export_graph(model_path, graph_name, target_nodes=','.join(nodes)) 
# export_graph should be in a utils class 

