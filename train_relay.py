#!/usr/bin/env python3

from baselines.common.cmd_util import make_dart_env, common_arg_parser,make_vec_env
from baselines.common import tf_util as U
from baselines import logger
from GENRollouts import *
import tensorflow as tf
def train(env_id, num_timesteps, seed):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    init_state = [0,0,0,0] # this can be a dict with
    threshold=0.


    env = make_dart_env(env_id, seed=seed) #make_vec_env(env_id,"dart",8,seed=seed)#
    # over all while loop for training relay policies

    #train first policy

    #generate rollouts from the policy

    #tune Value function and generate a new node init state

    # check if new node is close to goal state if then break

    #else - create a new pol with new name 

    #initial state for cartpole 
    init_cartpole = np.array([0.,0.,0.,0.])
    #initial state for hopper
    init_hopper = np.zeros(11,)
    init_hopper[1] = -0.2
    #initial state for walker 2d
    init_walker2d = np.zeros(17,)

    #define the node  here
    node = 0
    done = False
    iters = 1
    # Dictionary passed into Environment to control the relay chain generation
    init = {'initstate':init_hopper,'relay':False,'TuneValueFunction':False,'model':None,'RegressorModel':None,"ComputeThreshold":False,
    		 'node':node}
    # Flag for completion
    complete = False
    while not done:

    	policy = 'Relay'+str(node)
    	
		# Define the policy here
		hid_size = 32
		num_layers = 2

    	def policy_fn(name, ob_space, ac_space):
        	return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            	hid_size=hid_size, num_hid_layers=num_layers)
    	
    	# This is passed into PPO to break 
    	if node == 0:
    		iter_limit = 200
    	else:
    		iter_limit = 75

    	# RUN PPO
    	pi=pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',initstate=init,pol_name=policy,iters=iter_limit,relay_num = node
        	)
    	if complete:
    		break

    	pol_name = "Relay"+str(node)

    	
    	# This class member handles Post-Processing of the trained policy
    	generator = RelayFunctions(env,pol_name=pol_name,node=node,info=init,_gamma=1.0,_time_step=0.002,hid_size=32,num_layers =2)
    	# Tune the current ValueFunction
    	regressorModel = generator.TuneValueFunction(_num_rollouts=20)
    	# Fit a classifier
    	classifierModel = generator.ComputeThreshold(_num_rollouts=20)
    	
    	# Define Goal States for generating new nodes in the ENv
    	goal_state_cartpole = np.array([0,np.pi,0,0])

    	goal_state_hopper = np.zeros(11,)
    	goal_state_hopper[[0,1]] = [np.pi/2,1.0525]

    	goal_state_walker2d = np.zeros(17,)
    	goal_state_walker2d[[0,1]] = [-1.15,np.pi/2]

    	# This gives me a new node
    	newInitState,complete = generator.GenerateNewState(goal_state=goal_state_hopper,init_state=init['initstate'])

    	# Update the Dict input to DartEnv
    	init['model'] = classifierModel
    	init['RegressorModel'] = regressorModel
    	init['initstate'] = newInitState
    	init['relay'] = True
    	init['TuneValueFunction'] = False
    	
    	#print("New state",newInitState)
    	
    	node+=1
    	iters+=1
    	#done = True
    	#print("node",node)
    	#print("policy name",policy)

    env.close()



def main():
    args = common_arg_parser().parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == '__main__':
    main()
