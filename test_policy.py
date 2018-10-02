import pydart2 as pydart
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
try:
    import pydart2 as pydart
    from pydart2.gui.trackball import Trackball
    pydart.init()
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install pydart2.)".format(e))
from pydart2.collision_result import CollisionResult
from pydart2.bodynode import BodyNode
import pydart2.pydart2_api as papi


import pickle
import copy
import sys
import gym 
from gym import error, spaces
import baselines.common.tf_util as U
import tensorflow as tf
from baselines.ppo1.mlp_policy import MlpPolicy
from gym import wrappers
from GENRollouts import *


num_pols = int(sys.argv[1])


if num_pols>2:
	raise ValueError("Currently only two nodes are trained!!!")

state0 = np.array([0,0,0,0])
state1 = np.array([0,np.pi,0,0])

normal = np.linalg.norm(state0-state1)

print("Normal",normal)
U.make_session(num_cpu=1).__enter__()
obs_dim = 4
act_dim = 1
env = gym.make('DartHopper-v1')
action_bounds = np.array([[1.0],[-1.0]])
action_space = spaces.Box(action_bounds[1],action_bounds[0])
high = np.inf*np.ones(obs_dim)
low = -high
obs_space =spaces.Box(low,high)

sess = tf.get_default_session()

node = 0

init_state = np.array([-0.00079183, -1.3055722 , 0.,          0.,          0.,          0.,
  0.,          0.,          0. ,         0. ,         0. ,       ])





init = {'initstate':init_state,'relay':False,'TuneValueFunction':False,'model':None,'RegressorModel':None,
    		 'node':node,'ComputeThreshold':False}


dump_video = True



classifiermodels = []
regressormodels = []
policies = []

for i in range(num_pols):
	print("i",i)
	pol_name = "Relay"+str(i)+"pi"
	policies.append(MlpPolicy(pol_name,env.observation_space,env.action_space,hid_size=32,num_hid_layers=2))
	param_name = "./Policies/Relay"+str(i)+".pkl"
	with open(param_name,"rb") as fp:
		params = pickle.load(fp)

	for item in policies[i].get_variables():
		obj = item.assign(params[item.name])
		sess.run(obj)

for i in range(num_pols-1):
	
	modelname = "Models/Relay"+str(i)+"pimodel.sav"
	
	regname = "Models/Relay"+str(i)+"piRegressormodel.sav"


	with open(modelname,"rb") as fp:
		classifiermodels.append(pickle.load(fp))

	with open(regname,"rb") as fp:
		regressormodels.append(pickle.load(fp))

	



if dump_video:
	env_wrapper = wrappers.Monitor(env, 'Data/videos/', force=True)

if dump_video:
	o = env_wrapper.reset(init)
else:
	o = env.reset(init)

rew = 0
switch = 0
traj = 8000
ct = 0
count = 0
timer = 0
rewards = []
returns = []
values = []
observations = []
i = 0
tune_VF= True

switch = False
active = num_pols -1
relay = True
while ct < traj:
	print("active",active)

	if relay == False:
		action,_ = policies[0].act(False,o)

	else:
		action,_ = policies[active].act(False,o)
		vf = regressormodels[active-1].predict(o.reshape(1,env.observation_space.shape[0]))
		#print("vf",vf)

		if classifiermodels[active-1].predict(vf.reshape(-1,1)) == 1 and relay == True:
			print("here")
			active -= 1
			action,vf = policies[active].act(False,o)
			if active == 0:

				relay = False
			switch = True


	#ac = ac_1
	if dump_video:
		o,r,d,_ = env_wrapper.step(action)
		env_wrapper.render()
	else:
		o,r,d,_ = env.step(action)
		env.render()
	
	i+=1
	if d:

		#d = False
		#print("observations length",len(observations))
		switch = False
		if dump_video:
			o = env_wrapper.reset(init)
		else:

			o = env.reset(init)
		
		observations = []
		

		timer = 0.0
		ct+=1
		print("count",ct)
		#input()
		print("****************************************************************************************")

