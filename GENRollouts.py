import pydart2 as pydart
import numpy as np
from scipy.optimize import minimize
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import pickle
import copy
import sys
import gym 
from gym import error, spaces
import baselines.common.tf_util as U
import tensorflow as tf
from baselines.ppo1.mlp_policy import MlpPolicy
from gym import wrappers
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.ensemble import AdaBoostRegressor,AdaBoostClassifier

class RelayFunctions(object):

	def __init__(self,env,pol_name,node,info,_gamma,_time_step,hid_size,num_layers):
		self.env = env
		self.var_name = pol_name + 'pi'
		self.policy = MlpPolicy(self.var_name,env.observation_space,env.action_space,hid_size=hid_size,num_hid_layers=num_layers)
		U.initialize()
		self.Pol_path = './Policies/'+pol_name
		#U.load_state(self.Pol_path)
		with open(self.Pol_path+".pkl","rb") as fp:
			params = pickle.load(fp)

		sess = tf.get_default_session()
		for item in self.policy.get_variables():
			obj = item.assign(params[item.name])
			sess.run(obj)
		self.current_node = node
		self.num_rollouts = 10
		self.info = info
		self.discount_rate = _gamma
		self.t = _time_step
		self.hid_size = hid_size
		self.num_layers = num_layers
		self.node = node

		self.model = AdaBoostClassifier(n_estimators=50,random_state=3)#svm.SVC()#MLPClassifier(solver='adam',hidden_layer_sizes=(64,64), alpha=1e-3, random_state=1)#activation='tanh',max_iter = 1000
		self.regressor =  AdaBoostRegressor(loss='square',random_state=1,n_estimators=50)# 	MLPRegressor(solver='adam',activation='tanh',hidden_layer_sizes=(64,32),max_iter = 1000,random_state=1)#

	def rollout_generator(self,num_Rollouts=1):
		# returns training data set of value function ,initial state and returns 
		node = self.node
		rews = []
		vfs = []
		states = []

		returns_initState = []
		vf_initState = []
		initState = []
		termination = []
		Node_state = self.info['initstate']
		initstates = np.zeros((self.num_rollouts,self.env.observation_space.shape[0]))
		initstates[:,1] = [1]
		if self.env.observation_space.shape[0] == 11:#hopper:
			init_angle_dist = np.array([[-np.pi/2,-1.0525],[-np.pi/3,-0.45],[-np.pi/4,-0.25],[-np.pi/6,-0.125],[0,0]])
			x = init_angle_dist[:,0]
			y = init_angle_dist[:,1]
			z = np.polyfit(x, y, 3)
			f = np.poly1d(z)
			initstates[:,1] = Node_state[1] + np.linspace(-0.8,0.8,self.num_rollouts)
			initstates[:,2:] = Node_state[2:] + np.random.uniform(low=-0.0015,high=0.0015,size=(self.num_rollouts,9))
			for i in range(self.num_rollouts):
				initstates[i,0] = f(initstates[i,1])
		
		#print("node",node)
		'''
		if self.info['relay']:
			policies = []
			classifiers = []
			for i in range(node):
				name = 'Relay'+str(i)+"pi"
				policies.append(MlpPolicy(name,self.env.observation_space,self.env.action_space,hid_size=self.hid_size,num_hid_layers=self.num_layers))
				param_name = 'Relay'+str(i)+".pkl"
				with open("./Policies/"+param_name,"rb") as fp:
					params = pickle.load(fp)
				sess = tf.get_default_session()
				for item in policies[i].get_variables():
					if "Adam" not in item.name:
						if "Relay"+str(i)+"pi" in name:
							obj = item.assign(params[item.name])
							sess.run(obj)

				with open("Relay"+str(i)+"pimodel.sav","rb") as fp:
					model = pickle.load(fp)
				#print("model",model.predict(0))
				classifiers.append(model)
			#print("classifiers",len(classifiers))
			#print("policies",policies)
		'''
		ct = 0
		self.info['initstate'] = initstates[ct,:]
		o = self.env.reset(self.info)
		
		
		time_step=0
		notDone = True
		switch = False
		active = node
		#print("active",active)
		relay = True
		while notDone == True:
			#print(ct)
			#print("active",active)
			action,vf = self.policy.act(False,o)
			'''
			if self.info['relay'] == False:
				action,vf = self.policy.act(False,o)

			else:
				action,vf = self.policy.act(False,o)
				if switch : 
					#print("here")
					action,vf = policies[active].act(False,o)

				if classifiers[active-1].predict(vf) == 1 and relay == True:
					#print("here")
					active = node -1
					action,vf = policies[active].act(False,o)
					node-=1
					if active == 0:

						relay = False
					switch = True
			'''
			states.append(o)
			vfs.append(vf)

			o,r,d,_ = self.env.step(action)

			#self.env.render()
			


			rews.append(r)
			time_step+=1
			
			if d:
				d = False
				#print("##################################  START")
				ct += 1
				#print("count",ct)
				#print("length of rollout",len(rews))
				self.info['initstate'] = initstates[ct,:]
				o = self.env.reset(self.info)
				actual_returns = []
				gamma = self.discount_rate
				#print("rews",rews[:10])
				num_states = 5
				index = np.arange(5)#np.random.randint(low=0,high=len(rews),size=num_states)
				#print("indexes",index)
				if len(rews) <=5 :
					ind = len(rews)
					index =np.arange(ind)# np.random.randint(low=0,high=len(rews),size=num_states)
				else:
					ind = 5
				if self.info['ComputeThreshold']:
					ind = 1
					index = np.array([0])

				for i in range(ind):

					re = rews[index[i]]
					t = self.t
					for j in range(index[i]+1,len(rews)):
						re+= (gamma**(t))*rews[j]
						t+=self.t

					actual_returns.append(re)


				#print("states",states)
				#actual_returns = np.asarray(actual_returns)
				#print("actual_returns",actual_returns.shape)
				#vfs = np.asarray(vfs)
				#states = np.asarray(states)
				if self.info['ComputeThreshold'] == True:
					#if ct == self.num_rollouts:#len(rews) == 1000
					returns_initState.append(actual_returns[0])
					vf_initState.append(vfs[0])
					initState.append(states[0])
					#print("len",len(rews))

					#print("state",states[0])
					#print("vf",vf_initState)
					print("rets",actual_returns[0])
					#input()
					#print(len(returns_initState))
					#print("actual Returns",actual_returns[0])
					#print("valueFunc",vfs[0])
	

					if self.info['relay'] == False:
						if len(rews) > 200:
							termination.append(1)
								#print("unterminated")
						else:
							termination.append(0)

					if self.info['relay'] ==True:
						if len(rews) > 200:
							termination.append(0)
						else:
							termination.append(1)

					if ct == initstates.shape[0]-1:
						notDone = False
				else:


					returns_initState.extend(actual_returns[:ind])
					for i in range(index.shape[0]):
						vf_initState.append(vfs[index[i]])
						initState.append(states[index[i]])
						if self.info['relay'] == False:
							if len(rews) > 800:
								termination.append(1)
								#print("unterminated")
							else:
								termination.append(0)

						if self.info['relay'] ==True:
							if len(rews) > 800:
								termination.append(0)
							else:
								termination.append(1)


					'''
					with open("returns.txt","ab") as fp:
						np.savetxt(fp,np.asarray(returns_initState),fmt="%1.5f")
					with open("vfs.txt","ab") as fp:
						np.savetxt(fp,np.asarray(vf_initState),fmt="%1.5f")
					with open("initstate.txt","ab") as fp:
						np.savetxt(fp,np.asarray(initState),fmt="%1.5f")
					'''
					#print("vfs",vfs[index[0]])
					#vf_initState.extend(vfs[index])


					#initState.extend(states[index])
					print(len(returns_initState))

					print("actual Returns",actual_returns[0])
					#print("valueFunc",vfs[0])
			

					if ct == initstates.shape[0]-1:
						notDone = False
				#if self.info['relay'] == False:
				#	if len(rews) > 800:
				#		termination.append(1)
				#	else:
				#		termination.append(0)

				#if self.info['relay'] ==True:
				#	if len(rews) > 800:
				#		termination.append(0)
				#	else:
				#		termination.append(1)

				time_step = 0
				rews = []
				vfs = []
				states = []
				#print("##################################  END")
		returns_initState = np.asarray(returns_initState).flatten()

		self.info['initstate'] = Node_state
		
		#print("states saved")
		#print("actual_returns",returns_initState.shape)
		#print("vf ini",len(vf_initState))
		#print("states",np.asarray(initState).shape)


		return np.asarray(returns_initState),np.asarray(vf_initState),np.asarray(initState),termination


	def ComputeThreshold(self,_num_rollouts):

		#update policy parameters
		'''
		sess = tf.get_default_session()
		with open(self.Pol_path+".pkl","rb") as fp:
			params = pickle.load(fp)

		for item in self.policy.get_variables():
			if "Adam" not in item.name:
				if pol_name+"pi" in item.name:
					obj = item.assign(params[item.name])
					sess.run(obj)
		'''
		self.num_rollouts = _num_rollouts
		self.info['TuneValueFunction'] = False
		self.info['ComputeThreshold'] = True
		TrueReturns,valueFunc,state,d = self.rollout_generator(self.num_rollouts)
		self.info['ComputeThreshold'] = False

		print("TrueReturns",TrueReturns.shape[0])
		print("value",valueFunc.shape[0])
		print("states",state.shape[0])
		#input()
		#print("TrueReturns",TrueReturns)
		if self.info['relay'] == False:
			unterminated_returns = []
			for i in range(len(TrueReturns)):
				print("di",d[i])
				if d[i] == 1:
					unterminated_returns.append(TrueReturns[i])
		elif self.info['relay'] == True:
			unterminated_returns = []
			for i in range(len(TrueReturns)):
				if d[i] == 1:
					unterminated_returns.append(TrueReturns[i])

		print("unterminated_returns",unterminated_returns)
		#input()
		averageReturn = np.mean(unterminated_returns)
		#print("unterminated returns",unterminated_returns)
		print("Average Return",averageReturn)

		for i in range(state.shape[0]):
			valueFunc[i] = self.regressor.predict(state[i,:].reshape(1,self.env.observation_space.shape[0]))

		label = []
		for i in range(state.shape[0]):
			valueFunc[i] = self.regressor.predict(state[i,:].reshape(1,self.env.observation_space.shape[0]))
		vfs = []
		re = []
		for i in range(len(TrueReturns)):
			if TrueReturns[i] in unterminated_returns:
				label.append(1.0)
				vfs.append(valueFunc[i])
				re.append(TrueReturns[i])
			else:
				label.append(0.0)
		

		plt.plot(valueFunc,'b')
		plt.plot(TrueReturns,'r')
		plt.show()

		#input()
		self.model.fit(np.array([valueFunc]).reshape(len(TrueReturns),1),label)#np.array([valueFunc]).reshape(len(TrueReturns),1)
		'''
		splits = []
		best_splits = self.model.support_vectors_
		for i in range(best_splits.shape[0]):
			if best_splits[i] >= np.min(vfs):
				splits.append(best_splits[i])

		self.threshold = np.min(splits)

		print("THRESHOLD",self.threshold)
		'''
		#plt.plot(valueFunc,'r')
		#plt.plot(re,'b')
		#plt.show()

		# naive classifier
		self.threshold =  np.min(vfs)
		#print("THRESHOLD",self.threshold)
		pred = []
		for i in range(valueFunc.shape[0]):
			pred.append(self.model.predict(valueFunc[i]))
		
		plt.plot(label,'b')
		plt.plot(pred,'r')
		plt.show()



		# save classifier model
		model_name = self.var_name + "model.sav"

		with open(model_name,"wb") as fp:
			pickle.dump(self.model,fp)

		return self.model


	def TuneValueFunction(self,_num_rollouts=None):
		self.num_rollouts = _num_rollouts
		self.info['TuneValueFunction'] = True
		TrueReturns,valueFunc,state,d = self.rollout_generator(self.num_rollouts)
		#print("TrueReturns",valueFunc.shape)
		
		self.regressor.fit(state,TrueReturns)

		pred  = []
		for i in range(TrueReturns.shape[0]):
			pred.append(self.regressor.predict(state[i,:].reshape(1,self.env.observation_space.shape[0])))

		plt.plot(pred,'b')
		plt.plot(TrueReturns,'r')
		plt.show()


		model_name = self.var_name + "Regressormodel.sav"

		with open(model_name,"wb") as fp:
			pickle.dump(self.regressor,fp)


		#############3 CAN USE THIS METHOD AS WELL
		'''
		v = tf.placeholder(tf.float32,[None,])
		s = U.get_placeholder_cached(name="ob")

		vf_loss = tf.losses.mean_squared_error(self.policy.vpred, v)#tf.reduce_sum(tf.log(tf.cosh(self.policy.vpred - v)))#tf.sqrt(tf.reduce_sum(tf.square(self.policy.vpred - v)))#tf.reduce_mean(tf.square(self.policy.vpred - v))
		# stores action and log net params
		sess = tf.get_default_session()
		stored_params = dict()
		for item in self.policy.get_variables():
			print(item.name)
			if 'pol' in item.name or 'log' in item.name:
				obj = sess.run(item)
				stored_params[item.name] = obj
		learning_rate = 0.20
		training_epochs = 1000
		cost_history = np.empty(shape=[1],dtype=float)
		training_step = tf.train.AdamOptimizer(learning_rate).minimize(vf_loss)
		U.initialize()
		
		Y_train = TrueReturns#.reshape(TrueReturns.shape[0],)
		
		
		print("tuning....")
		for epoch in range(training_epochs):

			ind = np.random.randint(0,len(Y_train),32)

			sess.run(training_step,feed_dict={s:state[ind,:],v:Y_train[ind]})
			cost_history = np.append(cost_history,sess.run(vf_loss,feed_dict={s:state[ind,:],v:Y_train[ind]}))

		print(cost_history)
		plt.plot(cost_history)
		plt.show()

		#restores the action and log net params

		for item in self.policy.get_variables():
			if "Adam" not in item.name:
				if "pol" in item.name or "log" in item.name:
					obj = item.assign(stored_params[item.name])
					sess.run(obj)


		stored_params = dict()
		for item in self.policy.get_variables():
			#print(item.name)
			if "Adam" not in item.name:
				if self.var_name in item.name:
					#print(item.name)
					obj = sess.run(item)
					stored_params[item.name] = obj

		with open(self.Pol_path+".pkl","wb") as fp:
			pickle.dump(stored_params,fp)
		print("policy Stored..............................................")
		
		'''


		return self.regressor

	def getGrad(self,state):
		dx = []
		for i in range(state.shape[0]):
			x = state.reshape(1,self.env.observation_space.shape[0])
			x[0,i]+=0.05
			dx1 = self.regressor.predict(x)
			x[0,i]-=0.10
			dx2 = self.regressor.predict(x)
			diff = dx1[0]-dx2[0]
			if i >= 2:
				diff = 0.
			dx.append(diff)

		print(dx)
		return dx


	def GenerateNewState(self,goal_state=None,init_state=None,weight=3.0e2):
		
		sess = tf.get_default_session()
		ob = U.get_placeholder_cached(name="ob")

		dV_ds = tf.gradients(self.policy.vpred,ob)
		vf_output = U.function([ob],self.policy.vpred)
		
		dim = self.env.observation_space.shape[0]
		#print("dim",dim)
		# hopper
		if dim == 11:
			init_angle_dist = np.array([[-np.pi/2,-1.0525],[-np.pi/3,-0.45],[-np.pi/4,-0.25],[-np.pi/6,-0.125],[0,0]])
			x = init_angle_dist[:,0]
			y = init_angle_dist[:,1]

			z = np.polyfit(x, y, 3)
			f = np.poly1d(z)
		# walker 2d
		if dim == 17:
			init_angle_dist = np.array([[np.pi/2,-1.1525],[np.pi/3,-0.60],[np.pi/4,-0.35],[np.pi/6,-0.175],[0,0]])
			x = init_angle_dist[:,0]
			y = init_angle_dist[:,1]

			z = np.polyfit(x, y, 3)
			f = np.poly1d(z)

		learningRate = 0.001
		notdone = True
		state = copy.deepcopy(init_state.reshape(dim,))

		GradientSteps = 0
		W = weight# make this a tunable parameter
		while notdone:
			grads = sess.run(dV_ds,feed_dict={ob:state.reshape(1,dim)})
			#print("grads",grads)
			grad = self.getGrad(state)
			heuristicGrad = W*2*(goal_state - state.reshape(dim,))
			#print("heuristicGrad",heuristicGrad)
			gradientss = copy.deepcopy(grad + heuristicGrad)
			
			state += learningRate*gradientss/np.linalg.norm(gradientss)
			if dim == 11:
				state[0] = f(state[1])
				state[2:] = 0
			if dim == 17:
				state[0] = f(state[1])
			
			print("state",state)
			
			stateValue = self.regressor.predict(state.reshape(1,dim))
			print("value",stateValue)

			GradientSteps+=1
			StateNorm = np.linalg.norm(abs(state)-goal_state)
			print("norm",StateNorm)
			pred = self.model.predict(stateValue.reshape(-1,1))
			
			if pred == 0 or StateNorm < 0.74:
				notdone = False
				#return False
				if StateNorm < 0.74:
					complete = True
				else:
					complete = False

			




		return state,complete








