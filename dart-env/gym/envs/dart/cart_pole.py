import numpy as np
from gym import utils
from gym.envs.dart import dart_env
import baselines.common.tf_util as U
import tensorflow as tf
from baselines.ppo1.mlp_policy import MlpPolicy
from gym import wrappers,spaces
import pickle

class DartCartPoleEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        control_bounds = np.array([[1.0],[-1.0]])
        self.action_scale = 100
        #self.reset_keywords = ['initstate']
        obs_dim = 4
        self.first_pass = True
        high = np.inf*np.ones(obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)
        self.action_space = spaces.Box(control_bounds[1], control_bounds[0])
        dart_env.DartEnv.__init__(self, 'cartpole.skel', 2, 4, control_bounds, dt=0.010, disableViewer=False)
        utils.EzPickle.__init__(self)

    def step(self, a):
        #reward = 1.0

        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[0] = a[0] * self.action_scale

        self.do_simulation(tau, self.frame_skip)
        ob = self._get_obs()
        


        ctrl_cost = 0.01*np.square(a).sum()
        swing_cost = 0.1*abs(self.robot_skeleton.dq[1])
        com = np.abs(self.robot_skeleton.q[0])
        com_cost = 0
        if abs(com) > 1.5:
            com_cost = 5. 
        ob = self._get_obs()
        #print("ob",ob[1])
        reward =  0.1*( -com_cost - ctrl_cost + 10*np.cos(ob[1]))# 

        notdone = True
        if self.relay == False:
            notdone = np.isfinite(ob).all()  and not np.isnan(ob).all() and (np.abs(self.robot_skeleton.q[1]) <= 1.57)


        #print("first term",np.isfinite(ob).all())
        #print("second term",(np.abs(self.robot_skeleton.q[1]) <= 0.7))
        #print("nan term",np.isnan(ob).all())
        #print("state",ob[1])

        done = not notdone
        
        #if done:
        #    reward = 0        

        #if self.genrollouts and not self.ComputeThreshold:
        #    done = False

        if self.relay:
            #c,vpred = self.pol.act(False,ob)
            vpred = self.regressor.predict(ob.reshape(1,self.observation_space.shape[0]))
            modelpred = self.model.predict(vpred[0])
            #print("model pred",vpred)
            if  modelpred == 1:
                #reward += 30*vpred
                if self.tune == False:
                    reward += 30*vpred
                else:
                    reward += vpred
                done = True

        if self.tune == True and self.relay == False:
            done = False
        #done = False
        #print("done",done)
        return ob, reward, done, {}


    def _get_obs(self):
        ang = self.robot_skeleton.q[1]
        #print("ang",ang)
        
        if not np.isnan(self.robot_skeleton.q).all():
            n = int(abs(ang)/(2*np.pi))
        else:
            n = 1
        state = ang
        
        if ang < 0:
            state = ang + (n+1)*2*np.pi

        if ang > 2*np.pi:
            state = ang -(n)*2*np.pi
        state = np.array([state])
        '''
        
        # reset to 0 scale when moves one rotation
        if ang < -2*np.pi:
            ang = ang + n*2*np.pi
        elif ang  > 2*np.pi:
            ang = ang - 2*np.pi*n

        if ang < -np.pi:
            ang = np.pi + (np.pi + ang)

        elif ang > np.pi:
            ang = -np.pi + (ang - np.pi) 
        #print("angle",ang)np.asarray([ang])
        '''
        return np.concatenate([np.array([self.robot_skeleton.q[0]]),state,self.robot_skeleton.dq]).ravel()#np.concatenate([self.robot_skeleton.q,self.robot_skeleton.dq]).ravel()# 

    def reset_model(self,init):
        #for k in self.reset_keywords:
        #    initstate = kwargs.get(k)
        #print("initstate",init)
        #self.dart_world.reset()
        #init = self.set_initState()
        initstate = init['initstate']
        self.relay = init['relay']
        #self.threshold = init['threshold']
        self.tune = init['TuneValueFunction']
        self.training_node = init['node']
        self.model = init['model']
        self.regressor = init['RegressorModel']
        self.ComputeThreshold = init['ComputeThreshold']
        pol_name = "Relay" + str(self.training_node) 
        pol_path = './Policies/'+pol_name+".pkl"
        #if self.first_pass:
        #    if self.relay:
        #        self.pol = MlpPolicy(pol_name+ "pi",self.observation_space,self.action_space,hid_size=64,num_hid_layers=2)
        #        sess = tf.get_default_session()
        #        with open(pol_path,"rb") as fp:
        #            self.par = pickle.load(fp)
        #        for item in self.pol.get_variables():
        #            #print(item.name)
        #            if "Adam" not in item.name:
        #                if pol_name+"pi" in item.name:
        #                    obj = item.assign(self.par[item.name])
        #                    sess.run(obj)
        #        self.first_pass = False

        qpos = np.zeros(self.robot_skeleton.ndofs)
        qpos[0] = initstate[0] + self.np_random.uniform(low=-.010, high=.010, size=1)#np.random.normal(0,0.02)#
        qpos[1] = initstate[1] + self.np_random.uniform(low=-0.4, high=0.4, size=1)#np.random.normal(0,1.5) #
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.0050, high=.0050, size=self.robot_skeleton.ndofs)#np.random.normal(0,0.0050,2)#
        #print("qpos",qpos)        
        if self.tune:
            print("here")
            qpos[0] = initstate[0] + self.np_random.uniform(low=-.035, high=.035, size=1)#np.random.normal(0,0.03)#
            qpos[1] = initstate[1] + self.np_random.uniform(low=-1.2, high=1.2, size=1)#  np.random.normal(0,2.2)#
            qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.0015, high=.0015, size=self.robot_skeleton.ndofs)#np.random.normal(0,0.015,2)#
            print("qpos",qpos)


        if self.ComputeThreshold == True:
            qpos[0] = initstate[0] + self.np_random.uniform(low=-.035, high=.035, size=1)
            qpos[1] = initstate[1] + self.np_random.uniform(low=-0.4, high=0.4, size=1)
            qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.0015, high=.0015, size=self.robot_skeleton.ndofs)#np.random.normal(0,0.015,2)#
            print("qpos compute",qpos)

        self.set_state(qpos, qvel)
        return self._get_obs()


    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -3.5
        self._get_viewer().scene.tb._set_theta(0)
        self.track_skeleton_id = 0
