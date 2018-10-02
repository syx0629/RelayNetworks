import numpy as np
from gym import utils
from gym.envs.dart import dart_env
import baselines.common.tf_util as U
import tensorflow as tf
from baselines.ppo1.mlp_policy import MlpPolicy
from gym import wrappers,spaces
import pickle

class DartHopperEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0, 1.0, 1.0],[-1.0, -1.0, -1.0]])
        self.action_scale = 200
        obs_dim = 11
        self.first_pass = True
        high = np.inf*np.ones(obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)
        self.action_space = spaces.Box(self.control_bounds[1], self.control_bounds[0])
        dart_env.DartEnv.__init__(self, 'hopper_capsule.skel', 4, obs_dim, self.control_bounds, disableViewer=False)

        try:
            self.dart_world.set_collision_detector(3)
        except Exception as e:
            print('Does not have ODE collision detector, reverted to bullet collision detector')
            self.dart_world.set_collision_detector(2)
        

        utils.EzPickle.__init__(self)


    def advance(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[3:] = clamped_control * self.action_scale

        self.do_simulation(tau, self.frame_skip)

    def step(self, a):
        pre_state = [self.state_vector()]

        posbefore = self.robot_skeleton.q[0]
        self.advance(a)
        posafter,ang = self.robot_skeleton.q[0,2]
        height = self.robot_skeleton.bodynodes[2].com()[1]


        joint_limit_penalty = 0
        for j in [-2]:
            if (self.robot_skeleton.q_lower[j] - self.robot_skeleton.q[j]) > -0.05:
                joint_limit_penalty += abs(1.5)
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[j]) < 0.05:
                joint_limit_penalty += abs(1.5)

        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt# +   (0.1 / (abs(self.robot_skeleton.q[2])*5.0+1))
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        reward -= 5e-1 * joint_limit_penalty
        reward*=0.1
        s = self.state_vector()
        done = False
        if self.relay == False:
            done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                        (height > .7) and (height < 1.8) and (abs(ang) < .8))
        ob = self._get_obs()


        if self.relay:
            #c,vpred = self.pol.act(False,ob)
            vpred = self.regressorModel.predict(ob.reshape(1,self.observation_space.shape[0]))
            modelpred = self.model.predict(vpred.reshape(-1,1))#ob.reshape(1,self.observation_space)
            #print("model pred",vpred)
            if  modelpred == 1:#
                #reward += 30*vpred
                if self.tune == False:
                    reward += 30*vpred[0]
                else:
                    reward += vpred[0]
                done = True

        if self.tune == True and self.relay == False:
            done = False
        done = False
        return ob, reward, done, {}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            np.clip(self.robot_skeleton.dq,-10,10)
        ])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]

        return state

    def reset_model(self,init):
        self.dart_world.reset()

        initstate = init['initstate']
        self.relay = init['relay']
        #self.threshold = init['threshold']
        self.tune = init['TuneValueFunction']
        self.training_node = init['node']
        self.model = init['model']
        self.regressorModel = init['RegressorModel']
        self.ComputeThreshold = init['ComputeThreshold']
        pol_name = "Relay" + str(self.training_node) 
        pol_path = './Policies/'+pol_name+".pkl"
        

        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        #print("inittate",initstate)
        qpos[1] = initstate[0]# +  self.np_random.uniform(low=-.005, high=.005, size=1)
        qpos[2] = initstate[1]# +  self.np_random.uniform(low=-.7, high=.7, size=1)
        if qpos[2] < -1.57:
            qpos[2] = -1.57
        #print("qpos",qpos)
        if self.tune:
            print("here")
            qpos[1] = initstate[0]# + self.np_random.uniform(low=-.0075, high=.0075, size=1)#np.random.normal(0,0.03)#
            qpos[2] = initstate[1]# + self.np_random.uniform(low=-0.8, high=0.8, size=1)#  np.random.normal(0,2.2)#
            if qpos[2] < -1.57:
                qpos[2] = -1.57
            qpos[3:] = initstate[2:5]
            qvel = initstate[5:]#self.robot_skeleton.dq + self.np_random.uniform(low=-.015, high=.015, size=self.robot_skeleton.ndofs)
            print("qpos",qpos)


        if self.ComputeThreshold == True:
            print("here")
            qpos[1] = initstate[0]# + self.np_random.uniform(low=-.0075, high=.0075, size=1)#np.random.normal(0,0.03)#
            qpos[2] = initstate[1]# + self.np_random.uniform(low=-0.8, high=0.8, size=1)#  np.random.normal(0,2.2)#
            if qpos[2] < -1.57:
                qpos[2] = -1.57
            qpos[3:] = initstate[2:5]
            qvel = initstate[5:]#self.robot_skeleton.dq + self.np_random.uniform(low=-.015, high=.015, size=self.robot_skeleton.ndofs)
            print("qpos",qpos)
        
        init_angle_dist = np.array([[-np.pi/2,-1.0525],[-np.pi/3,-0.45],[-np.pi/4,-0.25],[-np.pi/6,-0.125],[0,0]])
        x = init_angle_dist[:,0]
        y = init_angle_dist[:,1]
        z = np.polyfit(x, y, 3)
        f = np.poly1d(z)
        qpos[1] = f(qpos[2])
        #qpos[1]-=0.045

        self.set_state(qpos, qvel)

        state = self._get_obs()

        return state

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5
