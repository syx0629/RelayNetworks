import numpy as np
from gym import utils
from gym.envs.dart import dart_env
import baselines.common.tf_util as U
import tensorflow as tf
from baselines.ppo1.mlp_policy import MlpPolicy
from gym import wrappers,spaces
import pickle

class DartWalker2dEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0]*6,[-1.0]*6])
        self.action_scale = np.array([100, 100, 20, 100, 100, 20])
        obs_dim = 17
        self.first_pass = True
        dart_env.DartEnv.__init__(self, 'walker2d.skel', 4, obs_dim, self.control_bounds, disableViewer=False)
        high = np.inf*np.ones(obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)
        self.action_space = spaces.Box(self.control_bounds[1], self.control_bounds[0])
        try:
            self.dart_world.set_collision_detector(3)
        except Exception as e:
            print('Does not have ODE collision detector, reverted to bullet collision detector')
            self.dart_world.set_collision_detector(2)

        utils.EzPickle.__init__(self)

    def step(self, a):
        pre_state = [self.state_vector()]

        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[3:] = clamped_control * self.action_scale
        posbefore = self.robot_skeleton.q[0]
        self.do_simulation(tau, self.frame_skip)
        posafter,ang = self.robot_skeleton.q[0,2]
        height = self.robot_skeleton.bodynodes[2].com()[1]

        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()

        alive_bonus = 1.0
        vel = (posafter - posbefore) / self.dt
        reward = vel
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()

        # uncomment to enable knee joint limit penalty
        '''joint_limit_penalty = 0
        for j in [-2, -5]:
            if (self.robot_skeleton.q_lower[j] - self.robot_skeleton.q[j]) > -0.05:
                joint_limit_penalty += abs(1.5)
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[j]) < 0.05:
                joint_limit_penalty += abs(1.5)

        reward -= 5e-1 * joint_limit_penalty'''

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .8) and (height < 2.0) and (abs(ang) < 1.0))

        ob = self._get_obs()

        if self.relay:
            c,vpred = self.pol.act(False,ob)
            modelpred = self.model.predict(vpred)
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
        self.ComputeThreshold = init['ComputeThreshold']
        pol_name = "Relay" + str(self.training_node) 
        pol_path = './Policies/'+pol_name+".pkl"
        if self.first_pass:
            if self.relay:
                self.pol = MlpPolicy(pol_name+ "pi",self.observation_space,self.action_space,hid_size=64,num_hid_layers=2)
                sess = tf.get_default_session()
                with open(pol_path,"rb") as fp:
                    self.par = pickle.load(fp)
                for item in self.pol.get_variables():
                    #print(item.name)
                    if "Adam" not in item.name:
                        if pol_name+"pi" in item.name:
                            obj = item.assign(self.par[item.name])
                            sess.run(obj)
                self.first_pass = False




        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        #print("inittate",initstate)
        qpos[1] = initstate[0] +  self.np_random.uniform(low=-.005, high=.005, size=1)
        qpos[2] = initstate[1] +  self.np_random.uniform(low=-.1, high=.1, size=1)

        if self.tune:
            print("here")
            qpos[1] = initstate[0] + self.np_random.uniform(low=-.0075, high=.0075, size=1)#np.random.normal(0,0.03)#
            qpos[2] = initstate[1] + self.np_random.uniform(low=-0.5, high=0.5, size=1)#  np.random.normal(0,2.2)#
            qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.015, high=.015, size=self.robot_skeleton.ndofs)
            print("qpos",qpos)


        if self.ComputeThreshold == True:
            qpos[1] = initstate[0] + self.np_random.uniform(low=-.0075, high=.0075, size=1)#np.random.normal(0,0.03)#
            qpos[2] = initstate[1] + self.np_random.uniform(low=-0.35, high=0.35, size=1)#  np.random.normal(0,2.2)#
            
            qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.015, high=.015, size=self.robot_skeleton.ndofs)
            print("qpos compute",qpos)

        self.set_state(qpos, qvel)

        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5
