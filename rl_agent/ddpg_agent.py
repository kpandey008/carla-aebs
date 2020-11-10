import tensorflow as tf
from keras import backend as K


import numpy as np
from rl_agent.ActorNetwork import ActorNetwork
from rl_agent.CriticNetwork import CriticNetwork
from rl_agent.OU import OU
from rl_agent.ReplayBuffer import ReplayBuffer
import json

state_dim = 2
action_dim = 1
BATCH_SIZE = 32
GAMMA = 0.99
TAU = 0.001
LRA = 0.0001
LRC = 0.001
BUFFER_SIZE = 100000

class ddpgAgent():
    def __init__(self, Testing=False):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)
        self.testing = Testing

        self.actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
        try:
            self.actor.model.load_weights("./nn_model/controller/actormodel.h5")
            self.actor.target_model.load_weights("./nn_model/controller/actormodel.h5")
            print("Load actor model successfully")
        except:
            print("Cannot find actor weights in this directory")
        
        if self.testing is False:
            self.buff = ReplayBuffer(BUFFER_SIZE)
            self.OU = OU()
            self.critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
            try:
                self.critic.model.load_weights("./nn_model/controller/criticmodel.h5")
                self.critic.target_model.load_weights("./nn_model/controller/criticmodel.h5")
                print("Load critic model successfully")
            except:
                print("Cannot find critic weights in this directory")
    
    def getAction(self, state, epsilon):
        action = np.zeros([1, action_dim])
        noise = np.zeros([1, action_dim])
        action_original = self.actor.model.predict(state.reshape(1, state.shape[0]))
        if self.testing is False:
            noise[0][0] = (1.0-float(self.testing)) * max(epsilon, 0) * self.OU.function(action_original[0][0], 0.2, 1.00, 0.10)
        action[0][0] = action_original[0][0] + noise[0][0]
        if action[0][0] < 0.0:
            action[0][0] = 0.0
        if action[0][0] > 1.0:
            action[0][0] = 1.0
        print("NN Controller: {:5.4f}, Noise NN Controller: {:5.4f}".format(action_original[0][0], action[0][0]))
        return action
    
    def storeTrajectory(self, s, a, r, s_, done):
        self.buff.add(s, a[0], r, s_, done)
    
    def learn(self):
        batch = self.buff.getBatch(BATCH_SIZE)
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])
        y_t = np.asarray([e[1] for e in batch])

        target_q_values = self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states)])  

        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + GAMMA*target_q_values[k]

        loss = self.critic.model.train_on_batch([states, actions], y_t)
        #print("critic loss value: {:5.4f}".format(loss))
        a_for_grad = self.actor.model.predict(states)
        grads = self.critic.gradients(states, a_for_grad)
        self.actor.train(states, grads)
        self.actor.target_train()
        self.critic.target_train()
    
    def save_model(self):
        print("Saving model now...")
        self.actor.model.save_weights("./nn_model/controller/actormodel.h5", overwrite=True)
        with open("./nn_model/controller/actormodel.json", "w") as outfile:
            json.dump(self.actor.model.to_json(), outfile)
        
        self.critic.model.save_weights("./nn_model/controller/criticmodel.h5", overwrite=True)
        with open("./nn_model/controller/criticmodel.json", "w") as outfile:
            json.dump(self.critic.model.to_json(), outfile)
        
