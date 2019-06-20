import tensorflow as tf
from tensorflow import keras
import numpy as np
from agent.losses import Losses
import tensorflow.contrib.slim as slim

import matplotlib.pyplot as plt


class SharedConvLayers(keras.Model):
    def __init__(self):
        super(SharedConvLayers, self).__init__(name="SharedConvLayers")
        self.conv1 = keras.layers.Conv2D(32, 8, (4, 4), padding='VALID', activation='elu')
        self.conv2 = keras.layers.Conv2D(64, 4, (2, 2), padding='VALID', activation='elu')
        self.conv3 = keras.layers.Conv2D(64, 3, (1, 1), padding='VALID', activation='elu')
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(512)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)

        return x


class SharedDenseLayers(keras.Model):
    def __init__(self):
        super(SharedDenseLayers, self).__init__(name="SharedDenseLayers")
        self.dense = keras.layers.Dense(64, activation='elu')

    def call(self, x):

        x = self.dense(x)

        return x


class CriticNetwork(keras.Model):
    def __init__(self, h_size, shared_observation_model=None):
        super(CriticNetwork, self).__init__(name="CriticNetwork")
        self.dense1 = keras.layers.Dense(h_size, activation='elu')
        self.out = keras.layers.Dense(1, activation='linear')
        self.shared_observation_model = shared_observation_model

    def call(self, x):

        if self.shared_observation_model is not None:
            x = self.shared_observation_model(x)

        x = self.dense1(x)
        x = self.out(x)
        return x


class ActorNetwork(keras.Model):
    def __init__(self, h_size, n_actions, shared_observation_model=None):
        super(ActorNetwork, self).__init__(name="ActorNetwork")
        self.dense1 = keras.layers.Dense(h_size, activation='elu')
        self.out = keras.layers.Dense(n_actions, activation=keras.activations.softmax)
        self.shared_observation_model = shared_observation_model

    def call(self, x):

        print(x.shape)
        plt.imshow(x[0])
        plt.show()

        if self.shared_observation_model is not None:

            x = self.shared_observation_model(x)

        print(x.numpy().shape)

        x = self.dense1(x)
        x = self.out(x)
        return x


class A2CEager:

    def __init__(self, input_shape, h_size, n_actions, scope_var, device, model_critic, model_actor, learning_rate_actor, learning_rate_critic, shared_observation_model=None):

        print(shared_observation_model)

        tf.set_random_seed(1)
        with tf.device(device):
            self.shared_observation_model = shared_observation_model
            self.model_critic = model_critic(h_size, self.shared_observation_model)
            self.model_actor = model_actor(h_size, n_actions, self.shared_observation_model)
            dummy_x = tf.zeros([1] + input_shape[1::])

            self.model_critic._set_inputs(dummy_x)
            self.model_actor._set_inputs(dummy_x)

            print("\n OBSERVATION ENCODING MODEL \n")

            slim.model_analyzer.analyze_vars(self.shared_observation_model.trainable_variables, print_info=True)

            print("\n ACTOR MODEL \n")

            slim.model_analyzer.analyze_vars(self.model_actor.trainable_variables, print_info=True)

            print("\n CRITIC MODEL \n")

            slim.model_analyzer.analyze_vars(self.model_critic.trainable_variables, print_info=True)

            self.optimizer_critic = tf.train.AdamOptimizer(learning_rate=learning_rate_critic)
            self.optimizer_actor = tf.train.RMSPropOptimizer(learning_rate=learning_rate_actor)
            self.global_step = tf.Variable(0)

    def prediction(self, s):

        s = np.array(s, dtype=np.float32)

        a = self.model_actor(s)

        return np.argmax(a, 1)

    def prediction_actor(self, s):

        s = np.array(s, dtype=np.float32)

        return self.model_actor(s).numpy()

    def prediction_critic(self, s):

        s = np.array(s, dtype=np.float32)

        return self.model_critic(s).numpy()

    def grad_actor(self, model, inputs, one_hot_a, advantage, weight_ce=0.01):

        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_pg = Losses.reinforce_loss(outputs, one_hot_a, advantage)
            loss_ce = Losses.entropy_exploration_loss(outputs)
            loss_value = loss_pg - weight_ce * loss_ce

        # print(outputs.numpy(), loss_pg.numpy(), loss_ce.numpy(), loss_value.numpy())

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def grad_critic(self, model, inputs, targets, weight_mse=0.5):

        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_value = weight_mse * Losses.mse_loss(outputs, targets)

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def train_critic(self, s, y, max_grad_norm=1.):

        s = np.array(s, dtype=np.float32)

        loss_value, grads = self.grad_critic(self.model_critic, s, y)

        # print("CRITIC", loss_value)

        # grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer_critic.apply_gradients(zip(grads, self.model_critic.trainable_variables), self.global_step)

        return [None, None]

    def train_actor(self, s, one_hot_a, advantage, weight_ce=0, max_grad_norm=0.5):
        s = np.array(s, dtype=np.float32)

        loss_value, grads = self.grad_actor(self.model_actor, s, one_hot_a, advantage, weight_ce)

        # print("ACTOR", loss_value)

        # grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer_actor.apply_gradients(zip(grads, self.model_actor.trainable_variables),
                                             self.global_step)

        return [None, None]