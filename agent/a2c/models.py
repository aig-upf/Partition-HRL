import tensorflow as tf
from tensorflow import keras
import numpy as np
from agent.a2c.losses import Losses
import inspect


class SharedConvLayers(keras.Model):

    idCounter = 0

    def __init__(self):
        super(SharedConvLayers, self).__init__(name="SharedConvLayers")
        self.conv1 = keras.layers.Conv2D(32, 8, (4, 4), padding='VALID', activation='elu', kernel_initializer='he_normal')
        self.conv2 = keras.layers.Conv2D(64, 4, (2, 2), padding='VALID', activation='elu', kernel_initializer='he_normal')
        self.conv3 = keras.layers.Conv2D(64, 3, (1, 1), padding='VALID', activation='elu', kernel_initializer='he_normal')
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(256)

        self.id = SharedConvLayers.idCounter
        SharedConvLayers.idCounter += 1

    def call(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)

        return [x] # super importante ricordati che negli actor e critic modelli stai indicizzando a 0 ho bisogno di questo per la vae observation


class SharedDenseLayers(keras.Model):
    def __init__(self):
        super(SharedDenseLayers, self).__init__(name="SharedDenseLayers")
        self.dense = keras.layers.Dense(128, activation='relu')

    def call(self, x):

        x = self.dense(x)

        return [x]


class CriticNetwork(keras.Model):

    idCounter = 0

    def __init__(self, h_size):
        super(CriticNetwork, self).__init__(name="CriticNetwork")
        self.dense1 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')
        self.out = keras.layers.Dense(1, activation='linear')

        self.id = CriticNetwork.idCounter
        CriticNetwork.idCounter += 1

    def call(self, x):

        x = self.dense1(x)
        x = self.out(x)
        return x


class ActorNetwork(keras.Model):

    idCounter = 0

    def __init__(self, h_size, n_actions):
        super(ActorNetwork, self).__init__(name="ActorNetwork")
        self.dense1 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')
        self.out = keras.layers.Dense(n_actions, activation=keras.activations.softmax)

        self.id = ActorNetwork.idCounter
        ActorNetwork.idCounter += 1

    def call(self, x):

        x = self.dense1(x)
        x = self.out(x)
        return x


class ActorCriticNetwork(keras.Model):

    idCounter = 0

    def __init__(self, critic_model, actor_model, shared_observation_model=None):
        super(ActorCriticNetwork, self).__init__(name="ActorCriticNetwork")
        self.shared_observation_model = shared_observation_model
        self.critic_model = critic_model
        self.actor_model = actor_model

        self.id = ActorCriticNetwork.idCounter
        ActorCriticNetwork.idCounter += 1

    def call(self, x):

        if self.shared_observation_model is not None:

            x = self.shared_observation_model(x)[0] # Just the dense output

        actor = self.actor_model(x)

        critic = self.critic_model(x)

        return actor, critic


class A2CEager:

    def __init__(self, h_size, n_actions, scope_var, device, model_critic, model_actor, learning_rate, shared_observation_model=None, train_observation=False):

        tf.set_random_seed(1)

        with tf.device(device):

            if inspect.isclass(shared_observation_model):
                self.shared_observation_model = shared_observation_model()
            else:
                self.shared_observation_model = shared_observation_model

            if inspect.isclass(model_critic):
                self.model_critic = model_critic(h_size)
            else:
                self.model_critic = model_critic

            if inspect.isclass(model_actor):
                self.model_actor = model_actor(h_size, n_actions)
            else:
                self.model_actor = model_actor

            self.model_actor_critic = ActorCriticNetwork(self.model_critic, self.model_actor, self.shared_observation_model)

            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            self.global_step = tf.Variable(0)

            print("NUMBER OF OBSERVATION NETWORK: ", self.shared_observation_model.idCounter)
            print("NUMBER OF ACTOR NETWORK: ", self.model_critic.idCounter)
            print("NUMBER OF CRITIC NETWORK: ", self.model_actor.idCounter)

    def prediction_actor(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        return self.model_actor_critic(s)[0].numpy()

    def prediction_critic(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        return self.model_actor_critic(s)[1].numpy()


    def grad(self, model_actor_critic, inputs, targets, one_hot_a, advantage, weight_ce, weight_mse = 0.5):

        with tf.GradientTape() as tape:
            softmax_logits, value_critic = model_actor_critic(inputs)
            loss_pg = Losses.reinforce_loss(softmax_logits, one_hot_a, advantage)
            loss_ce = Losses.entropy_exploration_loss(softmax_logits)
            loss_critic = Losses.mse_loss(value_critic, targets)
            loss_value = (weight_mse * loss_critic) + (loss_pg - weight_ce * loss_ce)

        return loss_value, tape.gradient(loss_value, model_actor_critic.trainable_variables)

    def grad_observation(self, model, inputs):

        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_value = Losses.vae_loss(inputs, outputs[3], outputs[1], outputs[2])

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def train(self, s, y, one_hot_a, advantage, weight_ce, weight_mse = 0.5, max_grad_norm=5):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad(self.model_actor_critic, s, y, one_hot_a, advantage, weight_ce, weight_mse)

        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer.apply_gradients(zip(grads, self.model_actor_critic.trainable_variables), self.global_step)

        return [None, None]

    def train_obs(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad_observation(self.shared_observation_model, s)

        print("OBSERVATION", loss_value)

        # grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer_observation.apply_gradients(zip(grads, self.shared_observation_model.trainable_variables),
                                                   self.global_step)

        return [None, None]