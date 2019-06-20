import tensorflow as tf


class Losses:

    # Huber loss.
    # https://en.wikipedia.org/wiki/Huber_loss

    @staticmethod
    def huber_loss(x, y):

        return tf.losses.huber_loss(labels=y, predictions=x, delta=1.0)

    @staticmethod
    def huber_loss_importance_weight(x, y, weights):

        return tf.losses.huber_loss(labels=y, predictions=x, weights=weights, delta=1.0)

    @staticmethod
    def reinforce_loss(logits, one_hot_a, advantage):

        neg_log_policy = - tf.log(tf.clip_by_value(logits, 1e-7, 1))

        reinforce_loss = tf.reduce_mean(tf.reduce_sum(neg_log_policy * one_hot_a, axis=1) * advantage)

        return reinforce_loss

    @staticmethod
    def mse_loss(x, y):

        loss = tf.losses.mean_squared_error(y, x)

        return loss

    @staticmethod
    def entropy_exploration_loss(x):

        neg_log_policy = - tf.log(tf.clip_by_value(x, 1e-7, 1))
        loss = tf.reduce_mean(tf.reduce_sum(x * neg_log_policy, axis=1))

        return loss
