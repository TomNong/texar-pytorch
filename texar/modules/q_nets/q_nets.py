#
"""
Various Q networks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.modules.q_nets.q_net_base import QNetBase
from texar.modules.networks import FeedForwardNetwork


class NatureQNet(QNetBase):
    """
    Module of Basic DQN

    ref:  "Human-level control through deep reinforcement learning
    (Nature 518(7540)))
    """
    def __init__(self, hparams=None):
        QNetBase.__init__(self, hparams=hparams)
        with tf.variable_scope(self.variable_scope):
            self.qnet = FeedForwardNetwork(
                hparams=self.hparams.network_hparams)
            self.target = FeedForwardNetwork(
                hparams=self.hparams.network_hparams)

    @staticmethod
    def default_hparams():
        return {
            'name': 'nature_q_net',
            'network_hparams': FeedForwardNetwork.default_hparams()
        }

    def _build(self, inputs): # pylint: disable=arguments-differ
        qnet_result, target_result = self.qnet(inputs), self.target(inputs)

        if not self._built:
            self._add_internal_trainable_variables()
            self._add_trainable_variable(self.qnet.trainable_variables)
            self._add_trainable_variable(self.target.trainable_variables)
            self._built = True

        return qnet_result, target_result

    def copy_qnet_to_target(self):
        """Copy the parameters from qnet to target.

        Returns:
            A list of assign tensors
        """
        result = []
        for i in range(len(self.qnet.trainable_variables)):
            result.append(tf.assign(ref=self.target.trainable_variables[i],
                                    value=self.qnet.trainable_variables[i]))
        return result