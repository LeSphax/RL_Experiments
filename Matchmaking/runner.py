from multiprocessing import Process

from Matchmaking.wrappers import AutoResetEnv, NormalizeEnv, TensorboardEnv
import numpy as np
import utils.keyPoller as kp
import time
from datetime import datetime

class EnvRunner(object):
    def __init__(self, session, env, policy_estimator, value_estimator, discount_factor=0.9999, gae_weighting=0.95):
        self.sess = session
        self.policy_estimator = policy_estimator
        self.value_estimator = value_estimator
        self.env = AutoResetEnv(env, 500)
        if self.policy_estimator.normalize():
            self.env = NormalizeEnv(self.env)
        self.obs = self.env.reset()
        self.discount_factor = discount_factor
        self.gae_weighting = gae_weighting

    def run_timesteps(self, nb_timesteps):
        batch = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'dones': [],
            'neglogp_actions': [],
            'advantages': [],
            'returns': []
        }
        epinfos = []

        for t in range(nb_timesteps):
            # kp.checkKeyStrokes(None, None)
            # if kp.render:
            #     self.env.render()

            batch['obs'].append(self.obs)
            value = self.value_estimator.get_value(self.obs)
            batch['values'].append(value)

            action, neglogp_action = self.policy_estimator.get_action(self.obs)

            self.obs, reward, done, info = self.env.step(action)

            batch['neglogp_actions'].append(neglogp_action)
            batch['actions'].append(action)
            batch['rewards'].append(reward)
            batch['dones'].append(done)

            maybeepinfo = info.get('episode')
            if maybeepinfo:
                epinfos.append(maybeepinfo)

        advantages = np.zeros_like(batch['rewards'], dtype=float)

        next_value = 0 if done else self.value_estimator.get_value(self.obs)
        batch['values'] = np.append(batch['values'], next_value)
        last_discounted_adv = 0
        for idx in reversed(range(nb_timesteps)):
            if batch['dones'][idx] == 1:
                next_value = 0
                use_last_discounted_adv = 0
            else:
                next_value = batch['values'][idx + 1]
                use_last_discounted_adv = 1

            td_error = self.discount_factor * next_value + batch['rewards'][idx] - batch['values'][idx]
            advantages[idx] = last_discounted_adv = td_error + self.discount_factor * self.gae_weighting * last_discounted_adv * use_last_discounted_adv
        returns = advantages + batch['values'][:-1]

        batch['advantages'] = advantages
        batch['returns'] = returns

        return {k: np.asarray(batch[k]) for k in batch}, epinfos


class EnvRunnerProcess(Process):
    def __init__(self, process_id, make_session, make_env, make_model, parameters, result_queue, new_policies_queue):
        Process.__init__(self)
        self.process_id = process_id
        self.parameters = parameters
        self.make_session = make_session
        self.make_env = make_env
        self.make_model = make_model
        self.result_queue = result_queue
        self.new_policies_queue = new_policies_queue

    def run(self):
        import tensorflow as tf
        sess = self.make_session()
        env = self.make_env()
        env = TensorboardEnv(env, './train/{env_name}/{name}_{id}_{date}'.format(env_name='CartPole-v1', name="Test_Each_Worker", id=str(self.process_id), date=datetime.now().strftime("%m%d-%H%M")))

        policy_estimator, value_estimator = self.make_model(env)

        runner = EnvRunner(sess, env, policy_estimator, value_estimator)
        while True:
            print("Wait for weights")
            start_time = time.time()
            decay, policy_weights, value_weights = self.new_policies_queue.get()
            print("Got weights", time.time() - start_time)
            start_time = time.time()
            policy_estimator.set_weights(policy_weights)
            value_estimator.set_weights(value_weights)
            print("Set weights", time.time() - start_time)

            learning_rate = self.parameters.learning_rate * (1 - decay)
            clipping = self.parameters.clipping * (1 - decay)

            start_time = time.time()
            training_batch, epinfos = runner.run_timesteps(self.parameters.batch_size)
            print("Run batch", time.time() - start_time)

            start_time = time.time()
            value_gradients = []
            policy_gradients = []
            inds = np.arange(self.parameters.batch_size)
            for _ in range(self.parameters.nb_epochs):

                np.random.shuffle(inds)
                minibatch_size = self.parameters.batch_size // self.parameters.nb_minibatch
                for start in range(0, self.parameters.batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = inds[start:end]

                    entropy, policy_loss, policy_gradient_step = policy_estimator.get_gradients(
                        obs=training_batch['obs'][mb_inds],
                        actions=training_batch['actions'][mb_inds],
                        neglogp_actions=training_batch['neglogp_actions'][mb_inds],
                        advantages=training_batch['advantages'][mb_inds],
                        clipping=clipping,
                        learning_rate=learning_rate,
                    )
                    policy_estimator.apply_gradients(policy_gradient_step, learning_rate)
                    policy_gradients.append(policy_gradient_step)

                    value_loss, value_gradient_step = value_estimator.get_gradients(
                        obs=training_batch['obs'][mb_inds],
                        values=training_batch['values'][mb_inds],
                        returns=training_batch['returns'][mb_inds],
                        clipping=clipping,
                        learning_rate=learning_rate,
                    )
                    value_estimator.apply_gradients(value_gradient_step, learning_rate)
                    value_gradients.append(value_gradient_step)

            print("Train model", time.time() - start_time)

            policy_gradients = np.sum(policy_gradients, axis=0)
            value_gradients = np.sum(value_gradients, axis=0)

            # print(np.shape(policy_gradients))
            # print(np.shape(np.sum(policy_gradients, axis=0)))
            self.result_queue.put((policy_gradients, value_gradients, epinfos))


        return
