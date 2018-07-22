from multiprocessing import Process

from Matchmaking.wrappers import AutoResetEnv, NormalizeEnv
import numpy as np
import utils.keyPoller as kp
import time


class EnvRunner(object):
    def __init__(self, make_session, make_env, policy_estimator, value_estimator, discount_factor=0.9999, gae_weighting=0.95):
        self.sess = make_session()
        self.env = make_env()
        self.policy_estimator = policy_estimator
        self.value_estimator = value_estimator

        self.env = AutoResetEnv(self.env, 300)
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
    def __init__(self, make_session, make_env, make_model, parameters, result_queue, new_policies_queue):
        Process.__init__(self)
        self.parameters = parameters
        self.make_session = make_session
        self.make_env = make_env
        self.make_model = make_model
        self.result_queue = result_queue
        self.new_policies_queue = new_policies_queue

    def run(self):
        import tensorflow as tf
        print(tf.get_default_session())
        policy_estimator, value_estimator = self.make_model()
        runner = EnvRunner(self.make_session, self.make_env, policy_estimator, value_estimator)
        while True:
            decay = self.new_policies_queue.get()
            training_batch = runner.run_timesteps(self.parameters.batch_size)
            self.result_queue.put(training_batch)

            start_time = time.time()
            inds = np.arange(self.parameters.batch_size)
            for _ in range(self.parameters.nb_epochs):

                np.random.shuffle(inds)
                minibatch_size = self.parameters.batch_size // self.parameters.nb_minibatch
                for start in range(0, self.parameters.batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = inds[start:end]

                    entropy, policy_loss = policy_estimator.train_model(
                        obs=training_batch['obs'][mb_inds],
                        actions=training_batch['actions'][mb_inds],
                        neglogp_actions=training_batch['neglogp_actions'][mb_inds],
                        advantages=training_batch['advantages'][mb_inds],
                        clipping=self.parameters.clipping * (1 - decay),
                        learning_rate=self.parameters.learning_rate * (1 - decay),
                    )

                    value_loss = value_estimator.train_model(
                        obs=training_batch['obs'][mb_inds],
                        values=training_batch['values'][mb_inds],
                        returns=training_batch['returns'][mb_inds],
                        clipping=self.parameters.clipping * (1 - decay),
                        learning_rate=self.parameters.learning_rate * (1 - decay),
                    )
            print("Train model", time.time() - start_time)

        return
