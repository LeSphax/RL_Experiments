import numpy as np

from Matchmaking.wrappers import AutoResetEnv, NormalizeEnv


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
            advantages[idx] = last_discounted_adv = td_error \
                                                    + self.discount_factor * self.gae_weighting * last_discounted_adv * use_last_discounted_adv
        returns = advantages + batch['values'][:-1]

        batch['advantages'] = advantages
        batch['returns'] = returns

        return {k: np.asarray(batch[k]) for k in batch}, epinfos