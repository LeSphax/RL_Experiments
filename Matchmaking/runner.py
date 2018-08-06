import numpy as np


class EnvRunner(object):
    def __init__(self, session, env, policy_estimator, value_estimator, discount_factor=0.99, gae_weighting=0.95):
        self.sess = session
        self.policy_estimator = policy_estimator
        self.value_estimator = value_estimator
        self.env = env
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

            batch['obs'].append(self.obs)
            values = self.value_estimator.get_value(self.obs)
            batch['values'].append(values)

            actions, neglogp_actions = self.policy_estimator.get_action(self.obs)
            batch['neglogp_actions'].append(neglogp_actions)
            batch['actions'].append(actions)

            self.obs, rewards, dones, infos = self.env.step(actions)

            batch['rewards'].append(rewards)
            batch['dones'].append(dones)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)

        advantages = np.zeros_like(batch['rewards'], dtype=float)

        last_values = self.value_estimator.get_value(self.obs)
        batch['values'].append(last_values)
        last_discounted_adv = np.zeros(self.env.num_envs)
        for idx in reversed(range(nb_timesteps)):
            use_future_rewards = 1 - batch['dones'][idx]
            next_value = batch['values'][idx + 1] * use_future_rewards

            td_error = self.discount_factor * next_value + batch['rewards'][idx] - batch['values'][idx]
            advantages[idx] = last_discounted_adv \
                = td_error + self.discount_factor * self.gae_weighting * last_discounted_adv * use_future_rewards

        batch['values'] = batch['values'][:-1]
        returns = advantages + batch['values']

        batch['advantages'] = advantages
        batch['returns'] = returns

        trajectory = {k: flatten_venv(np.asarray(batch[k])) for k in batch}


        return trajectory, epinfos


def flatten_venv(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
