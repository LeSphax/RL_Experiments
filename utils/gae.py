import numpy as np


def gae(rewards, values, next_value, discount_factor=0.99, gae_weighting=0.95):
    returns = np.zeros_like(rewards, dtype=float)
    advantages = np.zeros_like(rewards, dtype=float)

    values = np.append(values, next_value)
    last_discounted_advantage = 0
    for idx in reversed(range(len(rewards))):
        td_error = discount_factor * values[idx+1] + rewards[idx] - values[idx]
        # print(td_error)
        advantages[idx] = last_discounted_advantage = td_error + discount_factor * gae_weighting * last_discounted_advantage
    returns = advantages + values[:-1]

    return advantages, returns





def test1():
    rewards = [0, 0, 1, 0, 0, 2]
    values = [0, 1, 0, 2, 2, 0]
    next_value = 3
    advantages, returns = gae(rewards, values, next_value, 0.5, 0.9)

    print(np.round(advantages, 2))
    expected_advantages = [0.35, -0.34, 1.46, -1.19, -0.43, 3.5]
    assert np.array_equal(np.round(advantages, 2), expected_advantages)

    print(np.round(returns, 2))
    expected_returns = [0.35, 0.66, 1.46, 0.81, 1.58, 3.5]
    assert np.array_equal(np.round(returns, 2), expected_returns)


def test2():
    values = [4.9384604 , 4.8105726 , 4.397197  , 3.2387118,  1.7485781,  0.49508286,
              0.49508286, 0.49508286, 0.49508286, 0.49508286, 0.49508286, 0.49508286,
              0.49508286, 0.49508286, 0.49508286, 0.49508286, 0.49508286, 0.49508286,
              0.49508286, 0.49508286,]
    rewards = [0.9999892532466681, 0.997897100326244, 0.9869773168251323, 0.8831162576956572, 0.7990210862671885, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    advantages, returns = gae(rewards, values, 0.49508286, 0.99, 0.95)
    print(advantages)


if __name__ == "__main__":
    test2()