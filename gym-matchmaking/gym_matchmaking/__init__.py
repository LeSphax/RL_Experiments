from gym.envs.registration import register

register(
    id='Matchmaking-v0',
    entry_point='gym_matchmaking.envs:MatchmakingEnv',
)
register(
    id='Matchmaking-harder-v0',
    entry_point='gym_matchmaking.envs:MatchmakingHarderEnv',
)