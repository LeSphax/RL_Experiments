import time
from multiprocessing import Process, Pipe

from Matchmaking.wrappers import AutoResetEnv


def make_renderer(*args):
    parent_pipe, child_pipe = Pipe()
    renderer = EnvRendererProcess(*args, child_pipe)
    return parent_pipe, renderer


class EnvRendererProcess(Process):
    def __init__(self, make_session, make_env, make_model, new_policies_queue, instruction_pipe):
        Process.__init__(self)
        self.make_session = make_session
        self.make_env = make_env
        self.make_model = make_model
        self.instruction_pipe = instruction_pipe
        self.new_policies_queue = new_policies_queue

    def run(self):
        self.make_session()
        env = self.make_env()
        env = AutoResetEnv(env, 500)
        policy_estimator, value_estimator = self.make_model(env)
        render = False

        obs = env.reset()
        while True:
            if self.instruction_pipe.poll():
                msg = self.instruction_pipe.recv()
                if msg == 'start':
                    print("Receive start")
                    policy_weights, value_weights = self.new_policies_queue.get()
                    policy_estimator.set_weights(policy_weights)
                    value_estimator.set_weights(value_weights)
                    render = True
                else:
                    render = False

            if render:
                env.render()
                action, neglogp_action = policy_estimator.get_action(obs)

                obs, reward, done, info = env.step(action)
