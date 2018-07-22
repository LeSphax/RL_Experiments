import multiprocessing
import time
import random


class Consumer(multiprocessing.Process):

    def __init__(self, result_queue, new_policies_queue):
        multiprocessing.Process.__init__(self)
        self.result_queue = result_queue
        self.new_policies_queue = new_policies_queue
        self.value = 0

    def run(self):
        while True:
            time.sleep(random.random())
            self.result_queue.put(self.value)
            self.value = self.new_policies_queue.get()
        return


if __name__ == '__main__':
    start=time.time()
    # Establish communication queues
    new_policies = multiprocessing.Queue()
    results = multiprocessing.Queue()

    # Start consumers
    num_consumers = 1
    print('Creating %d consumers' % num_consumers)
    consumers = [Consumer(results, new_policies)
                 for i in range(num_consumers)]
    for w in consumers:
        w.start()

    policy = 0
    while True:
        time.sleep(random.random())
        new_policies.put(policy)
        result = results.get()
        policy += 1
        print('Result:', result)
