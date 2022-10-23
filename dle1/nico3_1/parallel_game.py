import gym
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing



def game(pi_trainable):
    def new_pi_double(init=False):
        card_values = [x for x in range(1,12)]
        possible_scores = [x for x in range(1,23)]  # + перебор
        p = {}
        for cv in card_values:
            for ps in possible_scores:
                p[(ps,cv,False)] = np.array([[1, 1, 1],[0.33,0.33,0.33]]) if init else np.array([0, 0, 0])
                p[(ps,cv,True)] = np.array([[1, 1, 1],[0.33,0.33,0.33]]) if init else np.array([0, 0, 0])
        return(p)

    g = gym.make("Blackjack-v1", render_mode="rgb_array")
    terminated = False
    truncated = False
    observation, info = g.reset()
    pi_current = new_pi_double(init=False)
    used_keys = set()
    actions = []
    states = []
    multiplyer = 1
    while not (terminated or truncated):
        rnd_action=np.random.random()
        score = observation[0] if observation[0] < 22 else 22
        action_array = pi_trainable[(score, observation[1], observation[2])][1]
        key = (score,observation[1], observation[2])
        used_keys.add(key)
        states.append(key)
        #print((score, observation[1]))
        action_ranges = pi_trainable[key][1]
        if rnd_action < action_ranges[0]:
            #do hit
            observation, reward, terminated, truncated, info = g.step(1)
            pi_current[key][0] += 1
            actions.append('take')
            #print((score,observation[1]), pi_current[(score,observation[1])]) 
        elif action_ranges[0] <= rnd_action < (action_ranges[0]+action_ranges[1]):
            #stop
            observation, reward, terminated, truncated, info = g.step(0)
            pi_current[key][1] += 1
            actions.append('check')
        else:
            #double
            pi_current[key][2] += 1
            observation, reward, terminated, truncated, info = g.step(1)
            key = (observation[0],observation[1], observation[2])

            states.append(key)
            if not (terminated or truncated):
                observation, reward, terminated, truncated, info = g.step(0)
            multiplyer=2   
            actions.append('double')
    return [pi_current, reward, actions, used_keys, ]