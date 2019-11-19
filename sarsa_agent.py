# saras agent
import numpy as np
#   Should try to peramatize action space
#   each of the 4 action_spaces have range [-1,1]
#   Should allow np.array([i for i in range(-10,11)])/10
#

# from stackoverflow
def softmax(x):
    z = x - max(x)
    num = np.exp(z)
    dem = np.sum(num)
    return num/dem


class agent:
    def __init__(self,env, weights=None):
        self.num_features = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]
        self.action_map = np.array([i for i in range(-10,11)])/10
        self.gamma = 0.9
        self.nu = 0.2
        # have a fully connected layer
        # output has 4 action spaces each with 21 possible actions so a total of 84 output nodes
        # for the output layer, let the first 21 nodes be computed with a softmax for action 1, next for action 2, and so on
        if weights == None:
            weights = np.random.random((self.action_space * self.action_map.shape[0], self.num_features + 1))
        self.weights = weights
        self.MAX_ITER = 500
        self.epsilon = 0.2


    def compute_action(self,ob, env, greedy):
        # find random action with probably epsilon
        if np.random.random() < self.epsilon and not greedy:
            r = np.random.randint(0, self.action_map.shape[0], 4)
            return np.array([self.action_map[i] for i in r])

        ob_mod = np.append(ob,1)
        output_layer = np.matmul(self.weights, ob_mod)
        a_layers = np.split(output_layer, self.action_space)
        action = np.array([])
        for outputs in a_layers:
            # get max index, then map it to action values
            # soft_outputs = softmax(outputs)
            argmax = np.argmax(outputs)
            action = np.append(action, self.action_map[argmax])
        return action


    def Q(self,s,a):
        s_mod = np.append(s, 1)
        output_layer = np.matmul(self.weights, s_mod)
        a_layers = np.split(output_layer, self.action_space)
        Q = 0
        for i in range(len(a_layers)):
            # a_layers[i] = softmax(a_layers[i])
            Q = Q + a_layers[i][np.argwhere(self.action_map == a[i])[0,0]]/len(a_layers)
        return Q


    def episode(self,env, output, greedy):
        done = False
        iteration_count = 0
        s = env.reset()
        a  = self.compute_action(s, env, greedy)
        while not done:
            if output:
                env.render()
            s1, r, done, _ = env.step(a)
            if done:
                break
            #print("reward: {}".format(r))
            a1 = self.compute_action(s1, env, greedy)
            # print("Q(s,a): {}".format(self.Q(s,a)))
            # print("delta: {}".format(delta))
            if not greedy:
                delta = r + (self.gamma * self.Q(s1,a1)) - self.Q(s,a)
                s_mod = np.append(s,1)
                for i in range(self.weights.shape[0]):
                    self.weights[i] = self.weights[i] + self.nu * delta * s_mod[i%s_mod.shape[0]]
            s = s1
            a = a1
            iteration_count = iteration_count + 1
            done = done or iteration_count > self.MAX_ITER
        # print("end of episode\n\n")
        if output:
            env.close()

    def learn(self,env):
        for i in range(600):
            self.episode(env, i%100 == 0, False)


    def run_greedy(self,env):
        for i in range(5):
            self.episode(env, True, True)
