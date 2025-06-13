import random
import pickle
class QLearningAgent:
    def __init__(self, actions, alpha=0.7, gamma=0.95, epsilon=0.05):
        self.q_table = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.q_table.get(state, {}).get(action, 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = {a: self.get_q(state, a) for a in self.actions}
            max_q = max(q_values.values())
            max_actions = [a for a, q in q_values.items() if q == max_q]
            return random.choice(max_actions)

    def update_q(self, state, action, reward, next_state):
        current_q = self.get_q(state, action)
        next_max_q = max([self.get_q(next_state, a) for a in self.actions], default=0.0)
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * next_max_q)
        self.q_table.setdefault(state, {})[action] = new_q

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * 0.995)