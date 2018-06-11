from game_state import QuestionGameState


class QAAgent(object):

    def __init__(self, sess, depth_scope):
        self.game_state = QuestionGameState(sess=sess, depth_scope=depth_scope)
        self.sess = sess
        self.depth_scope = depth_scope

    def reset(self, seed=None, test_ind=None):
        raise NotImplementedError

    def inference(self):
        raise NotImplementedError

    def get_next_output(self):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError

    def get_action(self, action_ind):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def draw_state(self):
        raise NotImplementedError

