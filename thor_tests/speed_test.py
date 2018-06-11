from utils import game_util
import time


def test_random_initialize_speed(env):
    print('Reset/Initialize speed')
    t_start = time.time()
    time_count = 0
    for scene in range(1,31):
        time_count += 1
        game_util.reset(env, scene)
        event = env.random_initialize(random_seed=0)
    t_end = time.time()
    total_time = t_end - t_start
    print('Total time %.3f' % (total_time))
    print('Mean time  %.3f' % (total_time / time_count))
    print('FPS =      %.3f' % (time_count / total_time))
    print('')


def test_movement_speed(env):
    print('Movement speed')
    time_count = 0
    total_time = 0
    for scene in range(1,31):
        game_util.reset(env, scene)
        event = env.random_initialize(random_seed=0)
        t_start = time.time()
        for jj in range(10):
            event = env.step({'action': 'MoveAhead'})
            event = env.step({'action': 'RotateRight'})
            time_count += 2
        total_time += time.time() - t_start
        print('Total time %.3f' % (total_time))
        print('Mean time  %.3f' % (total_time / time_count))
        print('FPS =      %.3f' % (time_count / total_time))
        print('')


def run_tests(env=None):
    create_env = (env is None)
    if create_env:
        env = game_util.create_env()
    test_random_initialize_speed(env)
    test_movement_speed(env)

if __name__ == '__main__':
    run_tests()
