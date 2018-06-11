from utils import game_util

def assert_successful_action(env, action):
    event = env.step(action)
    assert(event.metadata['lastActionSuccess']), 'Action Failed: ' + str(action)
    return event

def assert_failed_action(env, action):
    event = env.step(action)
    assert(not event.metadata['lastActionSuccess']), 'Action Succeeded when it should have failed: ' + str(action)
    return event

def test_random_initialize(env):
    event = env.reset('FloorPlan1')
    env.step(dict(action='Initialize', gridSize=0.25, cameraY=0.75, qualitySetting='High'))
    event = env.random_initialize(random_seed=0)
    event = env.step({'action' : 'RotateRight'})
    event = env.step({'action' : 'RotateRight'})
    event = assert_successful_action(env,
            {'action' : 'OpenObject', 'objectId' : 'Microwave|-00.25|+01.71|-02.65'})
    # Can't open twice in a row
    event = assert_failed_action(env,
            {'action' : 'OpenObject', 'objectId' : 'Microwave|-00.25|+01.71|-02.65'})
    visible_objects = {obj['objectId'] for obj in event.metadata['objects'] if obj['visible']}
    assert(visible_objects == {'Microwave|-00.25|+01.71|-02.65', 'Cabinet|+00.68|+02.02|-02.46'})
    event = assert_successful_action(env,
            {
                'action' : 'TeleportFull',
                'x' : -1,
                'y': 1,
                'z': 1,
                'rotation': 270,
                'horizon' : 30
            })
    event = assert_successful_action(env,
            {'action' : 'OpenObject', 'objectId': 'Fridge|-01.89|+00.00|+01.07'})
    visible_objects = {obj['objectId'] for obj in event.metadata['objects'] if obj['visible']}
    assert(visible_objects ==
        {
            'Fridge|-01.89|+00.00|+01.07',
        })

def test_random_initialize_randomize_open(env):
    event = env.reset('FloorPlan1')
    env.step(dict(action='Initialize', gridSize=0.25, cameraY=0.75, qualitySetting='High'))
    event = env.random_initialize(random_seed=0, randomize_open=True)
    event = env.step({'action' : 'RotateRight'})
    event = env.step({'action' : 'RotateRight'})
    event = assert_failed_action(env,
            {'action' : 'OpenObject', 'objectId' : 'Microwave|-00.25|+01.71|-02.65'})
    visible_objects = {obj['objectId'] for obj in event.metadata['objects'] if obj['visible']}
    assert(visible_objects == {'Microwave|-00.25|+01.71|-02.65',})
    assert(game_util.get_object('Microwave|-00.25|+01.71|-02.65', event.metadata)['isopen'])
    event = assert_successful_action(env,
            {'action' : 'CloseObject', 'objectId' : 'Microwave|-00.25|+01.71|-02.65'})
    assert(not game_util.get_object('Microwave|-00.25|+01.71|-02.65', event.metadata)['isopen'])
    event = assert_successful_action(env,
            {
                'action' : 'TeleportFull',
                'x' : -1,
                'y': 1,
                'z': 1,
                'rotation': 270,
                'horizon' : 30
            })
    assert(game_util.get_object('Fridge|-01.89|+00.00|+01.07', event.metadata)['isopen'])
    event = assert_failed_action(env,
            {'action' : 'OpenObject', 'objectId': 'Fridge|-01.89|+00.00|+01.07'})
    visible_objects = {obj['objectId'] for obj in event.metadata['objects'] if obj['visible']}
    assert(visible_objects ==
        {
            'Fridge|-01.89|+00.00|+01.07',
        })

def test_random_initialize_with_remove_prob(env):
    event = env.reset('FloorPlan1')
    env.step(dict(action='Initialize', gridSize=0.25, cameraY=0.75, qualitySetting='High'))
    event = env.random_initialize(random_seed=4, remove_prob=0)
    event = env.step({'action' : 'RotateRight'})
    event = env.step({'action' : 'RotateRight'})
    event = assert_successful_action(env,
            {'action' : 'OpenObject', 'objectId' : 'Microwave|-00.25|+01.71|-02.65'})
    visible_objects = {obj['objectId'] for obj in event.metadata['objects'] if obj['visible']}
    assert(visible_objects ==
        {
            'Bowl|+00.97|+01.64|-02.59_copy_0',
            'Microwave|-00.25|+01.71|-02.65',
            'Cabinet|+00.68|+02.02|-02.46'
        })
    event = env.reset('FloorPlan1')
    event = env.random_initialize(random_seed=4, remove_prob=0.5)
    event = env.step({'action' : 'RotateRight'})
    event = env.step({'action' : 'RotateRight'})
    event = assert_successful_action(env,
            {'action' : 'OpenObject', 'objectId' : 'Microwave|-00.25|+01.71|-02.65'})
    visible_objects = {obj['objectId'] for obj in event.metadata['objects'] if obj['visible']}
    assert(visible_objects ==
        {
            'Microwave|-00.25|+01.71|-02.65',
            'Mug|-02.00|+01.65|+00.06_copy_0',
            'Cabinet|+00.68|+02.02|-02.46'
        })

    event = env.reset('FloorPlan1')
    event = env.random_initialize(random_seed=4, remove_prob=0.9)
    event = env.step({'action' : 'RotateRight'})
    event = env.step({'action' : 'RotateRight'})
    event = assert_successful_action(env,
            {'action' : 'OpenObject', 'objectId' : 'Microwave|-00.25|+01.71|-02.65'})
    visible_objects = {obj['objectId'] for obj in event.metadata['objects'] if obj['visible']}
    assert(visible_objects ==
        {
            'Microwave|-00.25|+01.71|-02.65',
            'Cabinet|+00.68|+02.02|-02.46'
        })


def test_random_initialize_with_repeats(env):
    event = env.reset('FloorPlan1')
    env.step(dict(action='Initialize', gridSize=0.25, cameraY=0.75, qualitySetting='High'))
    event = env.random_initialize(random_seed=1, remove_prob=0.5, max_num_repeats=5)
    event = env.step({'action' : 'RotateRight'})
    event = env.step({'action' : 'RotateRight'})
    event = assert_successful_action(env,
            {'action' : 'OpenObject', 'objectId' : 'Microwave|-00.25|+01.71|-02.65'})
    visible_objects = {obj['objectId'] for obj in event.metadata['objects'] if obj['visible']}
    assert(visible_objects ==
            {
                'Microwave|-00.25|+01.71|-02.65',
                'Cabinet|+00.68|+02.02|-02.46',
                'Mug|-00.16|+01.10|+00.74_copy_0'
            })

    event = assert_successful_action(env,
            {
                'action' : 'TeleportFull',
                'x' : -1,
                'y': 1,
                'z': 1,
                'rotation': 270,
                'horizon' : 30
            })
    event = assert_successful_action(env,
            {'action' : 'OpenObject', 'objectId': 'Fridge|-01.89|+00.00|+01.07'})
    visible_objects = {obj['objectId'] for obj in event.metadata['objects'] if obj['visible']}
    assert(visible_objects ==
            {
                'Apple|-01.98|+01.28|+01.42_copy_2',
                'Egg|-01.92|+01.23|+01.09_copy_0',
                'Tomato|-00.29|+01.10|+00.34_copy_0',
                'Apple|-01.98|+01.28|+01.42_copy_1',
                'Tomato|-00.29|+01.10|+00.34_copy_1',
                'Lettuce|-01.94|+00.64|+01.42_copy_0',
                'Fridge|-01.89|+00.00|+01.07'
            })

def run_tests(env=None):
    create_env = (env is None)
    if create_env:
        env = game_util.create_env()
    test_random_initialize(env)
    # test multiple times in a row to make sure nothing isn't reset.
    test_random_initialize(env)
    test_random_initialize_randomize_open(env)
    test_random_initialize_with_remove_prob(env)
    test_random_initialize_with_repeats(env)
    print('All test_random_initialize tests passed!')

if __name__ == '__main__':
    run_tests()
