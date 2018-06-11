def main():
    from utils import game_util
    env = game_util.create_env()
    print('\nStarting Tests\n')

    import test_image_overlays
    test_image_overlays.run_tests(env)

    import test_random_initialize
    test_random_initialize.run_tests(env)

    import speed_test
    speed_test.run_tests(env)
    print('\nAll tests passed!\n')

if __name__ == '__main__':
    main()
