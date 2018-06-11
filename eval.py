from constants import TASK
#if TASK == 'navigation':
    #print('evaluate navigation agent')
    #from supervised import test_navigation
    #test_navigation.main()
if TASK == 'rl':
    print('evaluate rl agent')
    from reinforcement_learning import a3c_test
    a3c_test.main()
elif TASK == 'end_to_end_baseline':
    print('evaluate end_to_end a3c baseline agent')
    from reinforcement_learning import a3c_test
    a3c_test.main()
