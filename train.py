from constants import TASK

if TASK == 'navigation':
    print('train navigation agent')
    from supervised import train_navigation_agent
    train_navigation_agent.run()
elif TASK == 'language_model':
    print('train language model')
    from question_embedding import train_question_embedding
    train_question_embedding.run()
elif TASK == 'question_map_dump':
    print('generate ground truth maps')
    from reinforcement_learning import a3c_train
    a3c_train.run()
elif TASK == 'semantic_map_pretraining':
    print('pretraining answerer on ground truth maps')
    from supervised import semantic_map_pretrain
    semantic_map_pretrain.run()
elif TASK == 'rl':
    print('train rl agent')
    from reinforcement_learning import a3c_train
    a3c_train.run()
elif TASK == 'end_to_end_baseline':
    print('train end_to_end a3c baseline agent')
    from reinforcement_learning import a3c_train
    a3c_train.run()
