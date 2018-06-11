import constants

def vocabulary_size():
    return sum(1 for line in (open('vocabulary.txt'))) + 2 # all words in vocabulary + terminal + empty (for shorter sequences)

def tokenize_sentence(sentence, vocabulary):
    parsed_question = sentence.replace('.', '').replace('?', '').lower().split(' ')
    tokenized_input = [vocabulary[word] for word in parsed_question]
    tokenized_input = tokenized_input + [len(vocabulary)] + [len(vocabulary) + 1] * (constants.MAX_SENTENCE_LENGTH - len(tokenized_input) - 1)
    return tokenized_input

def get_sequences(data_file):
    vocabulary = {line.strip():ii for ii, line in enumerate(open('vocabulary.txt'))}
    questions = []
    question_types = []
    object_ids = []
    container_ids = []
    data = [line.strip().split(',') for line in open(data_file)][1:] # leave off header row
    for line in data:
        questions.append(tokenize_sentence(line[3], vocabulary))
        question_types.append(int(line[0]))
        object_ids.append(int(line[5]))
        container_ids.append(int(line[6]))
    return {'questions': questions, 'question_types': question_types, 'object_ids': object_ids, 'container_ids': container_ids}







