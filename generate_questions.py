from generate_questions import combine_hdf5
from generate_questions import generate_existence_questions
from generate_questions import generate_contains_questions
from generate_questions import generate_counting_questions
import question_to_text
for dataset_type in ['test', 'train_test', 'train']:
    generate_existence_questions.main(dataset_type)
    combine_hdf5.combine(dataset_type, 'existence')
    generate_contains_questions.main(dataset_type)
    combine_hdf5.combine(dataset_type, 'contains')
    generate_counting_questions.main(dataset_type)
    combine_hdf5.combine(dataset_type, 'counting')
question_to_text.main()
