# probably-smart-eval
Smart-Eval-StartUp-Idea-Implementations

# Dependencies
* sentence-transformers
* textblob
* symspellpy
* tqdm
* neuralcoref
* flask
* dill
* fuzzywuzzy


# Target Services

♟ Space to Realtime edit the model answer

♟ Automatically identify the acronymns used and present a dictionary of editable abbreviations for acronyms in the model answer's context

♟ Automatically perform typo corrections

♟ Automatically resolve pronouns

♟ Identify the **I**mmutable-**P**hrases(s) [ proper nouns, common nouns, date objects, time objects etc], make it realtime editable and weight the sentence similarities scores with the jaccard similarity scores containing these phrases

♟ Present to the user the scores of student answer against model answer with a highlighting of the orienting and disorienting sentences and phrases in the student answer

♟ Near-Realtime updation of score with the updation of sentences in the student or model answer 

♟ Blue-Print of Output Features : https://docs.google.com/presentation/d/1B8184sIq5UEOrNoA136GnUzvSoB10K5FbbEDzDopzng/edit?usp=sharing


# Challenges 

♟Develop better embedded vector representation for the sentences using latest nlp models like LongFormers

♟Rejoin split sentences with the potential of being a single sentence to bring it to the form of model answer

♟Split single sentence with a potential of being split sentence to bring it to the form of model answer

♟Better the quality of Immutable Phrases(that is to identify ImPs among Nouns rather than declaring nouns as ImPs)

♟️Better Thresholds for semantic similarities

♟️Make an end-to-end architecture that considers factors on which marking should be done automatically

♟Identify the Strongest and Weakest Concepts of a student from his/her report

♟Recommend "similar to your answer" using inter-student-answer-simularity

♟Predict the scores of students in their future tests given the questions and model answers and some history of student answers(don't consider each student individual but rather consider each user as a vector of [{student_answet_type_label_over_t_tests}, {student_rank_type_label_over_t_tests}] and perform collabarative filtering or clustering then consider a single model for each of the cluster and use a NN model to map (document_concepts->Mean_test_score_of_cluster) where document_concepts would be identified using another NN model or TF-IDF
