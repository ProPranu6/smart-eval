
from sentence_transformers import SentenceTransformer
import sklearn
import numpy as np
import warnings
import re
from textblob import TextBlob
from symspellpy.symspellpy import SymSpell
from tqdm import tqdm
import pandas as pd
#import bar_chart_race as bcr 
import copy
from math import *
#import matplotlib.animation as ani
#import matplotlib.pyplot as plt
import json
import spacy
import pickle
import dill


warnings.filterwarnings("ignore")


from sentence_transformers import SentenceTransformer
import sklearn
import numpy as np
import warnings
import re
from textblob import TextBlob
from symspellpy.symspellpy import SymSpell
from tqdm import tqdm
import pandas as pd
#import bar_chart_race as bcr 
import copy
from math import *
#import matplotlib.animation as ani
#import matplotlib.pyplot as plt
import json
import spacy
import pickle
import dill


warnings.filterwarnings("ignore")


class SmartEval():
  
  def __init__(self):
    #dynamic imports
    

    pass

  def abbreviate_in_sen(self, sen, abbr_vocab, show_abbreviated=True):
    #sen = 'KL is a powerful way to automate things today'
    acronyms = re.findall('[A-Z][A-Z]+',sen)
    for acronym in acronyms:
      try:
        sen = re.sub(acronym, abbr_vocab[acronym],sen)
      except:
        new_acronym = acronym
        abbr_vocab[new_acronym] = new_acronym #"<undefined abbreviation>"
        print('New Abbreviation Vocab', new_acronym, 'detected')
        sen = re.sub(acronym, abbr_vocab[acronym],sen)
    
    if show_abbreviated:
          print(sen)
    return sen
      


  def csv_from_json_v1(self, teacher_answers_json, student_answers_json):
    """converts teacher_answers in array form and student_answer in records json form to csvs"""

    tea_answers = pd.DataFrame({str(k+1):[v] for k,v in enumerate(teacher_answers_json) })
    question_count = len(tea_answers)
    arrlike_parsed = json.loads(student_answers_json)
    stu_answers = dict()
    for i in range(1, question_count+1):
        stu_answers[str(i)] = []
    for record in arrlike_parsed:
      for rk, rv in record.items():
        stu_answers[rk] += [rv]
    stu_answers = pd.DataFrame(stu_answers)

    return tea_answers, stu_answers, question_count

  def csv_from_json_v2(self, teacher_student_json_path):
    """converts json of format {question_id:[ teacher_answer, [student_scores], [[student_answers]]]} to csvs"""

    jf = open(teacher_student_json_path)
    teacher_student_json = json.load(jf)
    tea_answers = pd.DataFrame({str(k):[v[0]] for k,v in teacher_student_json.items() })
    question_count = len(tea_answers.columns)

    stu_answers = dict()
    for i in range(1, question_count+1):
      stu_answers[str(i)] = []
    
    class_strength = 60
    for k,v in teacher_student_json.items():
      stu_answers[str(k)] = np.array(v[2]).reshape(-1)
      
      stu_answers[str(k)] = np.concatenate([stu_answers[str(k)], np.full((class_strength -stu_answers[str(k)].shape[0], ), np.nan)])
      
    stu_answers = pd.DataFrame(stu_answers)

    return tea_answers, stu_answers, question_count

  def ground_truth_scores(self, teacher_student_json_path):

    jf = open(teacher_student_json_path)
    teacher_student_json = json.load(jf)
    ground_truth_marks = pd.DataFrame({str(k):v[1] +[0 for i in range(60-len(v[1]))] for k,v in teacher_student_json.items() })
    return ground_truth_marks

  def mae(self, ground_truth_marks, predicted_marks):
    gtm, pm = np.array(ground_truth_marks).astype(np.float64), np.array(predicted_marks)
    return np.mean(np.absolute(gtm - pm), axis = 0)
  def evaluate(self, student_answer=None, teacher_answer=None, encoders = [], abbr_vocab=None, gen_context_spread=50, spell_check='None', student_id=-1, question_id=-1, encode_teacher_answer=True):

    #Acronym Replacement helps pick right match( because the trained data are highly likely to have seen abbreviations than acronyms and thus they know the context well when they have seen it)
    #Removing the stop words significantly improves the preference ratio but could affect negation statements try and use later
    #Passive and Active vocies are calculated well
    
    global qpembed1
    global qpembed2
    global qpembed3

    if student_answer == None or teacher_answer == None or abbr_vocab == None:

      print("Running Evaluation On Default Inputs")
      abbr_vocab = {'AI':'artificial intelligence', 'ML':'machine learning', 'DL' : 'deep learning', 'DS' : 'data science'}
      student_answer = 'The prominence of AI today has grown lot Artificial intelligence receives, understands and manipulates the world artificial intelligence is used to automate things'
      teacher_answer = 'AI is an important field present day Artificial intelligence means to be able to receive, understand, reason and change the environment AI is used to automate things'

      print("Student Answer : ")
      print(student_answer)

      print("Teacher Answer : ")
      print(teacher_answer)

    student_answer = re.sub("\\n+|\t+", " " , student_answer.strip())
    teacher_answer = re.sub("\\n+|\t+", " ", teacher_answer.strip())
    
    student_answer = re.sub("\[[0-9]+\]|:", "" , student_answer)
    teacher_answer = re.sub("\[[0-9]+\]|:", "", teacher_answer)
    
    teacher_answer_chunks = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', teacher_answer)
    student_answer_chunks = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', student_answer)

    teacher_answer_chunks_refined = []
    student_answer_chunks_refined = []

    for se in teacher_answer_chunks:
      if len(se.split(" ")) >=250 :
        for part in range(len(se.split())//250):
          teacher_answer_chunks_refined.append(se[part*250:(part+1)*250])
      else:
        teacher_answer_chunks_refined.append(se)
    
    for se in student_answer_chunks:
      if len(se.split(" ")) >=250 :
        for part in range(len(se.split())//250):
          student_answer_chunks_refined.append(se[part*250:(part+1)*250])
      else:
        student_answer_chunks_refined.append(se)
      
    

    teacher_answer_chunks_ref = [TextBlob(self.abbreviate_in_sen(se, abbr_vocab, show_abbreviated=False)).correct().string if spell_check == 'textblob' else self.abbreviate_in_sen(se, abbr_vocab, show_abbreviated=False) for se in teacher_answer_chunks_refined]
    student_answer_chunks_ref = [TextBlob(self.abbreviate_in_sen(se, abbr_vocab, show_abbreviated=False)).correct().string if spell_check == 'textblob' else self.abbreviate_in_sen(se, abbr_vocab, show_abbreviated=False) for se in student_answer_chunks_refined]
    
    

    #models = ['sentence-transformers/paraphrase-MiniLM-L6-v2','distilbert-base-nli-stsb-mean-tokens'] #'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
    correctnesses = []
    for model in tqdm(range(len(encoders)), desc='Evaluating Student' +str(student_id)+' against Question' + str(question_id)+' ...', ncols=100):
      encoder = encoders[model]
      
      if encode_teacher_answer:
        embed_teacher_answer = encoder.encode(teacher_answer_chunks_ref)
        if model == 0:
          qpembed1 = embed_teacher_answer
        else:
          qpembed2 = embed_teacher_answer
      else:
        if model == 0:
          embed_teacher_answer = qpembed1
        else:
          embed_teacher_answer = qpembed2

      embed_student_answer = encoder.encode(student_answer_chunks_ref)
      
      #embed_student_answer = np.expand_dims(np.average(embed_student_answer, axis=0), axis=0)
      #embed_teacher_answer = np.expand_dims(np.average(embed_teacher_answer, axis=0), axis=0)

      correctness, answer_sen_dist = self.new_similarities(embed_teacher_answer, embed_student_answer) #sklearn.metrics.pairwise.cosine_similarity(embed_teacher_answer, embed_student_answer, dense_output=True)[0][0] #rows are teacher_sens, cols are student_sens
      correctnesses.append(correctness)
      #print("\nStudent Answer Sentences Distribution : ", answer_sen_dist) 
    print("\nDone")
    correctnesses.append(np.mean(np.array(correctnesses), axis=0))
    return correctnesses, teacher_answer_chunks, student_answer_chunks

  
  def new_similarities(self, a, b):
    repeat_penalty = [0 for z in range(b.shape[0])]
    repeat_count = [0 for z in range(b.shape[0])]
    for i in range(b.shape[0]):
      repeat_penalty[i] = 1
      repeat_count[i] = 0
    repeat_penalty = np.array(repeat_penalty)
    repeat_count = np.array(repeat_count)

    scores_matrix = sklearn.metrics.pairwise.cosine_similarity(a, b)
    final_score = 0
    for index in range(a.shape[0]):
      stu_answer_id = np.argmax(scores_matrix[index, :])
      repeat_count[stu_answer_id] += 1
      final_score += (1/a.shape[0])*np.max(scores_matrix[index, :])
    
    return final_score, repeat_count

  def report_card(self, teacher_student_answers_json=(None, None), student_answers_path=None, teacher_answers_path=None, max_marks=5, relative_marking=False, json_load_version ="v1"):

    if teacher_student_answers_json[0] != None:
      
      if json_load_version == "v1":
        tea_answers, stu_answers, question_count = self.csv_from_json_v1(teacher_student_answers_json[0], teacher_student_answers_json[1])   #Works only for JSON in format {ColumnName : {Index : RowValue}}   
      elif json_load_version == "v2":
        tea_answers, stu_answers, question_count = self.csv_from_json_v2(teacher_student_answers_json[0])
                  
    else:  
      stu_answers = pd.read_csv(student_answers_path)
      tea_answers = pd.read_csv(teacher_answers_path)
      question_count = len(tea_answers.columns)
    
    
    class_strength = len(stu_answers)
    #question_count = 1
    student_marks = dict()
    for q in range(question_count):
      student_marks["Question " + str(q+1)] = []

    models = ['sentence-transformers/paraphrase-MiniLM-L6-v2','distilbert-base-nli-stsb-mean-tokens']
    encoders = []
    for model in models:
      encoder = SentenceTransformer(model)
      encoders.append(encoder)
    
    encode_ta = True
    for col in range(1, question_count+1):
        teacher_answer = tea_answers[str(col)][0]
        encode_ta = True
        for stu in range(class_strength):
          if stu_answers[str(col)][stu] != str(np.nan): #len(teacher_answer.split(" "))
            correctnesses, _, _ = self.evaluate(stu_answers[str(col)][stu], teacher_answer,encoders=encoders, abbr_vocab={}, gen_context_spread=125, spell_check='None', student_id = stu+1, question_id=col, encode_teacher_answer=encode_ta) #smaller context spread is more specific but less accurate at encoding
            to_append = correctnesses[-1]
          else:
            to_append = 0
          
          if to_append <0.5:
            to_append = max(0, to_append - (1-to_append)/4) #(correctnesses[-1] -(1 - correctnesses[-1])/2) if correctnesses[-1] <0.6 else correctnesses[-1]) #human_induced bias = sqrt(distance_from_high_score)
          elif to_append >0.95:
            to_append = 1.
          else:
            pass

          student_marks["Question " + str(col)].append(to_append)
          encode_ta = False
    
    report = student_marks
    report_norm = np.empty((question_count, class_strength)) #it's row = class_strength*questions = len
    for r in range(1, question_count+1):
        report_norm[r-1] = report["Question " + str(r)]

    report_norm_copy = copy.deepcopy(report_norm) #max mark
    #print(report_norm_copy)
    for r in range(question_count):
      true_max =  max(report_norm_copy[r]) if relative_marking else 1
      true_min = 0 
      for c in range(class_strength):
        report_norm[r][c] = np.round((report_norm_copy[r][c]-true_min)/(true_max-true_min)*max_marks, decimals=1) # mark = normalize(percentage), base_min_mark = 0, base_max_mark = true_max
    
    for q in range(question_count):
      student_marks["Question " + str(q+1)] = report_norm[q]
      
    report_csv = pd.DataFrame(student_marks)
    report_csv["Student Aggregate"] = np.sum(report_norm, axis=0)
    try:
      report_csv["Timestamp"] = stu_answers["Timestamp"]
      report_csv["Username"] = stu_answers["Username"]
      report_csv["Rollno"] = stu_answers["Rollno"]
    except:
      pass
    
    extra_info = dict()
    for k, v in enumerate(np.mean(report_norm, axis=1)):
      extra_info["Question " + str(k+1)] = v
    extra_info["Student Aggregate"] = np.mean(report_csv["Student Aggregate"])
    extra_info = pd.DataFrame(extra_info, index=["Class Mean"])
    report_csv = pd.concat([report_csv, extra_info], axis=0)
    return report_csv

if __name__ == "__main__":
  json_obj = [{"1":"i m bad boy"}, {"1":"i m bad boy"}, {"1":"i dont know the answer"}, {"1":"i m good boy"}]
  inputs = """{0}"""
  #import dill
  #dill.dump_session('test.pkl')
  #exit()
  #report = report_card(teacher_student_answers_json = ('/content/data.json',) , max_marks=5, relative_marking=False, json_load_version="v2")
  #print(report.to_string())
