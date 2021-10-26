# -*- coding: utf-8 -*-

from sentence_transformers import SentenceTransformer
import sklearn
import numpy as np
import warnings
import re
from textblob import TextBlob
from symspellpy.symspellpy import SymSpell
from tqdm import tqdm
import pandas as pd
import bar_chart_race as bcr 
import copy
from math import *
#import matplotlib.animation as ani
#import matplotlib.pyplot as plt
import json
import spacy


warnings.filterwarnings("ignore")


def abbreviate_in_sen(sen, abbr_vocab, show_abbreviated=True):
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
    
def tfidf_weights(corpus):

  vectorizer = TfidfVectorizer()
  X = vectorizer.fit_transform(corpus)
  print(X.shape)
  X = np.sum(X, axis=1)
  print(X.shape)
  return X/np.sum(X)

def csv_from_json_v1(teacher_answers_json, student_answers_json):
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

def csv_from_json_v2(teacher_student_json_path):
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

def ground_truth_scores(teacher_student_json_path):

  jf = open(teacher_student_json_path)
  teacher_student_json = json.load(jf)
  ground_truth_marks = pd.DataFrame({str(k):v[1] +[0 for i in range(60-len(v[1]))] for k,v in teacher_student_json.items() })
  return ground_truth_marks

def mae(ground_truth_marks, predicted_marks):
  gtm, pm = np.array(ground_truth_marks).astype(np.float64), np.array(predicted_marks)
  return np.mean(np.absolute(gtm - pm), axis = 0)
def evaluate(student_answer=None, teacher_answer=None, encoders = [], abbr_vocab=None, gen_context_spread=50, spell_check='None', student_id=-1, question_id=-1, encode_teacher_answer=True):

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
    if len(se.split(" ")) >=128 :
      for part in range(len(se.split())//128):
        teacher_answer_chunks_refined.append(se[part*128:(part+1)*128])
    else:
      teacher_answer_chunks_refined.append(se)
  
  for se in student_answer_chunks:
    if len(se.split(" ")) >=128 :
      for part in range(len(se.split())//128):
        student_answer_chunks_refined.append(se[part*128:(part+1)*128])
    else:
      student_answer_chunks_refined.append(se)
    
  

  teacher_answer_chunks_ref = [TextBlob(abbreviate_in_sen(se, abbr_vocab, show_abbreviated=False)).correct().string if spell_check == 'textblob' else abbreviate_in_sen(se, abbr_vocab, show_abbreviated=False) for se in teacher_answer_chunks_refined]
  student_answer_chunks_ref = [TextBlob(abbreviate_in_sen(se, abbr_vocab, show_abbreviated=False)).correct().string if spell_check == 'textblob' else abbreviate_in_sen(se, abbr_vocab, show_abbreviated=False) for se in student_answer_chunks_refined]
  

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
    
    embed_student_answer = np.expand_dims(np.average(embed_student_answer, axis=0), axis=0)
    embed_teacher_answer = np.expand_dims(np.average(embed_teacher_answer, axis=0), axis=0)

    correctness = sklearn.metrics.pairwise.cosine_similarity(embed_teacher_answer, embed_student_answer, dense_output=True)[0][0] #rows are teacher_sens, cols are student_sens
    correctnesses.append(correctness) 
  print("\nDone")
  correctnesses.append(np.mean(np.array(correctnesses), axis=0))
  return correctnesses, teacher_answer_chunks, student_answer_chunks

def report_card(teacher_student_answers_json=(None, None), student_answers_path=None, teacher_answers_path=None, max_marks=5, relative_marking=False, json_load_version ="v1"):

  if teacher_student_answers_json[0] != None:
    if json_load_version == "v1":
      tea_answers, stu_answers, question_count = csv_from_json_v1(teacher_student_answers_json[0], teacher_student_answers_json[1])   #Works only for JSON in format {ColumnName : {Index : RowValue}}   
    elif json_load_version == "v2":
      tea_answers, stu_answers, question_count = csv_from_json_v2(teacher_student_answers_json[0])
                
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
          correctnesses, _, _ = evaluate(stu_answers[str(col)][stu], teacher_answer,encoders=encoders, abbr_vocab={}, gen_context_spread=125, spell_check='None', student_id = stu+1, question_id=col, encode_teacher_answer=encode_ta) #smaller context spread is more specific but less accurate at encoding
          to_append = correctnesses[-1]
        else:
          to_append = 0
        
        if to_append <0.7:
          to_append = max(0, to_append - (1-to_append)/2) #(correctnesses[-1] -(1 - correctnesses[-1])/2) if correctnesses[-1] <0.6 else correctnesses[-1]) #human_induced bias = sqrt(distance_from_high_score)
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


  
 import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.layers import Dense, Input, Concatenate, Dot
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import datetime

def convert_to_BERT_vecs(sen):

  if sen != str(np.nan):
    sen = re.sub("\\n+|\t+", " " , sen.strip())
    sen = re.sub("\[[0-9]+\]|:", "" , sen)
    sen_chunks = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', sen)
    sen_chunks_ref = []
    for se in sen_chunks:
      if len(se.split(" ")) >=128 :
        for part in range(len(se.split())//128):
          sen_chunks_ref.append(se[part*128:(part+1)*128])
      else:
        sen_chunks_ref.append(se)
    
    embed_sen = enc.encode(sen_chunks_ref)
      
    embed_sen = np.average(embed_sen, axis=0)
  else:
    embed_sen = np.array([0.])
  return embed_sen
  
def prepare_dataset(data_json=None, data_csv_tea=None, data_csv_stu=None, train=True, from_json=True):
  global enc
  if from_json:
    tea_answers, stu_answers, question_count = csv_from_json_v2(data_json)
  else:
    tea_answers, stu_answers, question_count = pd.read_csv(data_csv_tea), pd.read_csv(data_csv_stu), pd.read_csv(data_csv_tea).shape[1] 

  model_name = 'distilbert-base-nli-stsb-mean-tokens'
  enc = SentenceTransformer(model_name)

  tea_answers = tea_answers.applymap(convert_to_BERT_vecs)
  stu_answers = stu_answers.applymap(convert_to_BERT_vecs)
  
  if train:
    gdm = ground_truth_scores(data_json)
    Y = []
   
  X1 = []
  X2 = []
  for ta in range(1, len(tea_answers.columns)+1):
    teacher_answer = tea_answers[str(ta)][0]
    for sa in range(stu_answers.shape[0]):
      student_answer = stu_answers[str(ta)][sa]
      if len(student_answer) != 1: 
        X1.append(teacher_answer)
        X2.append(student_answer)
        if train:
          Y.append(gdm[str(ta)][sa])
    
  X1, X2, Y = np.array(X1), np.array(X2), np.array(Y).astype(np.float32) if train else -1

  return X1, X2, Y

def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)    
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
   
    return np.array(norm_arr)

def model_and_train_v1(x1p, x2p, y, epo, lr):
  x1in = Input(shape=(768,))
  x2in = Input(shape=(768,))
  dot = Dot(axes=1, normalize=True)([x1in, x2in])
  
  d1 = Dense(100, activation='relu')
  d2 = Dense(50, activation='relu')
  #d3 = Dense(108, activation='linear')
  d4 = Dense(20, activation='relu')
  
  x1 = d1(x1in)
  x1 = d2(x1)
  #x1 = d3(x1)
  x1 = d4(x1)

  x2 = d1(x2in)
  x2 = d2(x2)
  #x2 = d3(x2)
  x2 = d4(x2)
  diff = x1-x2
  x = Concatenate(axis=1)([dot])
  
  xout = Dense(1, activation='relu')(x)
  

  model = Model(inputs=[x1in, x2in], outputs=xout)
  model.compile(Adam(lr), loss='mse', metrics=['mae'])
  model.fit(x = [x1p[:2000], x2p[:2000]], y = normalize(y[:2000], 0, 1), epochs=epo, validation_data=([x1p[2000:], x2p[2000:]], normalize(y[2000:], 0, 1)))

  return model

def model_and_train_v2(x1p, x2p, y, epo, lr):
  x1in = Input(shape=(768,))
  x2in = Input(shape=(768,))

  dot = Dot(axes=1, normalize=True)([x1in, x2in])
  diff = (x1in - x2in)
  x = Concatenate(axis=1)([diff, dot])

  #d1 = Dense(100, activation='relu')
  #d2 = Dense(50, activation='relu')
  #d3 = Dense(30, activation='linear')
  d4 = Dense(5, activation='relu')
  
  #x = d1(x)
  #x = d2(x)
  #x = d3(x)
  x = d4(x)

  
  
  xout = Dense(1, activation='relu')(x)
  

  model = Model(inputs=[x1in, x2in], outputs=xout)
  model.compile(Adam(lr), loss='mse', metrics=['mae'])
  model.fit(x = [x1p[:2000], x2p[:2000]], y = normalize(y[:2000], 0, 1), epochs=epo, validation_data=([x1p[2000:], x2p[2000:]], normalize(y[2000:], 0, 1)))

  return model
  

def model_and_train_v3(x1p, x2p, y, epo, lr, project_tensorboard=True):
  
  x1in = Input(shape=(768,))
  x2in = Input(shape=(768,))
  
  
  #d1 = Dense(200, activation='relu')
  #d2 = Dense(50, activation='relu')
  #d3 = Dense(108, activation='linear')
  #d4 = Dense(20, activation='relu')
  
  #x1 = d1(x1in)
  #x1 = d2(x1)
  #x1 = d3(x1)
  #x1 = d4(x1)

  #x2 = d1(x2in)
  #x2 = d2(x2)
  #x2 = d3(x2)
  #x2 = d4(x2)

  dot = Dot(axes=1, normalize=True)([x1in, x2in])
  diff = K.backend.sqrt(K.backend.sum(K.backend.square(x1in - x2in), axis=-1))
  x = Concatenate(axis=1)([dot, diff])
  
  xout = Dense(1, activation='relu')(x)
  log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

  model = Model(inputs=[x1in, x2in], outputs=xout)
  model.compile(Adam(lr), loss='mse', metrics=['mae'])
  model.fit(x = [x1p[:2000], x2p[:2000]], y = normalize(y[:2000], 0, 1), epochs=epo, validation_data=([x1p[2000:], x2p[2000:]], normalize(y[2000:], 0, 1)), callbacks=[tensorboard_callback])

  if project_tensorboard:
    # Load the TensorBoard notebook extension
    %load_ext tensorboard
    %tensorboard --logdir logs/fit
  return model

if __name__ == "__main__":
  
  X1, X2, Y = prepare_dataset('/content/data.json')
  #X1.shape, X2.shape, Y.shape
  model = model_and_train_v3(X1, X2, Y,400, 7e-5)
  print(np.squeeze(model.predict([X1, X2]))[:10]*5, Y[:10])
  
