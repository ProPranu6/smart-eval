from flask import Flask, request, jsonify
from smarteval import *
app = Flask(__name__)
@app.route('/test')
def juss_testing():
	return "Juss Testing"
@app.route('/gradeit', methods=["POST"])
def grade():
	evaluator = SmartEval()
	input_json = request.get_json(force=True)
	report = evaluator.report_card(student_answers_path=input_json['student_csv_path'], teacher_answers_path=input_json['teacher_csv_path'])
	dictToReturn = report.to_json()
	return json.loads(dictToReturn)


