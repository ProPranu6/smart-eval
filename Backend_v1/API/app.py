from fastapi import FastAPI, Request
from smart import *
app = FastAPI()

 

@app.get("/")
def test(request: Request):
    return {"testing passed"}
@app.post("/grade")
def grade(request: dict):


        evaluator = CannyEval()
        input_json = request 
        report = evaluator.report_card(data_json=input_json)
        
        
        
        return report
