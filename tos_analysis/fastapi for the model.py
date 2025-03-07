from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

app = FastAPI(
    title="TOS Analysis & Chatbot API",
    description=(
        "An API for analyzing TOS documents with multiple pretrained models:\n"
        "1. Fraud detection: classifies the TOS as fraudulent or not.\n"
        "2. Data access evaluation: determines if the TOS's data access clause is accepted or rejected.\n"
        "3. Summarization: generates a summary using a BERT-based summarizer.\n"
        "4. Chatbot: answers questions based on the TOS content."
    )
)

# Request model for endpoints that take only the TOS document.
class TOSRequest(BaseModel):
    tos_text: str

# Request model for the chatbot endpoint.
class ChatRequest(BaseModel):
    question: str
    tos_text: str

# ----------------------------
# Initialize the model pipelines
# ----------------------------

# 1. TOS Fraud Detection Model
# Replace 'your-tos-fraud-model' with your actual model identifier or local path.
fraud_model = pipeline("text-classification", model="./fastapi_model")

# 2. TOS Data Access Evaluation Model
# Replace 'your-tos-access-model' with your actual model identifier or local path.
access_model = pipeline("text-classification", model="./results")

# 3. BERT-based Summarizer for TOS
# Replace 'your-bert-summarizer' with your actual BERT summarizer model identifier or local path.
bert_summarizer = pipeline("summarization", model="./bert2bert-summarizer")

# 4. Chatbot / Question-Answering Model for TOS
# Here we're using a commonly available QA model.
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# ----------------------------
# Endpoints
# ----------------------------

@app.post("/analyze-tos")
async def analyze_tos(item: TOSRequest):
    """
    Analyze the given TOS document for fraudulence.
    Returns a label (e.g., "FRAUD" or "NOT_FRAUD") and a confidence score.
    """
    if not item.tos_text:
        raise HTTPException(status_code=400, detail="TOS text is required.")
    try:
        result = fraud_model(item.tos_text)
        label = result[0]["label"]
        score = result[0]["score"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fraud model error: {str(e)}")
    return {"fraudulent": label, "score": score}

@app.post("/check-access")
async def check_access(item: TOSRequest):
    """
    Evaluate whether the TOS's data access clause is acceptable.
    Returns a label (e.g., "ACCEPTED" or "REJECTED") and a confidence score.
    """
    if not item.tos_text:
        raise HTTPException(status_code=400, detail="TOS text is required.")
    try:
        result = access_model(item.tos_text)
        label = result[0]["label"]
        score = result[0]["score"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Access model error: {str(e)}")
    return {"access": label, "score": score}

@app.post("/summarize-tos")
async def summarize_tos(item: TOSRequest):
    """
    Summarize the given TOS document using a BERT-based summarizer.
    Returns the summarized text.
    """
    if not item.tos_text:
        raise HTTPException(status_code=400, detail="TOS text is required.")
    try:
        summary = bert_summarizer(item.tos_text, max_length=150, min_length=40, do_sample=False)
        summarized_text = summary[0]["summary_text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization model error: {str(e)}")
    return {"summary": summarized_text}

@app.post("/chat-tos")
async def chat_tos(item: ChatRequest):
    """
    Chatbot endpoint to answer questions based on the TOS document.
    Accepts a question and the TOS text, and returns an answer along with a confidence score.
    """
    if not item.question or not item.tos_text:
        raise HTTPException(status_code=400, detail="Both question and TOS text are required.")
    try:
        answer = qa_model(question=item.question, context=item.tos_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QA model error: {str(e)}")
    return {"answer": answer["answer"], "score": answer["score"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
