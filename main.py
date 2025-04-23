import openai
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json

# Set API key for GPT-4o-mini
openai.api_key = "your-api-key-here"  # Replace with your actual API key

app = FastAPI()

# Define the input data model
class QueryRequest(BaseModel):
    userId: str
    query: str

# Define the response model
class ReportResponse(BaseModel):
    status: str
    apiEndpoint: str = None
    reportCode: str
    message: str
    startDate: str = None
    endDate: str = None

# FAISS index initialization
dim = 768  # Dimensions for the embeddings (can vary depending on model)
index = faiss.IndexFlatL2(dim)  # This uses the L2 (Euclidean) distance

# Example report data: report name, required parameters (payload), API endpoint
report_data = [
    ("Balance Sheet", {"startDate": "2025-01-01", "endDate": "2025-12-31"}, "seamlessServer/Report1"),
    ("Profit and Loss", {"startDate": "2025-01-01", "endDate": "2025-12-31"}, "seamlessServer/Report2"),
    ("Cash Flow", {"startDate": "2025-01-01", "endDate": "2025-12-31"}, "seamlessServer/Report3"),
    # Add more report data as needed
]

# Convert report names to embeddings and populate FAISS index
def generate_embeddings(text: str):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",  # Replace with the appropriate model for embeddings
        input=text
    )
    return np.array(response['data'][0]['embedding'], dtype=np.float32)

# Prepopulate FAISS index with embeddings for report names
for report_name, _, _ in report_data:
    embedding = generate_embeddings(report_name)
    index.add(np.array([embedding]))

# Function to call GPT-4o-mini model for processing the query
def process_query_with_llm(query: str):
    try:
        response = openai.Completion.create(
            engine="gpt-4o-mini",  # Specify your engine version (use gpt-4o-mini for this case)
            prompt=query,
            max_tokens=100,
            temperature=0.7,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error calling GPT model: " + str(e))

# Function to perform FAISS similarity search
def faiss_similarity_search(query: str, k=1):
    # Generate embedding for the incoming query
    query_embedding = generate_embeddings(query)
    
    # Search the FAISS index for the most similar report(s)
    distances, indices = index.search(np.array([query_embedding]), k)
    
    return distances, indices

@app.post("/generate_report", response_model=ReportResponse)
async def generate_report(request: QueryRequest):
    query = request.query
    
    # Step 1: Check if the query is a general question (skip report search)
    is_general_query = not any(report_name.lower() in query.lower() for report_name, _, _ in report_data)
    
    if is_general_query:
        # Handle general questions directly with LLM
        llm_response = process_query_with_llm(query)
        return ReportResponse(
            status="200",
            reportCode="GeneralQuery",
            message=llm_response,
        )
    
    # Step 2: Perform similarity search using FAISS to find closest reports
    distances, indices = faiss_similarity_search(query, k=1)
    
    if len(indices[0]) == 0:  # No match found
        llm_response = process_query_with_llm("Sorry, as per your request, we are not able to find the report type. Please specify the report name.")
        return ReportResponse(
            status="422",
            reportCode="NoMatch",
            message=llm_response,
        )
    
    # Step 3: Fetch the best matching report
    best_match = report_data[indices[0][0]]
    report_name, required_payload, api_endpoint = best_match

    # Step 4: Check if all required parameters (payload) are present
    missing_params = [param for param in required_payload if param not in request.dict()]
    if missing_params:
        llm_response = process_query_with_llm(f"Please provide the following missing parameters for the {report_name} report: {', '.join(missing_params)}")
        return ReportResponse(
            status="422",
            reportCode=report_name,
            message=llm_response,
        )

    # Step 5: Everything is validated, generate a message for successful response
    llm_response = process_query_with_llm(f"Report generated for {report_name} with the provided parameters.")
    return ReportResponse(
        status="200",
        apiEndpoint=api_endpoint,
        reportCode=report_name,
        message=llm_response,
        startDate=required_payload.get("startDate"),
        endDate=required_payload.get("endDate"),
    )

# To run the server:
# uvicorn app:app --reload
