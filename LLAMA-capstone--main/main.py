
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from agent import agent_executor
app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="my-app/dist/assets"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_react():
    with open("my-app/dist/index.html") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/data")
async def get_data():
    return {"message": "Hello from the backend!"}


@app.post("/api/save-scan")
async def save_scanned_code(request: Request):
    body = await request.json()
    device_id = body.get("device_id")
    
    # ممكن تخزينه بملف، DB، أو جلسة session
    print(f"Scanned device ID: {device_id}")
    
    return {"status": "ok"}
    # Allow CORS for your frontend to access the API


class ChatRequest(BaseModel):
    message: str

# @app.post("/chat")
# async def chat_endpoint(req: ChatRequest):
#     print("Received message:", req.message)
#     try:
#         result = agent_executor.invoke({"input": req.message})
#         output = result.get("output", "No output from agent.")
        
#         return {"response": output}
#     except Exception as e:
#         print("Error:", e)
#         return {"error": str(e)}
import json
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    print("Received message:", req.message)
    try:
        result = agent_executor.invoke({"input": req.message})

        if isinstance(result, dict):
            output = result.get("output", result)
        elif isinstance(result, str):
            output = result
        else:
            output = str(result)

        # ✅ إذا كان output dict أو list أو أي نوع غير نصي، نحوله إلى JSON نصي:
        if not isinstance(output, str):
            output = json.dumps(output, indent=2)

        return {"response": output}
    except Exception as e:
        print("Error:", e)
        return {"error": str(e)}


# error Argument of type "str" cannot be assigned to parameter "input" of type "dict[str, Any]" in function "invoke"
#   "str" is not assignable to "dict[str, Any]"

