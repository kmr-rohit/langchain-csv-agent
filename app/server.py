from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from csv_agent.agent import agent_executor as csv_agent_chain

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


add_routes(app, csv_agent_chain, path="/csv-agent")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
