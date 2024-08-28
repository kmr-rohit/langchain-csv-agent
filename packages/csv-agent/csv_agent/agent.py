from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import create_retriever_tool
from langchain_experimental.tools import PythonAstREPLTool

load_dotenv()

MAIN_DIR = Path(__file__).parents[1]

pd.set_option("display.max_rows", 20)
pd.set_option("display.max_columns", 20)

embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.load_local(MAIN_DIR / "leadtime_data", embedding_model , allow_dangerous_deserialization=True)
retriever_tool = create_retriever_tool(
    vectorstore.as_retriever(), "person_name_search", "Search for a person by name"
)


TEMPLATE = """You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
It is important to understand the attributes of the dataframe before working with it. This is the result of running `df.head().to_markdown()`
<df>
{dhead}
</df>

"""  # noqa: E501


class PythonInputs(BaseModel):
    query: str = Field(description="code snippet to run")


df = pd.read_csv(MAIN_DIR / "data.csv")
template = TEMPLATE.format(dhead=df.head().to_markdown())

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ("human", "{input}"),
    ]
)

repl = PythonAstREPLTool(
    locals={"df": df},
    name="python_repl",
    description="Runs code and returns the output of the final line",
    args_schema=PythonInputs,
)

tools = [repl, retriever_tool]
agent = OpenAIFunctionsAgent(
    llm=ChatOpenAI(temperature=0, model="gpt-4"), prompt=prompt, tools=tools
)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, max_iterations=5, early_stopping_method="generate"
) | (lambda x: x["output"])



# Typing for playground inputs


class AgentInputs(BaseModel):
    input: str


agent_executor = agent_executor.with_types(input_type=AgentInputs)
