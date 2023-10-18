## Imports:
import os
from langchain.chat_models import ChatOpenAI
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.document_loaders import WebBaseLoader
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import LanceDB
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from app.task import Task, TaskDescription, TaskList
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import lancedb

## Set Env Variables
if None == os.environ.get('OPENAI_API_KEY'):
    raise ValueError("Env OPENAI_API_KEY not set")
else:
    OAI_TOKEN = os.environ.get('OPENAI_API_KEY')

if None == os.environ.get('OPENAI_CHAT_MODEL'):
    OPENAI_CHAT_MODEL = "gpt-3.5-turbo-0613"
else:
    OPENAI_CHAT_MODEL = os.environ.get('OPENAI_CHAT_MODEL')

if None == os.environ.get('OPENAI_CHAT_TEMPERATURE'):
    OPENAI_CHAT_TEMPERATURE = 0.3
else:
    OPENAI_CHAT_TEMPERATURE = os.environ.get('OPENAI_CHAT_TEMPERATURE')

if None == os.environ.get('OPENAI_MAX_CHAT_TOKENS'):
    OPENAI_MAX_CHAT_TOKENS = 200
else:
    OPENAI_MAX_CHAT_TOKENS = os.environ.get('OPENAI_MAX_CHAT_TOKENS')

if None == os.environ.get('OPENAI_RESEARCH_MODEL'):
    OPENAI_RESEARCH_MODEL = "gpt-3.5-turbo-16k-0613"
else:
    OPENAI_RESEARCH_MODEL = os.environ.get('OPENAI_RESEARCH_MODEL')

if None == os.environ.get('OPENAI_RESEARCH_TEMPERATURE'):
    OPENAI_RESEARCH_TEMPERATURE = 0.1
else:
    OPENAI_RESEARCH_TEMPERATURE = os.environ.get('OPENAI_RESEARCH_TEMPERATURE')

if None == os.environ.get('OPENAI_MAX_RESEARCH_TOKENS'):
    OPENAI_MAX_RESEARCH_TOKENS = 500
else:
    OPENAI_MAX_RESEARCH_TOKENS = os.environ.get('OPENAI_MAX_RESEARCH_TOKENS')

if None == os.environ.get('HELIOS_URL'):
    HELIOS_URL = "helios.latrobe.group"
else:
    HELIOS_URL = os.environ.get('HELIOS_URL')

## Set up OpenAI
VERBOSE = False
chat_model = ChatOpenAI(
    temperature=OPENAI_CHAT_TEMPERATURE,
    max_tokens=OPENAI_MAX_CHAT_TOKENS,
    model=OPENAI_CHAT_MODEL,
)
research_model = ChatOpenAI(
    temperature=OPENAI_RESEARCH_TEMPERATURE,
    max_tokens=OPENAI_MAX_RESEARCH_TOKENS,
    model=OPENAI_RESEARCH_MODEL,
)
embeddings = OpenAIEmbeddings()

## Set up FastAPI
helios_app = FastAPI()
origins = [
    f"http://{HELIOS_URL}",
    f"https://{HELIOS_URL}",
    "http://localhost",
    "http://localhost:8000",
]
helios_app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

## Set up Task Queue
TASKS = TaskList()

## Set up knowledge VectorStore and retriever chain
db_name = "./helios_kb.db"
table_name = "helios_kb"
db = lancedb.connect(db_name)
if table_name not in db.table_names():
    table = db.create_table(
        "helios_kb",
        data=[
            {
                "vector": embeddings.embed_query("You are Helios, an AI chatbot that can perform background research tasks."),
                "text": "You are Helios, an AI chatbot that can perform background research tasks with access to the internet.",
                "id": "1",
            }
        ],
        mode="create",
    )
else:
    table = db.open_table(table_name)
vectorstore = LanceDB(connection=table, embedding=embeddings)
kb_retriever = vectorstore.as_retriever()

def check_search_result(query: str, result: str) -> bool:
    '''Checks if the result of a search is a well informed answer to the query.'''
    prompt = PromptTemplate(
        input_variables=["search_query", "search_response"],
        template="Answer only 'Yes' or 'No' only - did the following response actually answer the question or include the right information to help the user with the query - yes or no:\n#####\nQuery: {search_query}\n#####\nResponse:{search_response}",
    )
    chain = LLMChain(llm=chat_model, prompt=prompt)
    check_response = chain.run(
        {
            "search_query": query,
            "search_response": result,
        }
    )
    if "YES" in check_response.upper():
        pass
    else:
        add_new_task(description=f"Use all of your tools to research this query: {query}")

def search_kb(query: str) -> str:
    compressor = LLMChainExtractor.from_llm(research_model)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=kb_retriever)
    compressed_docs = compression_retriever.get_relevant_documents(query)
    return compressed_docs

## Define Helper Functions
def add_new_task(description: str) -> Task:
    '''Adds a new task to the task queue.'''
    task_id = len(TASKS)
    task = Task(task_id=task_id, description=description)
    print(f"Adding new task: {description}")
    task.pending()
    TASKS.append(task)
    return task

def do_tasks():
    '''Runs the task queue.'''
    pending_tasks = [t for t in TASKS if t.get_status() == 'pending']
    for TASK in pending_tasks:
        run_task(task=TASK)

def run_task(task: Task):
    '''Runs a task.'''
    task.running()
    response = research_agent.run("Use your tools to create a knowledge graph on this topic: {TASK}".format(TASK=task.description))
    vectorstore.add_texts(texts=[response], metadatas=[{"id": task.task_id, "task": task.description}])
    task.done(result=response)
    return task

def load_web_page(url: str) -> str:
    '''Loads a web page and returns the text.'''
    loader = WebBaseLoader(url)
    loader.requests_kwargs = {'verify':False}
    data = loader.load()
    return f"Text from {url}:\n{data}"

## Set up LangChain Agents
python_agent = create_python_agent(
    llm=research_model,
    tool=PythonREPLTool(),
    agent_type=AgentType.OPENAI_FUNCTIONS,
    agent_executor_kwargs={"handle_parsing_errors": True},
)
chat_tools = [
    Tool(
        name="knowledgebase-search",
        func=search_kb,
        description="Searches the knowledge base for information. Input must be a list of key words or search terms.",
    ),
    Tool(
        name="add-research-task",
        func=add_new_task,
        description="If the knowledgebase doesn't have an answer, use this tool to start a background research task. Input must be a description of new research or a task to be done.",
    )
]

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
search = DuckDuckGoSearchRun()
research_tools = [
    Tool(
        name="knowledgebase-search",
        func=search_kb,
        description="Searches the knowledge base for information. Input must be a list of key words or search terms.",
    ),
    Tool(
        name="wikipedia-search",
        func=wikipedia.run,
        description="Searches Wikipedia information about people, places and historical facts. Input must be a list of key words or search terms.",
    ),
    Tool(
        name="web-search",
        func=search.run,
        description="Searches the web using DuckDuckGo for information from web pages. Input must be a list of key words or search terms.",
    ),
    Tool(
        name="run-python-code",
        func=python_agent.run,
        description="Sends a task to another agent that will write and run custom python code to achieve a task. Input must be a task or goal that a python programmer could achieve.",
    ),
    Tool(
        name="add-research-task",
        func=add_new_task,
        description="Only use this tool if the user asks you to add a new task. Input must be a description of new research or a task to be done.",
    )
]

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
}
memory = ConversationBufferWindowMemory(memory_key="memory", return_messages=True, k=4)

## Init Agents
chat_agent = initialize_agent(
    chat_tools, 
    chat_model, 
    agent=AgentType.OPENAI_MULTI_FUNCTIONS,
    agent_kwargs=agent_kwargs,
    memory=memory,
    handle_parsing_errors=True,
    verbose=VERBOSE)

research_agent = initialize_agent(
    research_tools, 
    research_model, 
    agent=AgentType.OPENAI_MULTI_FUNCTIONS, 
    handle_parsing_errors=True,
    verbose=VERBOSE)

## Add FastAPI Routes:
@helios_app.get("/")
async def root(background_tasks: BackgroundTasks):
    '''Returns the status of the bot.'''
    background_tasks.add_task(do_tasks)
    return {"status": "ok",
            "version": "helios v{VERSION}".format(VERSION=os.environ.get('VERSION'))}

@helios_app.get("/tasks")
async def get_tasks(background_tasks: BackgroundTasks):
    '''Returns a list of all tasks and their status.'''
    background_tasks.add_task(do_tasks)
    return TASKS.model_dump_json(indent = 2)

@helios_app.get("/tasks/{task_id}")
async def get_task(task_id: int, background_tasks: BackgroundTasks):
    '''Returns the status of a specific task.'''
    background_tasks.add_task(do_tasks)
    return TASKS[task_id].model_dump_json(indent = 2)

@helios_app.post("/tasks/")
async def create_task(task: TaskDescription, background_tasks: BackgroundTasks):
    '''Creates a new task.'''
    new_task = add_new_task(task.description)
    background_tasks.add_task(do_tasks)
    if isinstance(new_task, Task):
        background_tasks.add_task(do_tasks)
        return new_task.model_dump_json(indent = 2)
    else:
        raise HTTPException(status_code=500, detail="Task creation failed")

class SearchQuery(BaseModel):
    q: str

@helios_app.post("/search/")
async def search(q: SearchQuery, background_tasks: BackgroundTasks):
    '''Searches the knowledgebase for an answer to a question.'''
    response = search_kb(q.q)
    background_tasks.add_task(check_search_result, q.q, response)
    background_tasks.add_task(do_tasks)
    return response

class ChatMessage(BaseModel):
    message: str

@helios_app.post("/chat/")
async def chat(message: ChatMessage, background_tasks: BackgroundTasks):
    '''Chats with the bot.'''
    agent_prompt = f"Use your tools to give a detailed answer to this message - if you can't find the answer, say I don't know: {message.message}"
    response = chat_agent.run(agent_prompt)
    response = response.replace("\n", "<br />").replace("\r", "").replace("\"", "'")
    background_tasks.add_task(check_search_result, message.message, response)
    background_tasks.add_task(do_tasks)
    return response