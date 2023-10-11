## Imports:
import os
from langchain.chat_models import ChatOpenAI
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import LanceDB
from langchain.chains import RetrievalQA, LLMChain
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

## Set up OpenAI
VERBOSE = False
chat_model = ChatOpenAI(
    temperature=0.0,
    max_tokens=750,
    model="gpt-4-0613",
)
check_model = ChatOpenAI(
    temperature=0.0,
    max_tokens=750,
    model="gpt-3.5-turbo-0613",
)
embeddings = OpenAIEmbeddings()

## Set up FastAPI
helios_app = FastAPI()
origins = [
    "http://helios.latrobe.group",
    "https://helios.latrobe.group",
    "http://localhost",
    "http://localhost:8000",
]
helios_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
qa = RetrievalQA.from_chain_type(llm=check_model, chain_type="stuff", retriever=kb_retriever, verbose=VERBOSE)

def check_search_result(query: str, result: str) -> bool:
    '''Checks if the result of a search is a well informed answer to the query.'''
    prompt = PromptTemplate(
        input_variables=["search_query", "search_response"],
        template="Answer only 'Yes' or 'No' - did the following response actually answer the question or include the right information to help the user with the query:\n#####\nQuery: {search_query}\n#####\nResponse:{search_response}",
    )
    chain = LLMChain(llm=check_model, prompt=prompt)
    check_response = chain.run(
        {
            "search_query": query,
            "search_response": result,
        }
    )
    if "YES" in check_response.upper():
        pass
    else:
        prompt = PromptTemplate(
            input_variables=["search_query", "search_response"],
            template="Suggest one single and simple research task that could help improve future responses to this query:\n#####\nQuery: {search_query}\n#####\nResponse:{search_response}",
        )
        chain = LLMChain(llm=check_model, prompt=prompt)
        add_new_task(description=chain.run({"search_query": query, "search_response": result}))

def search_kb(query: str) -> str:
    results = qa.run(query)
    return results

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
    response = research_agent.run("Use all of your tools to perform detailed research to achieve following task or goal: {TASK}".format(TASK=task.description))
    vectorstore.add_texts(texts=[response], metadatas=[{"id": task.task_id, "task": task.description}])
    task.done(result=response)
    return task
    

## Set up LangChain Agents
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
        name="add-research-task",
        func=add_new_task,
        description="Only use this tool if the user asks you to add a new task. Input must be a description of new research or a task to be done.",
    )
]

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
}
memory = ConversationBufferWindowMemory(memory_key="memory", return_messages=True, k=3)

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
    chat_model, 
    agent=AgentType.OPENAI_MULTI_FUNCTIONS, 
    handle_parsing_errors=True,
    verbose=VERBOSE)

## Add FastAPI Routes:
@helios_app.get("/")
async def root(background_tasks: BackgroundTasks):
    '''Returns the status of the bot.'''
    background_tasks.add_task(do_tasks)
    return {"status": "ok",
            "version": "axebot v{VERSION}".format(VERSION=os.environ.get('VERSION'))}

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
    response = chat_agent.run(f"Use your tools to give a detailed answer to this message - if you can't find the answer, say I don't know: {message.message}")
    response = response.replace("\n", "<br />").replace("\r", "").replace("\"", "'")
    background_tasks.add_task(check_search_result, message.message, response)
    background_tasks.add_task(do_tasks)
    return response