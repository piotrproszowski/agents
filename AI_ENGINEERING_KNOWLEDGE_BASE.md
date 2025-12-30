# 🚀 AI Engineering Knowledge Base dla Backend Developera

## Kompletny przewodnik po integracji AI i tworzeniu Agentów na poziomie produkcyjnym

> **Dla kogo:** Backend Developer Node.js + TypeScript + PostgreSQL, który chce integrować AI do aplikacji i tworzyć rozwiązania w oparciu o AI Engineering.

---

## 📚 Spis treści

1. [Fundamenty AI Engineering](#1-fundamenty-ai-engineering)
2. [Praca z API modeli LLM](#2-praca-z-api-modeli-llm)
3. [Wzorce projektowe Agentic AI](#3-wzorce-projektowe-agentic-ai)
4. [Tool Use - Integracja funkcji z LLM](#4-tool-use---integracja-funkcji-z-llm)
5. [OpenAI Agents SDK](#5-openai-agents-sdk)
6. [CrewAI - Multi-Agent Systems](#6-crewai---multi-agent-systems)
7. [LangGraph - State Management](#7-langgraph---state-management)
8. [AutoGen - Agent Communication](#8-autogen---agent-communication)
9. [Model Context Protocol (MCP)](#9-model-context-protocol-mcp)
10. [Structured Outputs i Pydantic](#10-structured-outputs-i-pydantic)
11. [Guardrails i Bezpieczeństwo](#11-guardrails-i-bezpieczeństwo)
12. [Memory i Persistence](#12-memory-i-persistence)
13. [Async Python w AI Engineering](#13-async-python-w-ai-engineering)
14. [Observability i Tracing](#14-observability-i-tracing)
15. [Deployment i Produkcja](#15-deployment-i-produkcja)
16. [Najlepsze praktyki produkcyjne](#16-najlepsze-praktyki-produkcyjne)

---

## 1. Fundamenty AI Engineering

### 1.1 Podstawowe wywołanie API OpenAI

```python
# Z lab: 1_foundations/1_lab1.ipynb
from openai import OpenAI

openai = OpenAI()

messages = [{"role": "user", "content": "What is 2+2?"}]

response = openai.chat.completions.create(
    model="gpt-4.1-nano",
    messages=messages
)

print(response.choices[0].message.content)
```

### 1.2 Zarządzanie kluczami API i środowiskami

```python
# Z lab: 1_foundations/1_lab1.ipynb
from dotenv import load_dotenv
import os

load_dotenv(override=True)  # Zawsze z override=True!

openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')

# Walidacja klucza
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
```

**Kluczowe koncepcje:**
- Zmienne środowiskowe (.env) dla bezpieczeństwa kluczy API
- Weryfikacja istnienia kluczy przed użyciem
- Różne providery LLM (OpenAI, Anthropic, Google, DeepSeek)

---

## 2. Praca z API modeli LLM

### 2.1 Różni providerzy LLM z wspólnym interfejsem

```python
# Z lab: 1_foundations/2_lab2.ipynb
from openai import OpenAI
from anthropic import Anthropic

# OpenAI
openai = OpenAI()
response = openai.chat.completions.create(model="gpt-5-mini", messages=messages)

# Anthropic Claude - nieco inny API
claude = Anthropic()
response = claude.messages.create(
    model="claude-sonnet-4-5", 
    messages=messages, 
    max_tokens=1000  # Wymagane!
)
answer = response.content[0].text

# Gemini przez OpenAI-compatible endpoint
gemini = OpenAI(
    api_key=google_api_key, 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
response = gemini.chat.completions.create(model="gemini-2.5-flash", messages=messages)

# DeepSeek
deepseek = OpenAI(
    api_key=deepseek_api_key, 
    base_url="https://api.deepseek.com/v1"
)
response = deepseek.chat.completions.create(model="deepseek-chat", messages=messages)

# Ollama (lokalne modele - za darmo!)
ollama = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
response = ollama.chat.completions.create(model="llama3.2", messages=messages)
```

### 2.2 Pattern: Judge/Evaluator

```python
# Z lab: 1_foundations/2_lab2.ipynb
# Wzorzec gdzie jeden LLM ocenia wyniki innych

judge = f"""You are judging a competition between {len(competitors)} competitors.
Each model has been given this question:
{question}

Your job is to evaluate each response for clarity and strength of argument, 
and rank them in order of best to worst.
Respond with JSON, and only JSON, with the following format:
{{"results": ["best competitor number", "second best competitor number", ...]}}

Here are the responses from each competitor:
{together}

Now respond with the JSON with the ranked order of the competitors, nothing else.
"""

judge_messages = [{"role": "user", "content": judge}]
response = openai.chat.completions.create(model="gpt-5-mini", messages=judge_messages)
results = json.loads(response.choices[0].message.content)
```

---

## 3. Wzorce projektowe Agentic AI

### 3.1 Reflection Pattern (Evaluator + Retry)

```python
# Z lab: 1_foundations/3_lab3.ipynb
from pydantic import BaseModel

class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str

evaluator_system_prompt = f"""You are an evaluator that decides whether a response 
to a question is acceptable. Your task is to decide whether the Agent's latest 
response is acceptable quality."""

def evaluate(reply, message, history) -> Evaluation:
    messages = [
        {"role": "system", "content": evaluator_system_prompt},
        {"role": "user", "content": evaluator_user_prompt(reply, message, history)}
    ]
    response = gemini.beta.chat.completions.parse(
        model="gemini-2.0-flash", 
        messages=messages, 
        response_format=Evaluation
    )
    return response.choices[0].message.parsed

def rerun(reply, message, history, feedback):
    """Ponów próbę z feedbackiem dlaczego poprzednia odpowiedź była odrzucona"""
    updated_system_prompt = system_prompt + f"""
    Previously you tried to reply, but the quality control rejected your reply.
    Your attempted answer: {reply}
    Reason for rejection: {feedback}
    """
    # ... ponów wywołanie

def chat(message, history):
    # Pierwsza próba
    reply = get_llm_response(message, history)
    
    # Ewaluacja
    evaluation = evaluate(reply, message, history)
    
    if evaluation.is_acceptable:
        print("Passed evaluation - returning reply")
        return reply
    else:
        print(f"Failed evaluation - retrying. Feedback: {evaluation.feedback}")
        return rerun(reply, message, history, evaluation.feedback)
```

### 3.2 Parallelization Pattern

```python
# Z lab: 2_openai/2_lab2.ipynb
import asyncio
from agents import Agent, Runner

message = "Write a cold sales email"

# Równoległe wywołanie wielu agentów
with trace("Parallel cold emails"):
    results = await asyncio.gather(
        Runner.run(sales_agent1, message),
        Runner.run(sales_agent2, message),
        Runner.run(sales_agent3, message),
    )

outputs = [result.final_output for result in results]
```

### 3.3 Orchestrator Pattern (Sales Manager)

```python
# Z lab: 2_openai/2_lab2.ipynb
sales_manager_instructions = """
You are a Sales Manager at ComplAI. Your goal is to find the single best 
cold sales email using the sales_agent tools.
 
Follow these steps carefully:
1. Generate Drafts: Use all three sales_agent tools to generate three different 
   email drafts. Do not proceed until all three drafts are ready.
 
2. Evaluate and Select: Review the drafts and choose the single best email 
   using your judgment of which one is most effective.
 
3. Handoff for Sending: Pass ONLY the winning email draft to the 'Email Manager' 
   agent. The Email Manager will take care of formatting and sending.
 
Crucial Rules:
- You must use the sales agent tools to generate the drafts — do not write them yourself.
- You must hand off exactly ONE email to the Email Manager — never more than one.
"""

sales_manager = Agent(
    name="Sales Manager",
    instructions=sales_manager_instructions,
    tools=tools,
    handoffs=handoffs,
    model="gpt-4o-mini"
)
```

---

## 4. Tool Use - Integracja funkcji z LLM

### 4.1 Definiowanie narzędzi (tradycyjne)

```python
# Z lab: 1_foundations/4_lab4.ipynb
# JSON Schema dla funkcji
record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            },
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json}
]

# Wywołanie LLM z narzędziami
response = openai.chat.completions.create(
    model="gpt-4o-mini", 
    messages=messages, 
    tools=tools
)
```

### 4.2 Obsługa wywołań narzędzi

```python
# Z lab: 1_foundations/4_lab4.ipynb
def handle_tool_calls(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"Tool called: {tool_name}", flush=True)
        
        # Dynamiczne wywołanie funkcji przez globals()
        tool = globals().get(tool_name)
        result = tool(**arguments) if tool else {}
        
        results.append({
            "role": "tool",
            "content": json.dumps(result),
            "tool_call_id": tool_call.id
        })
    return results

def chat(message, history):
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    done = False
    while not done:
        response = openai.chat.completions.create(
            model="gpt-4o-mini", 
            messages=messages, 
            tools=tools
        )
        
        if response.choices[0].finish_reason == "tool_calls":
            message = response.choices[0].message
            tool_calls = message.tool_calls
            results = handle_tool_calls(tool_calls)
            messages.append(message)
            messages.extend(results)
        else:
            done = True
    return response.choices[0].message.content
```

### 4.3 Narzędzia z dekoratorem @function_tool (OpenAI Agents SDK)

```python
# Z lab: 2_openai/2_lab2.ipynb
from agents import function_tool

@function_tool
def send_email(body: str):
    """ Send out an email with the given body to all sales prospects """
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
    from_email = Email("ed@edwarddonner.com")
    to_email = To("ed.donner@gmail.com")
    content = Content("text/plain", body)
    mail = Mail(from_email, to_email, "Sales email", content).get()
    sg.client.mail.send.post(request_body=mail)
    return {"status": "success"}

# Automatycznie tworzy JSON schema z docstringu!
```

---

## 5. OpenAI Agents SDK

### 5.1 Podstawowy Agent

```python
# Z lab: 2_openai/1_lab1.ipynb
from agents import Agent, Runner, trace

# Tworzenie agenta
agent = Agent(
    name="Jokester", 
    instructions="You are a joke teller", 
    model="gpt-4o-mini"
)

# Uruchomienie z tracem
with trace("Telling a joke"):
    result = await Runner.run(agent, "Tell a joke about Autonomous AI Agents")
    print(result.final_output)
```

### 5.2 Agent jako narzędzie (Agent-as-Tool)

```python
# Z lab: 2_openai/2_lab2.ipynb
# Konwersja agenta do narzędzia
tool1 = sales_agent1.as_tool(
    tool_name="sales_agent1", 
    tool_description="Write a cold sales email"
)

# Zbieranie wszystkich narzędzi
tools = [tool1, tool2, tool3, send_email]

# Agent-menedżer korzystający z innych agentów jako narzędzi
sales_manager = Agent(
    name="Sales Manager", 
    instructions=instructions, 
    tools=tools, 
    model="gpt-4o-mini"
)
```

### 5.3 Handoffs (przekazanie kontroli między agentami)

```python
# Z lab: 2_openai/2_lab2.ipynb
# Handoff = agent któremu można przekazać kontrolę
emailer_agent = Agent(
    name="Email Manager",
    instructions="You format and send emails",
    tools=[subject_tool, html_tool, send_html_email],
    model="gpt-4o-mini",
    handoff_description="Convert an email to HTML and send it"
)

# Agent z możliwością handoffu
sales_manager = Agent(
    name="Sales Manager",
    instructions=sales_manager_instructions,
    tools=[tool1, tool2, tool3],
    handoffs=[emailer_agent],  # <-- Handoffs!
    model="gpt-4o-mini"
)
```

### 5.4 Różne modele w jednym systemie

```python
# Z lab: 2_openai/3_lab3.ipynb
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

deepseek_client = AsyncOpenAI(base_url=DEEPSEEK_BASE_URL, api_key=deepseek_api_key)
gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
groq_client = AsyncOpenAI(base_url=GROQ_BASE_URL, api_key=groq_api_key)

deepseek_model = OpenAIChatCompletionsModel(model="deepseek-chat", openai_client=deepseek_client)
gemini_model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=gemini_client)
llama3_3_model = OpenAIChatCompletionsModel(model="llama-3.3-70b-versatile", openai_client=groq_client)

# Agenci korzystający z różnych modeli
sales_agent1 = Agent(name="DeepSeek Sales Agent", instructions=instructions1, model=deepseek_model)
sales_agent2 = Agent(name="Gemini Sales Agent", instructions=instructions2, model=gemini_model)
sales_agent3 = Agent(name="Llama3.3 Sales Agent", instructions=instructions3, model=llama3_3_model)
```

### 5.5 Hosted Tools (WebSearch, FileSearch)

```python
# Z lab: 2_openai/4_lab4.ipynb
from agents import Agent, WebSearchTool
from agents.model_settings import ModelSettings

INSTRUCTIONS = """You are a research assistant. Given a search term, 
you search the web for that term and produce a concise summary of the results."""

search_agent = Agent(
    name="Search agent",
    instructions=INSTRUCTIONS,
    tools=[WebSearchTool(search_context_size="low")],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="required"),  # Wymuszenie użycia narzędzia
)
```

---

## 6. CrewAI - Multi-Agent Systems

### 6.1 Struktura projektu CrewAI

```
debate/
├── src/debate/
│   ├── crew.py         # Definicja Crew
│   ├── main.py         # Entry point
│   └── config/
│       ├── agents.yaml # Konfiguracja agentów
│       └── tasks.yaml  # Konfiguracja tasków
├── pyproject.toml
└── knowledge/          # RAG knowledge base
```

### 6.2 Definicja Crew

```python
# Z lab: 3_crew/debate/src/debate/crew.py
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class Debate():
    """Debate crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def debater(self) -> Agent:
        return Agent(
            config=self.agents_config['debater'],
            verbose=True
        )

    @agent
    def judge(self) -> Agent:
        return Agent(
            config=self.agents_config['judge'],
            verbose=True
        )

    @task
    def propose(self) -> Task:
        return Task(config=self.tasks_config['propose'])

    @task
    def oppose(self) -> Task:
        return Task(config=self.tasks_config['oppose'])

    @task
    def decide(self) -> Task:
        return Task(config=self.tasks_config['decide'])

    @crew
    def crew(self) -> Crew:
        """Creates the Debate crew"""
        return Crew(
            agents=self.agents,  # Auto z @agent
            tasks=self.tasks,    # Auto z @task
            process=Process.sequential,
            verbose=True,
        )
```

### 6.3 Uruchamianie Crew

```bash
# Z terminala w folderze projektu
crewai run
```

---

## 7. LangGraph - State Management

### 7.1 Podstawowa struktura grafu

```python
# Z lab: 4_langgraph/1_lab1.ipynb
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel

# Step 1: Define the State object
class State(BaseModel):
    messages: Annotated[list, add_messages]  # Reducer!

# Step 2: Start the Graph Builder with this State class
graph_builder = StateGraph(State)

# Step 3: Create a Node
def chatbot_node(old_state: State) -> State:
    response = llm.invoke(old_state.messages)
    return State(messages=[response])

graph_builder.add_node("chatbot", chatbot_node)

# Step 4: Create Edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Step 5: Compile the Graph
graph = graph_builder.compile()

# Wizualizacja
display(Image(graph.get_graph().draw_mermaid_png()))
```

### 7.2 Conditional Edges i Tools

```python
# Z lab: 4_langgraph/2_lab2.ipynb
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.agents import Tool

# Definiowanie narzędzi LangChain
tool_search = Tool(
    name="search",
    func=serper.run,
    description="Useful for when you need more information from an online search"
)

tools = [tool_search, tool_push]

# LLM z narzędziami
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))

# Conditional edge - decyzja czy wywołać narzędzie
graph_builder.add_conditional_edges("chatbot", tools_condition, "tools")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
```

### 7.3 Memory/Checkpointing

```python
# Z lab: 4_langgraph/2_lab2.ipynb
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Memory w RAM
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Memory w SQLite (persystentna)
db_path = "memory.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
sql_memory = SqliteSaver(conn)
graph = graph_builder.compile(checkpointer=sql_memory)

# Konfiguracja thread_id dla memory
config = {"configurable": {"thread_id": "1"}}
result = graph.invoke({"messages": [{"role": "user", "content": user_input}]}, config=config)

# Dostęp do historii
graph.get_state(config)
list(graph.get_state_history(config))
```

### 7.4 Zaawansowany Agent (Sidekick)

```python
# Z lab: 4_langgraph/sidekick.py
class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool

class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(description="True if more input is needed from the user")

class Sidekick:
    async def build_graph(self):
        graph_builder = StateGraph(State)

        # Dodanie nodes
        graph_builder.add_node("worker", self.worker)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_node("evaluator", self.evaluator)

        # Conditional routing
        graph_builder.add_conditional_edges(
            "worker", 
            self.worker_router, 
            {"tools": "tools", "evaluator": "evaluator"}
        )
        graph_builder.add_edge("tools", "worker")
        graph_builder.add_conditional_edges(
            "evaluator", 
            self.route_based_on_evaluation, 
            {"worker": "worker", "END": END}
        )
        graph_builder.add_edge(START, "worker")

        self.graph = graph_builder.compile(checkpointer=self.memory)
```

---

## 8. AutoGen - Agent Communication

### 8.1 Podstawowe koncepcje

```python
# Z lab: 5_autogen/1_lab1_autogen_agentchat.ipynb
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

# Model Client
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

# Ollama model
ollama_client = OllamaChatCompletionClient(model="llama3.2")

# Agent
agent = AssistantAgent(
    name="airline_agent",
    model_client=model_client,
    system_message="You are a helpful assistant for an airline.",
    model_client_stream=True
)

# Message
message = TextMessage(content="I'd like to go to London", source="user")

# Run
response = await agent.on_messages([message], cancellation_token=CancellationToken())
print(response.chat_message.content)
```

### 8.2 Agent z narzędziami

```python
# Z lab: 5_autogen/1_lab1_autogen_agentchat.ipynb
def get_city_price(city_name: str) -> float | None:
    """ Get the roundtrip ticket price to travel to the city """
    conn = sqlite3.connect("tickets.db")
    c = conn.cursor()
    c.execute("SELECT round_trip_price FROM cities WHERE city_name = ?", (city_name.lower(),))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

smart_agent = AssistantAgent(
    name="smart_airline_agent",
    model_client=model_client,
    system_message="You are a helpful assistant for an airline. Include the price of a roundtrip ticket.",
    model_client_stream=True,
    tools=[get_city_price],
    reflect_on_tool_use=True  # Agent przemyśli wynik narzędzia
)
```

---

## 9. Model Context Protocol (MCP)

### 9.1 Podstawy MCP

```python
# Z lab: 6_mcp/1_lab1.ipynb
from agents import Agent, Runner, trace
from agents.mcp import MCPServerStdio

# Fetch MCP Server
fetch_params = {"command": "uvx", "args": ["mcp-server-fetch"]}

async with MCPServerStdio(params=fetch_params, client_session_timeout_seconds=60) as server:
    fetch_tools = await server.list_tools()

# Playwright MCP Server
playwright_params = {"command": "npx", "args": ["@playwright/mcp@latest"]}

# Filesystem MCP Server  
sandbox_path = os.path.abspath(os.path.join(os.getcwd(), "sandbox"))
files_params = {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", sandbox_path]}
```

### 9.2 Agent z wieloma MCP Servers

```python
# Z lab: 6_mcp/1_lab1.ipynb
instructions = """
You browse the internet to accomplish your instructions.
You are highly capable at browsing the internet independently to accomplish your task, 
including accepting all cookies and clicking 'not now' as appropriate.
"""

async with MCPServerStdio(params=files_params, client_session_timeout_seconds=60) as mcp_server_files:
    async with MCPServerStdio(params=playwright_params, client_session_timeout_seconds=60) as mcp_server_browser:
        agent = Agent(
            name="investigator", 
            instructions=instructions, 
            model="gpt-4.1-mini",
            mcp_servers=[mcp_server_files, mcp_server_browser]
        )
        with trace("investigate"):
            result = await Runner.run(
                agent, 
                "Find a great recipe for Banoffee Pie, then summarize it in markdown to banoffee.md"
            )
            print(result.final_output)
```

### 9.3 Zaawansowany Trading System z MCP

```python
# Z lab: 6_mcp/traders.py
from contextlib import AsyncExitStack
from agents.mcp import MCPServerStdio

class Trader:
    async def run_with_mcp_servers(self):
        async with AsyncExitStack() as stack:
            # Otwieranie wielu MCP servers
            trader_mcp_servers = [
                await stack.enter_async_context(
                    MCPServerStdio(params, client_session_timeout_seconds=120)
                )
                for params in trader_mcp_server_params
            ]
            async with AsyncExitStack() as stack:
                researcher_mcp_servers = [
                    await stack.enter_async_context(
                        MCPServerStdio(params, client_session_timeout_seconds=120)
                    )
                    for params in researcher_mcp_server_params(self.name)
                ]
                await self.run_agent(trader_mcp_servers, researcher_mcp_servers)
```

---

## 10. Structured Outputs i Pydantic

### 10.1 Definiowanie schematów

```python
# Z lab: 2_openai/4_lab4.ipynb
from pydantic import BaseModel, Field

class WebSearchItem(BaseModel):
    reason: str = Field(description="Your reasoning for why this search is important to the query.")
    query: str = Field(description="The search term to use for the web search.")

class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(
        description="A list of web searches to perform to best answer the query."
    )

# Agent z structured output
planner_agent = Agent(
    name="PlannerAgent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=WebSearchPlan,  # <-- Structured output!
)
```

### 10.2 Raportowanie z strukturą

```python
# Z lab: 2_openai/4_lab4.ipynb
class ReportData(BaseModel):
    short_summary: str = Field(description="A short 2-3 sentence summary of the findings.")
    markdown_report: str = Field(description="The final report")
    follow_up_questions: list[str] = Field(description="Suggested topics to research further")

writer_agent = Agent(
    name="WriterAgent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=ReportData,
)

# Wynik automatycznie sparsowany do ReportData
result = await Runner.run(writer_agent, input)
report: ReportData = result.final_output
print(report.short_summary)
print(report.markdown_report)
```

---

## 11. Guardrails i Bezpieczeństwo

### 11.1 Input Guardrails

```python
# Z lab: 2_openai/3_lab3.ipynb
from agents import input_guardrail, GuardrailFunctionOutput

class NameCheckOutput(BaseModel):
    is_name_in_message: bool
    name: str

guardrail_agent = Agent( 
    name="Name check",
    instructions="Check if the user is including someone's personal name in what they want you to do.",
    output_type=NameCheckOutput,
    model="gpt-4o-mini"
)

@input_guardrail
async def guardrail_against_name(ctx, agent, message):
    result = await Runner.run(guardrail_agent, message, context=ctx.context)
    is_name_in_message = result.final_output.is_name_in_message
    return GuardrailFunctionOutput(
        output_info={"found_name": result.final_output},
        tripwire_triggered=is_name_in_message  # True = blokuj!
    )

# Agent z guardrailem
careful_sales_manager = Agent(
    name="Sales Manager",
    instructions=sales_manager_instructions,
    tools=tools,
    handoffs=[emailer_agent],
    model="gpt-4o-mini",
    input_guardrails=[guardrail_against_name]  # <-- Guardrail!
)
```

---

## 12. Memory i Persistence

### 12.1 LangGraph Checkpointing

```python
# Z lab: 4_langgraph/2_lab2.ipynb
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Tworzenie połączenia
db_path = "memory.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
sql_memory = SqliteSaver(conn)

# Kompilacja z checkpointerem
graph = graph_builder.compile(checkpointer=sql_memory)

# Każda konwersacja ma swój thread_id
config = {"configurable": {"thread_id": "user_123"}}
result = graph.invoke({"messages": [{"role": "user", "content": user_input}]}, config=config)

# Podróż w czasie - powrót do wcześniejszego stanu
config = {"configurable": {"thread_id": "1", "checkpoint_id": "some_checkpoint_id"}}
graph.invoke(None, config=config)
```

---

## 13. Async Python w AI Engineering

### 13.1 Podstawy async/await

```python
# Z lab: guides/11_async_python.ipynb
import asyncio

async def do_some_work():
    print("Starting work")
    await asyncio.sleep(1)  # Nieblokujące czekanie
    print("Work complete")

# W Jupyter Notebook
await do_some_work()

# W module Python
if __name__ == "__main__":
    asyncio.run(do_some_work())
```

### 13.2 Równoległe wykonanie

```python
# Z lab: guides/11_async_python.ipynb
async def do_a_lot_of_work_in_parallel():
    # asyncio.gather uruchamia równolegle i czeka na wszystkie
    await asyncio.gather(
        do_some_work(), 
        do_some_work(), 
        do_some_work()
    )

await do_a_lot_of_work_in_parallel()  # 1s zamiast 3s!
```

### 13.3 Praktyczne zastosowanie w agentach

```python
# Z lab: 2_openai/4_lab4.ipynb
async def perform_searches(search_plan: WebSearchPlan):
    """Wykonaj wszystkie wyszukiwania równolegle"""
    print("Searching...")
    tasks = [asyncio.create_task(search(item)) for item in search_plan.searches]
    results = await asyncio.gather(*tasks)
    print("Finished searching")
    return results

async def search(item: WebSearchItem):
    """Pojedyncze wyszukiwanie"""
    input = f"Search term: {item.query}\nReason for searching: {item.reason}"
    result = await Runner.run(search_agent, input)
    return result.final_output
```

---

## 14. Observability i Tracing

### 14.1 OpenAI Traces

```python
# Z lab: 2_openai/1_lab1.ipynb
from agents import trace

with trace("Telling a joke"):
    result = await Runner.run(agent, "Tell a joke")
    print(result.final_output)

# Zobacz trace: https://platform.openai.com/traces
```

### 14.2 LangSmith (LangGraph)

```python
# Wymaga ustawienia LANGCHAIN_API_KEY w .env
# Zobacz: https://langsmith.com
```

### 14.3 Custom Tracers

```python
# Z lab: 6_mcp/trading_floor.py
from tracers import LogTracer
from agents import add_trace_processor

add_trace_processor(LogTracer())

# Custom trace ID
from tracers import make_trace_id
trace_id = make_trace_id(f"{self.name.lower()}")
with trace(trace_name, trace_id=trace_id):
    await self.run_with_mcp_servers()
```

---

## 15. Deployment i Produkcja

### 15.1 Gradio UI

```python
# Z lab: 1_foundations/3_lab3.ipynb
import gradio as gr

def chat(message, history):
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content

gr.ChatInterface(chat, type="messages").launch()
```

### 15.2 Deploy do HuggingFace Spaces

```bash
# Z lab: 1_foundations/4_lab4.ipynb
# 1. Zainstaluj HuggingFace CLI
uv tool install 'huggingface_hub[cli]'

# 2. Zaloguj się
hf auth login --token YOUR_TOKEN_HERE

# 3. Deploy
uv run gradio deploy
# Podaj nazwę (np. "career_conversation"), wybierz cpu-basic, dodaj secrets
```

### 15.3 Produkcyjna aplikacja

```python
# Z lab: 1_foundations/app.py
class Me:
    def __init__(self):
        self.openai = OpenAI()
        self.name = "Ed Donner"
        # Ładowanie danych
        reader = PdfReader("me/linkedin.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            })
        return results

    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini", 
                messages=messages, 
                tools=tools
            )
            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content

if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()
```

---

## 16. Najlepsze praktyki produkcyjne

### 16.1 Zarządzanie błędami

```python
# Z lab: 6_mcp/traders.py
async def run(self):
    try:
        await self.run_with_trace()
    except Exception as e:
        print(f"Error running trader {self.name}: {e}")
    self.do_trade = not self.do_trade
```

### 16.2 Konfiguracja przez zmienne środowiskowe

```python
# Z lab: 6_mcp/trading_floor.py
import os

RUN_EVERY_N_MINUTES = int(os.getenv("RUN_EVERY_N_MINUTES", "60"))
RUN_EVEN_WHEN_MARKET_IS_CLOSED = (
    os.getenv("RUN_EVEN_WHEN_MARKET_IS_CLOSED", "false").strip().lower() == "true"
)
USE_MANY_MODELS = os.getenv("USE_MANY_MODELS", "false").strip().lower() == "true"
```

### 16.3 Scheduler dla cyklicznych zadań

```python
# Z lab: 6_mcp/trading_floor.py
async def run_every_n_minutes():
    add_trace_processor(LogTracer())
    traders = create_traders()
    while True:
        if RUN_EVEN_WHEN_MARKET_IS_CLOSED or is_market_open():
            await asyncio.gather(*[trader.run() for trader in traders])
        else:
            print("Market is closed, skipping run")
        await asyncio.sleep(RUN_EVERY_N_MINUTES * 60)

if __name__ == "__main__":
    print(f"Starting scheduler to run every {RUN_EVERY_N_MINUTES} minutes")
    asyncio.run(run_every_n_minutes())
```

### 16.4 Narzędzia do integracji zewnętrznych

```python
# Z lab: 4_langgraph/sidekick_tools.py
from playwright.async_api import async_playwright
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit, FileManagementToolkit
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_experimental.tools import PythonREPLTool
from langchain_community.utilities import GoogleSerperAPIWrapper

async def playwright_tools():
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    return toolkit.get_tools(), browser, playwright

async def other_tools():
    push_tool = Tool(name="send_push_notification", func=push, description="Send push notification")
    file_tools = FileManagementToolkit(root_dir="sandbox").get_tools()
    
    tool_search = Tool(
        name="search",
        func=GoogleSerperAPIWrapper().run,
        description="Use this tool when you want to get the results of an online web search"
    )
    
    wikipedia = WikipediaAPIWrapper()
    wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia)
    
    python_repl = PythonREPLTool()
    
    return file_tools + [push_tool, tool_search, python_repl, wiki_tool]
```

---

## 🎯 Podsumowanie: Twoja ścieżka do AI Engineering

### Poziom 1: Fundamenty (Tydzień 1)
- ✅ Podstawowe wywołania API (OpenAI, Anthropic, Gemini)
- ✅ Zarządzanie kluczami i środowiskami
- ✅ Format messages i role (system, user, assistant)
- ✅ Wzorzec Judge/Evaluator

### Poziom 2: Tool Use i Agentic Patterns (Tydzień 2)
- ✅ Definiowanie narzędzi (JSON Schema)
- ✅ Obsługa wywołań narzędzi
- ✅ Decorator @function_tool
- ✅ Wzorce: Reflection, Parallelization, Orchestration

### Poziom 3: Frameworki (Tydzień 3-5)
- ✅ OpenAI Agents SDK: Agent, Runner, trace, handoffs
- ✅ CrewAI: Multi-agent teams z YAML config
- ✅ LangGraph: State machines, checkpointing, memory
- ✅ AutoGen: Agent communication patterns

### Poziom 4: Produkcja (Tydzień 6)
- ✅ MCP dla standardowych integracji
- ✅ Structured Outputs z Pydantic
- ✅ Guardrails i bezpieczeństwo
- ✅ Observability (traces, logging)
- ✅ Deployment (Gradio, HuggingFace Spaces)

---

## 📖 Dodatkowe zasoby

- [OpenAI Agents SDK Docs](https://openai.github.io/openai-agents-python/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [CrewAI Docs](https://docs.crewai.com/)
- [AutoGen Docs](https://microsoft.github.io/autogen/)
- [MCP Specification](https://modelcontextprotocol.io/)
- [OpenAI Traces](https://platform.openai.com/traces)
- [LangSmith](https://langsmith.com)

---

> **Autor:** Wygenerowano na podstawie repozytorium `agents` - kursu "Master AI Agentic Engineering"
