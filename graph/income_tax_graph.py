# %%
from dotenv import load_dotenv

load_dotenv()

# %%
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-4o-mini')

# %%
from langsmith import Client

client = Client()

# %%
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(model='text-embedding-3-large'),
    collection_name='income_tax_collections',
    persist_directory='./income_tax_collections'
)
retriever = vector_store.as_retriever(search_kwargs={'k': 3})

# %%
from typing import Literal
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str

# %%
def retrieve(state: AgentState) -> AgentState:
    """
    'retrieve' Node
    : 사용자의 질문에 기반하여, 벡터 스토어에서 관련 문서를 검색한다.

    Args:
        - state(AgentState): 사용자의 질문을 포함한 에이전트의 현재 state

    Returns:
        - AgentState: 검색된 문서가 추가된 state
    """
    
    query = state['query']
    context = retriever.invoke(query)
    
    return {'context': context}

# %%
rag_prompt = client.pull_prompt("rlm/rag-prompt", include_model=True)

def generate(state: AgentState) -> AgentState:
    """
    'generate' Node
    : 사용자의 질문과 검색된 문서를 기반으로 응답을 생성한다.

    Args:
        - state(AgentState): 사용자의 질문과 검색된 문서를 포함한 에이전트의 현재 state

    Returns:
        - AgentState: 생성된 응답이 추가된 state
    """
    
    query = state['query']
    context = state['context']
    
    rag_chain = rag_prompt | llm
    ai_message = rag_chain.invoke({'question': query, 'context': context})
    
    return {'answer': ai_message}

# %%
dictionary = ["사람과 관련된 표현 또는 단어 -> 거주자"]

rewrite_prompt = PromptTemplate.from_template(
    f"""
    사용자의 질문을 보고, 아래 사전을 참고해 질문을 변경해주세요.
    
    - 원본 질문의 의도와 맥락은 변경하지 않고, 사전에 정의된 정보만 변경해야합니다.
    - 제목, 설명 등 부가적인 요소들을 제외하고 변경한 질문만 반환해주세요.
    - 변경하지 않아도 된다고 판단되면 원본 질문을 그대로 반환해주세요.
    
    사전: {dictionary}
    질문: {{query}}
    """
)

def rewrite(state: AgentState) -> AgentState:
    """
    'rewrite' Node
    : 사용자의 질문을 사전을 참고하여 변경한다.

    Args:
        - state(AgentState): 사용자의 질문을 포함한 에이전트의 현재 state

    Returns:
        - AgentState: 변경된 질문을 포함하는 state
    """
    
    query = state['query']
    
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    ai_message = rewrite_chain.invoke({'query': query})
    
    return {'query': ai_message}

# %%
doc_relevance_prompt = client.pull_prompt("langchain-ai/rag-document-relevance", include_model=True)

def check_doc_relevance(state: AgentState) -> Literal['relevant', 'irrelevant']:
    """
    : 주어진 state를 기반으로 문서의 관련성을 판단한다.

    Args:
        - state(AgentState): 사용자의 질문과 문맥을 포함한 에이전트의 현재 state

    Returns:
        - Literal['relevant', 'irrelevant']: 문서가 관련성이 높으면 'relevant', 그렇지 않으면 'irrelevant' 반환
    """
    
    query = state['query']
    context = state['context']
    
    doc_relevance_chain = doc_relevance_prompt | llm
    ai_message = doc_relevance_chain.invoke({'question': query, 'documents': context})
    
    ## node를 직접 지정하는 방식 대신 실제 판단 결과를 리턴함으로써 해당 node의 재사용성을 높일 수 있다.
    return 'relevant' if ai_message['Score'] == 1 else 'irrelevant'

# %%
hallucination_prompt = client.pull_prompt("langchain-ai/rag-answer-hallucination", include_model=True)

def check_hallucination(state: AgentState) -> Literal['hallucinated', 'not hallucinated']:
    """
    : 주어진 state를 기반으로 답변의 할루시네이션을 판단한다.

    Args:
        - state(AgentState): 사용자의 질문과 문맥을 포함한 에이전트의 현재 state

    Returns:
        - Literal['hallucinated', 'not hallucinated']: 답변의 할루이네이션 여부
    """
    
    answer = state['answer']
    context = state['context']
    
    hallucination_chain = hallucination_prompt | llm
    ai_message = hallucination_chain.invoke({'documents': context, 'student_answer': answer})
    
    return 'not hallucinated' if ai_message['Score'] == 1 else 'hallucinated'

# %%
helpfulness_prompt = client.pull_prompt("langchain-ai/rag-answer-helpfulness", include_model=True)

def check_helpfulness_grader(state: AgentState) -> str:
    """
    : 주어진 state를 기반으로 답변의 유용성을 판단한다.

    Args:
        - state(AgentState): 사용자의 질문과 문맥을 포함한 에이전트의 현재 state

    Returns:
        - Literal['helpful', 'unhelpful']: 답변의 유용하다면 'helpful', 그렇지 않으면 'unhelpful' 반환
    """
    
    query = state['query']
    answer = state['answer']
    
    helpfulness_chain = helpfulness_prompt | llm
    ai_message = helpfulness_chain.invoke({'question': query, 'student_answer': answer})
    
    return 'helpful' if ai_message['Score'] == 1 else 'unhelpful'


def check_helpfulness(state: AgentState) -> AgentState:
    """
    'check_helpfulness' Node
    : graph에서 conditional_edge를 연속으로 사용하지 않고, node를 추가해 가독성을 높이기 위해 사용한다.

    Args:
        - state(AgentState): 에이전트의 현재 state

    Returns:
        - AgentState: 변경되지 않은 state
    """
    
    return state

# %%
from langgraph.graph import StateGraph, START, END

graph_builder = StateGraph(AgentState)

# nodes
graph_builder.add_node('retrieve', retrieve)
graph_builder.add_node('generate', generate)
graph_builder.add_node('rewrite', rewrite)
graph_builder.add_node('check_helpfulness', check_helpfulness)

# edges
graph_builder.add_edge(START, 'retrieve')
graph_builder.add_conditional_edges(
    'retrieve',
    check_doc_relevance,
    {
        'relevant': 'generate',
        'irrelevant': END
    }
)
graph_builder.add_conditional_edges(
    'generate',
    check_hallucination,
    {
        'not hallucinated': 'check_helpfulness',
        'hallucinated': 'generate'
    }
)
graph_builder.add_conditional_edges(
    'check_helpfulness',
    check_helpfulness_grader,
    {
        'helpful': END,
        'unhelpful': 'rewrite'
    }
)
graph_builder.add_edge('rewrite', 'retrieve')

# %%
graph = graph_builder.compile()
