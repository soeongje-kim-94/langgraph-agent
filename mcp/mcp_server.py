from datetime import datetime

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langsmith import Client

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base

load_dotenv()

# MCP server
mcp = FastMCP("house_tax")

# LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# vector store
vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    collection_name="real_estate_tax_collections",
    persist_directory="./real_estate_tax_collections",
)

# retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# web search tool
tavily_search_tool = TavilySearch(max_results=5, topic="general")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def basic_rag_prompt():
    client = Client()

    return client.pull_prompt("rlm/rag-prompt", include_model=True)


def search_market_ratio():
    return tavily_search_tool.invoke(f"{datetime.now().year}년도 공정시장가액비율은?")


###########################################
# MCP tools
###########################################


@mcp.tool(
    name="tax_base_tool",
    description="""
    종합부동산세 과세표준을 계산하기 위한 공식을 검색하고 형식화한다.
    
    이 도구는 RAG 방식을 통해 다음 두 단계로 동작한다.
    1. 지식 베이스에서 과세표준 계산 규칙을 검색한다.
    2. 검색한 규칙을 수학 공식으로 형식화한다.
        
    Returns:
        - str: 형식화된 과세표준 계산 공식
    """,
)
def tax_base_tool() -> str:
    tax_base_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | basic_rag_prompt()
        | llm
        | StrOutputParser()
    )

    return tax_base_chain.invoke(
        "주택에 대한 종합부동산세 과세표준을 계산하는 방법은 무엇인가요? 수식으로 표현해서 수식만 반환해주세요."
    )


@mcp.tool(
    name="tax_deduction_tool",
    description="""
    사용자의 부동산 소유 현황에 대한 질문을 기반으로 세금 공제액을 계산한다.
    
    이 도구는 다음 두 단계로 작동한다.
    1. 'tax_deduction_info_chain'을 사용하여 일반적인 세금 공제 규칙을 검색한다.
    2. 'user_deduction_chain'을 사용하여 사용자의 상황에 맞춰 규칙을 적용한다.
    
    Args:
        - question(str): 부동산 소유 현황에 대한 사용자의 질문
        
    Returns:
        - str: 세금 공제액 (예: '9억원','12억원')
    """,
)
def tax_deduction_tool(question: str) -> str:
    user_deduction_prompt = PromptTemplate(
        template="""
        아래 [Context]는 주택에 대한 종합부동산세의 공제액에 관한 내용입니다.
        사용자의 질문을 통해 보유 주택수에 대한 공제액이 얼마인지 금액만(예: '9억원','12억원') 반환해주세요.
        
        [Context]
        {tax_deduction_info}
        
        [Question]
        {question}
        """,
        input_variables=["tax_deduction_info", "question"],
    )

    # 세금 공제액 정보 검색 체인
    tax_deduction_info_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | basic_rag_prompt()
        | llm
        | StrOutputParser()
    )
    # 사용자별 공제액 계산 체인
    user_deduction_chain = user_deduction_prompt | llm | StrOutputParser()
    # 체인 연결
    tax_deduction_chain = (
        tax_deduction_info_chain
        | RunnableLambda(lambda x: {"tax_deduction_info": x, "question": question})
        | user_deduction_chain
    )

    return tax_deduction_chain.invoke("주택에 대한 종합부동산세 과세표준의 공제액을 알려주세요.")


@mcp.tool(
    name="market_ratio_tool",
    description="""
    사용자의 부동산 소유 현황에 대한 질문을 기반으로 적용되는 공정시장가액비율을 결정한다.
    
    이 도구는 다음 세 단계로 작동한다.
    1. 현재 공정시장가액비율 정보가 포함된 검색 결과를 사용한다.
    2. 사용자의 특정 상황(보유 부동산 수, 부동산 가치)을 분석한다.
    3. 적절한 공정시장가액비율을 백분율로 반환한다.
    
    Args:
        - question(str): 부동산 소유 현황에 대한 사용자의 질문
        
    Returns:
        - str: 백분율로 표현된 공정시장가액비율 (예: '60%','45%')
    """,
)
def market_ratio_tool(question: str) -> str:
    market_ratio_prompt = PromptTemplate.from_template(
        """
        아래 [Context]는 공정시장가액비율에 관한 내용입니다. 
        주어진 공정시장가액비율에 관한 내용을 기반으로, 사용자의 상황에 대한 공정시장가액비율을 알려주세요.
        별도의 설명 없이 공정시장가액비율만 반환해주세요.

        [Context]
        {context}

        [Question]
        {question}
        """
    )

    market_ratio_chain = market_ratio_prompt | llm | StrOutputParser()

    return market_ratio_chain.invoke({"context": search_market_ratio(), "question": question})


@mcp.tool(
    name="house_tax_tool",
    description="""
    수집된 모든 정보를 사용하여 최종 종합부동산세액을 계산한다.
    
    이 도구는 다음 정보들을 종합하여, 최종 세액을 계산한다.
    - 과세표준 계산 공식
    - 공제액
    - 공정시장가액비율
    - 세율표

    Args:
        - tax_base(str): 과세표준 계산 공식
        - tax_deduction(str): 공제액
        - market_ratio(str): 공정시장가액비율
        - question(str): 부동산 소유 현황에 대한 사용자의 질문
        
    Returns:
        - str: 설명이 포함된 최종 세금 계산 금액
    """,
)
def house_tax_tool(tax_base: str, tax_deduction: str, market_ratio: str, question: str) -> str:
    house_tax_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                당신은 부동산 관련 법률 전문가입니다.
                아래 정보들을 활용해서 세금을 계산해주세요.
                
                - 과세표준 계산 공식: {tax_base}
                - 공제액: {tax_deduction}
                - 공정시장가액비율: {market_ratio}
                - 세율: {tax_rate}
                """,
            ),
            ("human", "{question}"),
        ]
    )

    house_tax_chain = (
        {
            "tax_base": lambda _: tax_base,
            "tax_deduction": lambda _: tax_deduction,
            "market_ratio": lambda _: market_ratio,
            "tax_rate": RunnableLambda(lambda x: format_docs(retriever.invoke(x["question"]))),
            "question": lambda x: x["question"],
        }
        | house_tax_prompt
        | llm
        | StrOutputParser()
    )

    return house_tax_chain.invoke({"question": question})


###########################################
# MCP prompts
###########################################


@mcp.prompt(name="agent_system_prompt", description="Agent 시스템 프롬프트")
def agent_system_prompt():
    content = """
    당신의 역할은 주택에 대한 종합부동산세를 계산하는 것입니다.
    사용자의 질문을 바탕으로 종합부동산세를 계산해주세요.

    종합부동산세를 계산하기 위해서는 아래 세 가지의 정보를 파악해야합니다.
    첫 번째, 과세표준을 어떻게 계산해야하는가
    두 번째, 사용자에 질문에 따른 공제액이 얼마인가
    세 번째, 사용자에 질문에 따른 공정시장가액비율이 어떻게 되는가
    위 정보들을 모두 수집했다면, 종합부동산세를 계산해주세요.
    """

    return base.UserMessage(content=agent_system_prompt)


###########################################
# MCP server
###########################################

if __name__ == "__main__":
    mcp.run(transport="stdio")
