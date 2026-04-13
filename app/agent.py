from __future__ import annotations

from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, StateGraph

from app.config import settings

COUNTRY_KEYWORDS = {
    "germany": ["германия", "берлин", "германии", "германию", "germany", "berlin", "немецкий"],
    "france": ["франция", "париж", "франции", "францию", "france", "paris", "французский"],
}

COUNTRY_SPECIFIC_TOPICS = [
    "стипендия", "налог", "рабочий день", "рабочее время", "виза",
    "ставка", "зарплата", "оплата", "salary", "stipend", "tax", "visa",
    "рабочих", "часов", "работа", "hours", "локация", "location",
]

SYSTEM_PROMPT = """Ты — помощник по программе международной стажировки CdekStart от СДЭК.

Отвечай СТРОГО на основе предоставленных документов из базы знаний.
Если информации нет в документах — честно скажи, что не знаешь.
НЕ придумывай и НЕ домысливай информацию.
Отвечай на том языке, на котором задан вопрос.
Будь конкретным и лаконичным.

Если пользователь спрашивает про конкретную страну — отвечай только по документу этой страны.
Если в истории диалога упоминалась страна — используй эту информацию при ответе."""


class AgentState(TypedDict):
    query: str
    history: list[BaseMessage]
    retrieved_docs: list
    needs_clarification: bool
    response: str


def get_llm() -> BaseChatModel:
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )
    else:
        from langchain_openai import ChatOpenAI
        kwargs: dict = {"model": settings.openai_model}
        if settings.openai_api_key:
            kwargs["api_key"] = settings.openai_api_key
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        return ChatOpenAI(**kwargs)


def _detect_country(messages: list[BaseMessage]) -> tuple[bool, bool]:
    full_text = " ".join(m.content for m in messages).lower()
    germany = any(kw in full_text for kw in COUNTRY_KEYWORDS["germany"])
    france = any(kw in full_text for kw in COUNTRY_KEYWORDS["france"])
    return germany, france


def _is_country_specific_topic(query: str) -> bool:
    q = query.lower()
    return any(topic in q for topic in COUNTRY_SPECIFIC_TOPICS)


def _build_retrieval_query(state: AgentState) -> str:
    query = state["query"]
    history = state["history"]
    if not history:
        return query
    recent = history[-4:]
    context_text = " ".join(m.content for m in recent)
    return f"{context_text} {query}"


def make_agent(vectorstore: VectorStore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = get_llm()

    def retrieve_node(state: AgentState) -> AgentState:
        retrieval_query = _build_retrieval_query(state)
        docs = retriever.invoke(retrieval_query)
        return {**state, "retrieved_docs": docs}

    def check_ambiguity_node(state: AgentState) -> AgentState:
        query = state["query"]
        all_messages = state["history"] + [HumanMessage(content=query)]
        germany_mentioned, france_mentioned = _detect_country(all_messages)

        if germany_mentioned or france_mentioned:
            return {**state, "needs_clarification": False}

        if _is_country_specific_topic(query):
            return {**state, "needs_clarification": True}

        sources = [doc.metadata.get("source", "") for doc in state["retrieved_docs"]]
        has_germany = any("germany" in s for s in sources)
        has_france = any("france" in s for s in sources)
        if has_germany and has_france:
            return {**state, "needs_clarification": True}

        return {**state, "needs_clarification": False}

    def clarify_node(state: AgentState) -> AgentState:
        response = (
            "Уточните, пожалуйста, для какой страны вас интересует информация: "
            "Германия (Берлин) или Франция (Париж)?"
        )
        return {**state, "response": response}

    def answer_node(state: AgentState) -> AgentState:
        docs = state["retrieved_docs"]

        if not docs:
            return {
                **state,
                "response": "К сожалению, у меня нет информации по этому вопросу в базе знаний.",
            }

        all_messages = state["history"] + [HumanMessage(content=state["query"])]
        germany_mentioned, france_mentioned = _detect_country(all_messages)

        if germany_mentioned and not france_mentioned:
            country_docs = [d for d in docs if "germany" in d.metadata.get("source", "")]
            general_docs = [d for d in docs if "germany" not in d.metadata.get("source", "") and "france" not in d.metadata.get("source", "")]
            docs = country_docs + general_docs if country_docs else docs
        elif france_mentioned and not germany_mentioned:
            country_docs = [d for d in docs if "france" in d.metadata.get("source", "")]
            general_docs = [d for d in docs if "germany" not in d.metadata.get("source", "") and "france" not in d.metadata.get("source", "")]
            docs = country_docs + general_docs if country_docs else docs

        context = "\n\n".join(
            f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        )

        messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
        messages.extend(state["history"])
        messages.append(
            HumanMessage(
                content=(
                    f"Контекст из базы знаний:\n{context}\n\n"
                    f"Вопрос пользователя: {state['query']}"
                )
            )
        )

        ai_message = llm.invoke(messages)
        return {**state, "response": ai_message.content}

    def route_after_check(state: AgentState) -> str:
        return "clarify" if state["needs_clarification"] else "answer"

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("check_ambiguity", check_ambiguity_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "check_ambiguity")
    graph.add_conditional_edges(
        "check_ambiguity",
        route_after_check,
        {"clarify": "clarify", "answer": "answer"},
    )
    graph.add_edge("clarify", END)
    graph.add_edge("answer", END)

    return graph.compile()
