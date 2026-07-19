
"""
Chatbot with Conversation Summary + Window Memory (Single File)

Features
--------
- Groq (llama-3.3-70b-versatile)
- LCEL (Prompt | LLM | Parser)
- RunnableWithMessageHistory
- InMemoryChatMessageHistory
- Unique Session ID
- Window Memory (Last 10 messages)
- Conversation Summary (older messages)
- Debug prints
"""

import uuid

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# LLM
# ----------------------------
llm = ChatGroq(
    model=os.getenv("CHAT_MODEL_NAME"),
    temperature=0
)

parser = StrOutputParser()

# ----------------------------
# Chat Prompt
# ----------------------------
chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.
Conversation Summary:
{summary}

Use the summary together with the recent conversation
history to answer the user accurately.
""",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

chat_chain = chat_prompt | llm | parser

# ----------------------------
# Summary Prompt
# ----------------------------
summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You maintain a running summary.
Existing Summary:
{summary}
Conversation To Compress:
{conversation}

Update the summary.

Rules:
- Keep important facts.
- Keep names.
- Keep goals.
- Keep preferences.
- Remove unnecessary details.
- Return only the updated summary.
"""
        )
    ]
)

summary_chain = summary_prompt | llm | parser

# ----------------------------
# Stores
# ----------------------------
history_store = {}
summary_store = {}

WINDOW_SIZE = 10

def get_session_history(session_id: str):
    """Return chat history for a session."""
    if session_id not in history_store:
        history_store[session_id] = InMemoryChatMessageHistory()
    return history_store[session_id]

chatbot = RunnableWithMessageHistory(
    chat_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def update_summary(session_id: str):
    """Summarize older messages when history exceeds window."""
    history = get_session_history(session_id)

    if len(history.messages) <= WINDOW_SIZE:
        return

    old_messages = history.messages[:-WINDOW_SIZE]

    conversation = "\n".join(
        f"{m.type.upper()}: {m.content}" for m in old_messages
    )

    current_summary = summary_store.get(session_id, "")

    updated_summary = summary_chain.invoke(
        {
            "summary": current_summary,
            "conversation": conversation,
        }
    )

    summary_store[session_id] = updated_summary

    # keep only latest window
    history.messages = history.messages[-WINDOW_SIZE:]


def print_debug(session_id: str):
    """Print summary and recent history."""
    history = get_session_history(session_id)

    print("\n================ SUMMARY ================")
    print(summary_store.get(session_id, "(empty)"))

    print("\n=========== LAST 10 MESSAGES ============")
    for i, msg in enumerate(history.messages, 1):
        print(f"{i}. {msg.type.upper()}: {msg.content}")
    print("=========================================")


def chat():
    """Main chat loop."""
    session_id = str(uuid.uuid4())
    summary_store[session_id] = ""

    print("=" * 60)
    print("Chatbot Started")
    print("Session:", session_id)
    print("Type 'exit' to quit.")
    print("=" * 60)

    while True:
        user_input = input("\nYou : ")

        if user_input.lower() == "exit":
            break

        response = chatbot.invoke(
            {
                "input": user_input,
                "summary": summary_store[session_id],
            },
            config={
                "configurable": {
                    "session_id": session_id
                }
            },
        )

        print("\nAI :", response)

        update_summary(session_id)
        print_debug(session_id)

    print("\nGoodbye!")


if __name__ == "__main__":
    chat()
