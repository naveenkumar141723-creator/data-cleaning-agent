import pandas as pd
from typing import TypedDict

from langchain_ollama import ChatOllama
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode


df = pd.read_csv("data/sales.csv")


class State(TypedDict):
    messages: list

@tool
def inspect_data() -> str:


    info = []

    info.append(f"Rows: {len(df)}")
    info.append(f"Columns: {list(df.columns)}")

    info.append("\nData types:")
    info.append(str(df.dtypes))

    info.append("\nMissing values:")
    info.append(str(df.isnull().sum()))

    info.append(f"\nDuplicate rows: {df.duplicated().sum()}")

    return "\n".join(info)


@tool
def remove_duplicates() -> str:


    global df

    before = len(df)

    df = df.drop_duplicates()

    removed = before - len(df)

    return f"Removed {removed} duplicate rows."


@tool
def fill_missing_values() -> str:

    global df

    changes = []

    for column in df.columns:

        missing = df[column].isnull().sum()

        if missing == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):

            value = df[column].median()

            df[column] = df[column].fillna(value)

            changes.append(
                f"{column}: filled {missing} values with median {value}"
            )

        else:

            mode = df[column].mode()

            if not mode.empty:

                value = mode.iloc[0]

                df[column] = df[column].fillna(value)

                changes.append(
                    f"{column}: filled {missing} values with mode '{value}'"
                )

    if not changes:
        return "No missing values needed to be filled."

    return "\n".join(changes)


@tool
def fix_numeric_columns() -> str:


    global df

    changes = []

    for column in df.columns:

        converted = pd.to_numeric(df[column], errors="coerce")

        original_missing = df[column].isnull().sum()
        new_missing = converted.isnull().sum()

        if new_missing == original_missing:
            if converted.notnull().sum() == df[column].notnull().sum():
                df[column] = converted
                changes.append(f"{column}: converted to numeric.")

    if not changes:
        return "No numeric columns needed conversion."

    return "\n".join(changes)


@tool
def validate_data() -> str:


    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()

    result = [
        f"Rows after cleaning: {len(df)}",
        f"Columns after cleaning: {len(df.columns)}",
        f"Remaining missing values: {missing}",
        f"Remaining duplicate rows: {duplicates}",
    ]

    if missing == 0 and duplicates == 0:
        result.append("Validation successful: no missing values or duplicates.")

    else:
        result.append("Validation found remaining data-quality issues.")

    return "\n".join(result)


@tool
def save_cleaned_data() -> str:


    df.to_csv("output/cleaned_data.csv", index=False)

    return "Cleaned dataset saved as output/cleaned_data.csv."


tools = [
    inspect_data,
    remove_duplicates,
    fill_missing_values,
    fix_numeric_columns,
    validate_data,
    save_cleaned_data,
]


llm = ChatOllama(
    model="qwen3"
)

llm_with_tools = llm.bind_tools(tools)


SYSTEM_PROMPT = """
You are an AI data-cleaning agent.

Your job is to inspect a CSV dataset, identify data-quality
problems, clean the dataset using the available tools, validate
the result, and save the cleaned dataset.

Rules:

1. Always inspect the dataset before modifying it.
2. Never invent information about the dataset.
3. Use tools to perform actual data operations.
4. Remove duplicate rows if duplicates exist.
5. Fill missing values when they exist.
6. Fix numeric columns when appropriate.
7. Validate the dataset after cleaning.
8. Save the cleaned dataset after successful validation.
9. Explain what cleaning operations were performed.

Do not claim that something was fixed unless a tool actually
performed the operation.
"""


def chatbot(state: State):

    messages = state["messages"]

    messages_with_prompt = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ] + messages

    response = llm_with_tools.invoke(messages_with_prompt)

    return {
        "messages": messages + [response]
    }



def should_continue(state: State):

    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "end"



graph = StateGraph(State)

graph.add_node("chatbot", chatbot)

graph.add_node(
    "tools",
    ToolNode(tools)
)

graph.add_edge(START, "chatbot")

graph.add_conditional_edges(
    "chatbot",
    should_continue,
    {
        "tools": "tools",
        "end": END,
    }
)

graph.add_edge("tools", "chatbot")


app = graph.compile()


result = app.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": """
                Clean the sales.csv dataset.
                Inspect it first, fix any data-quality problems,
                validate the result, and save the cleaned dataset.
                """
            }
        ]
    }
)


final_message = result["messages"][-1]

print("\nFINAL AGENT RESPONSE:")
print(final_message.content)