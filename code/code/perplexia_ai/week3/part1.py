"""Part 1 - Tool-Using Agent implementation.

This implementation focuses on:
- Converting tools from Assignment 1 to use with LangGraph
- Using the ReAct pattern for autonomous tool selection
- Comparing manual workflow vs agent approaches
"""

from typing import Dict, List, Optional
from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.tools.calculator import Calculator
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from datetime import datetime
import os
from langgraph.graph import StateGraph, MessagesState, START, END

# Opik imports
from opik import track
from opik.integrations.langchain import OpikTracer
import opik


class WorkflowState(MessagesState):
    query: str
    result: str


class ToolUsingAgentChat(ChatInterface):
    """Week 3 Part 1 implementation focusing on tool-using agents."""

    def __init__(self):
        # Initialize Opik client
        self.opik_client = opik.Opik()
        
        model_kwargs = {"model": "gpt-4o-mini"}
        # Get environment variables at runtime
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_api_base = os.getenv("OPENAI_API_BASE")

        if openai_api_key:
            model_kwargs["api_key"] = openai_api_key
        if openai_api_base:
            model_kwargs["base_url"] = openai_api_base

        # Create OpikTracer for LangChain integration
        self.opik_tracer = OpikTracer()

        self.llm = create_react_agent(
            init_chat_model(**model_kwargs),
            tools=[self._calculator_tool, self._get_datetime, self._get_search_tool],
        )

        self.tools = []
        builder = StateGraph(WorkflowState)
        builder.add_node("answer_llm", self.answer_agent)
        builder.add_edge(START, "answer_llm")  # entry point
        builder.add_edge("answer_llm", END)  # exit point

        self.graph = builder.compile()

    def initialize(self) -> None:
        """Initialize components for the tool-using agent.

        Students should:
        - Initialize the chat model
        - Define tools for calculator, DateTime, and weather
        - Create the ReAct agent using LangGraph
        """
        pass

    @track(name="process_message", project_name="week3_tool_agent")
    def process_message(
        self, message: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Process a message using the tool-using agent.

        Args:
            message: The user's input message
            chat_history: Previous conversation history

        Returns:
            str: The assistant's response
        """
        result = self.graph.invoke(WorkflowState(query=message))
        return result["result"]

    @track(name="answer_agent", project_name="week3_tool_agent")
    def answer_agent(self, state: WorkflowState):
        """Use the agent to answer based on the current state."""
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an autonomous agent that can reason and take actions using the tools provided. 
Your primary goal is to provide accurate answers by using the tools first whenever possible. 
You should only answer directly without a tool if no tool is relevant for the query.

Available tools (use these in priority order):

1. Calculator
- Name: calculator
- Description: Can perform any mathematical calculation.
- Input: A math expression as a string, e.g., "2 + 2".

2. DateTime
- Name: get_datetime
- Description: Returns the current date and time.
- Input: No input required.

3. Weather
- Name: get_weather
- Description: Returns the current temperature in a given city.
- Input: A city name as a string, e.g., "Paris".

Instructions:
- Think step by step before taking any action.
- Always try to use the most relevant tool first, according to the priority order above.
- Only answer directly if no tool can help.
- Your output must follow this JSON format exactly:

{{
"thought": "Briefly explain your reasoning",
"action": {{
    "tool": "tool_name_or_null_if_answer_directly",
    "input": "input_for_tool_or_null"
}},
"answer": "Your final answer to the user, if ready"
}}

- If a tool is needed, fill in "tool" and "input", and set "answer" to null.  
- If you can answer without using any tool, set "tool" and "input" to null, and fill in "answer".""",
                ),
                ("user", "{input_message}"),
            ]
        )
        message = answer_prompt.format_messages(input_message=state["query"])
        
        # Use Opik tracer with LangChain
        response = self.llm.invoke(
            {"messages": [("user", state["query"])]}, 
            config={"callbacks": [self.opik_tracer]}
        )

        # Extract the final answer from the last message
        state["result"] = response["messages"][-1].content
        return state

    @tool("get_calculator_tool")
    def _calculator_tool(expression: str) -> str:
        """A tool to evaluate any mathematical expression, including arithmetic, percentages, and calculations. Always use this tool for math questions instead of solving them manually."""
        print("Calculating expression tool is invoked:", expression)
        result = Calculator.evaluate_expression(expression)
        return str(result)

    @tool("get_datetime_tool")
    def _get_datetime() -> str:
        """
        Returns the current date and time.
        Example: get_datetime()
        """
        now = datetime.now()
        print("DateTime tool is invoked")
        formatted = now.strftime("%A, %B %d, %Y %I:%M %p")
        return f"The current date and time is {formatted}."

    @tool("get_search_tool")
    @track(name="weather_search_tool", project_name="week3_tool_agent")
    def _get_search_tool(location: str) -> str:
        """
        A tool to return the weather of specific location .
        Example: get_search_tool("paris")
        """
        print("Search tool is invoked with query:", location)
        
        try:
            search_tool = TavilySearch(
                max_results=5,
                include_answer=False,  # True provides an answer from Tavily's side. Use False in assignments.
                include_raw_content=False,
                include_images=False,
                search_depth="basic",  # advanced/basic are the two options
            )

            model_kwargs = {"model": "gpt-4o-mini"}
            # Get environment variables at runtime
            openai_api_key = os.getenv("OPENAI_API_KEY")
            openai_api_base = os.getenv("OPENAI_API_BASE")

            if openai_api_key:
                model_kwargs["api_key"] = openai_api_key
            if openai_api_base:
                model_kwargs["base_url"] = openai_api_base

            weather_llm = init_chat_model(**model_kwargs)
            print("Created search tool and weather LLM successfully")

            search_results = search_tool.run({"query": f'What is the weather in that location {location} in celsius?'})
            print("Got search results successfully")
            
            if isinstance(search_results, dict) and "results" in search_results:
                res = search_results["results"]
            else:
                res = search_results

            # Take only first result and limit content size
            first_result = res[0] if res else {}
            content = first_result.get('content', '')[:1000]  # Limit to 1000 chars

            summarize_prompt = ChatPromptTemplate.from_messages([
                ("system", "Extract the current temperature in Celsius. Return only the number with °C."),
                ("user", "Weather data for {location}: {content}")
            ])

            message = summarize_prompt.format_messages(location=location, content=content)
            print("Summarize prompt message:", message)
            response = weather_llm.invoke(message)
            return response.content
            
        except Exception as e:
            print(f"Error in search tool: {e}")
            return "Error retrieving weather information"
