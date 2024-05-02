import os

from langchain.agents import AgentType, Tool, initialize_agent

from langchain.llms import Bedrock
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain import LLMMathChain
from langchain.memory import ConversationBufferMemory


def get_tools():
    return [
        Tool(
            name="Search",
            func=DuckDuckGoSearchAPIWrapper().run,
            description="useful for when you need to ask with search",
        ),
        Tool(
            name="Calculator",
            func=LLMMathChain.from_llm(llm=Bedrock(model_id="anthropic.claude-v2"), verbose=False).run,
            description="useful for when you need to answer questions about math"
        )
    ]

def main():
    user = os.getenv("USER")

    agent = initialize_agent(
        get_tools(), 
        Bedrock(model_id="anthropic.claude-v2"),
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=ConversationBufferMemory(memory_key="chat_history"),
        agent_kwargs={
            'prefix': """Você é um assistente financeiro especialista em Renda Fixa brasileira e investimentos no Brasil""", 
            'suffix': """Você deve usar as Tools como mecanismo para ir a internet buscar informações e fazer cálculos quando necessário"
                {chat_history}
                Question: {input}
                {agent_scratchpad}"""
        },
        verbose=True
    )

    while True:
        user_input = input(f'{user}: ')

        if (user_input.lower() == 'exit' or 
            user_input.lower() == 'clear'):

            print('Leaving chat... bye')
            break

        print(agent.run({"input": user_input}))

if __name__ == '__main__':
    main()