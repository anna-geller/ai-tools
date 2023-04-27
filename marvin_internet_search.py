import asyncio

from marvin import Bot
from marvin.plugins.duckduckgo import DuckDuckGo

instructions = """
    Act as an expert in AI, ML and data engineering. Be friendly and helpful.
    Your job is to answer questions about various aspects of working with data. 
    You will always need to call your plugins to get the most up-to-date information.
    Do not assume you know the answer without calling a plugin. 
    Do not ask the user for clarification before you attempt a plugin call. 
    Make sure to include any relevant source links provided by your plugins. 

    These are your plugins:
    - `DuckDuckGo`: search the web for answers that other plugins can't answer.
    """

ama_bot = Bot(
    name="AMA",
    personality="Friendly, helpful, and knowledgable in AI and data engineering",
    instructions=instructions,
    reminder="Remember to use your plugins!",
    plugins=[DuckDuckGo()],
    llm_model_name="gpt-3.5-turbo",
    llm_model_temperature=0.2,
)


async def main():
    await ama_bot.save(if_exists="update")


if __name__ == "__main__":
    asyncio.run(main())
