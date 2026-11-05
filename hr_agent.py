import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Load environment variables from the .env file
load_dotenv()

async def main():
    # 1. Initialize the model client
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    # 2. Create the AssistantAgent with the required system message
    agent = AssistantAgent(
        name="hr_assistant",
        model_client=model_client,
        system_message="You are an HR assistant."
    )

    # 3. Define the prompt requesting exactly 5 bullet points
    prompt = (
        "Write a 5-line job description for a Junior Python Developer. "
        "Format it strictly as exactly 5 bullet points, nothing more."
    )

    # 4. Stream the response directly to the console
    await Console(agent.run_stream(task=prompt))

    # 5. Properly close the client so the process doesn't hang
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())