import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

async def main():
    # 1. Get the topic/role from user input at runtime
    user_role = input("Enter job role: ").strip()
    if not user_role:
        print("No role provided. Exiting.")
        return

    # 2. Initialize the client and agent
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    agent = AssistantAgent(
        name="hr_assistant",
        model_client=model_client,
        system_message="You are an HR assistant."
    )

    # 3. Dynamic prompt
    prompt = (
        f"Write a 5-line job description for a {user_role}. "
        "Format it strictly as exactly 5 bullet points, nothing more."
    )

    # 4. Stream response and close client
    await Console(agent.run_stream(task=prompt))
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())