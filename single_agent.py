import os 
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI


load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

serper_tool=SerperDevTool()

llm=ChatOpenAI(model="gpt-3.5-turbo")


def create_research_agent():
    return Agent(
        role="Research Specialist",
        goal="Conduct thorough research on the topics",
        backstory="you are an experienced research specialist with a knack for finding information",
        llm=llm,
        verbose=True,
        tools=[serper_tool],
        allow_delegation=False,
    )



def create_research_task(agent, topic):
    return Task(
        description=f"Research the following topic and provide a comprehensive summary:`{topic}`",
        agent=agent,
        expected_output="A detailed summary of the research findings including key points, trends, and any other relevant information"
    )

def run_research(topic):
    research_agent=create_research_agent()
    task=create_research_task(research_agent, topic)
    crew=Crew(
        agents=[research_agent],
        tasks=[task],
        verbose=True
    )
    result=crew.kickoff()
    return result
    
   
if __name__=="__main__":
    print("Welcome to the Research Agent Demo!")
    topic=input("Enter the topic you want to research:")
    result=run_research(topic)
    print("Research Results:")
    print(result)


