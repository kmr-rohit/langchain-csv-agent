from csv_agent.agent import agent_executor

if __name__ == "__main__":
    question = "How many supplier sites do we have ? "
    print(agent_executor.invoke({"input": question}))
