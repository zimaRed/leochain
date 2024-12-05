# leochain is a simple framework for building multi-agent systems using GPT.
#
# it consists of a few parts:
# - a supervisor, which manages the conversation and the agents
# - agents, which are responsible for specific tasks and can use specific tools
# - tools, which are used by agents to perform specific tasks
# - an ensemble that runs the agents and the supervisor in a loop
#
# you can use this framework by instantiating an ensemble, then adding supervisor/agents/tools, and then running the ensemble in a loop.
#
# this system is fully composable-- ensembles can be stacked and nested to create complex systems of agent teams.
# (a tool can actually call an ensemble to run a set of agents, and then return the result)
#
# limitations: only works with openai models, agents can only call one tool at a time, agents can't write/read files... many more limitations

# import the necessary libraries
from dotenv import load_dotenv
from getpass import getpass
from openai import OpenAI
from pyswip import Prolog
import json
import os

load_dotenv()

# set the openai API key if it is not already set
def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass(f"Please provide your {var}")

_set_if_undefined("OPENAI_API_KEY")

# set up the openai client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# select openai model
model = "gpt-4-1106-preview"

# set a maximum loop count
max_loop_count = 10

# hide excess logs or not
hide_excess_logs = False

# allow repeat runs of the same worker
allow_repeat_runs = False

# define an openai agent utility that manages the openai interaction for agents
# - requires an agent object and message history
def agent_openai(agent, message_history):
    # get the agent's name, prompt, and tools
    name = agent.name
    prompt = agent.prompt
    tools = agent.tools

    # construct tools schema by mapping the tools to their schemas
    tool_schemas = [tool.schema for tool in tools]

    # construct the chat context (prompt + message history)
    prompt_message = {
        "role": "system",
        "content": prompt,
    }
    context = [prompt_message] + message_history

    # call openai api
    if len(tool_schemas) == 0:
        response = client.chat.completions.create(
            model=model,
            messages=context,
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=context,
            tools=tool_schemas,
        )
    message = response.choices[0].message
    
    # check if a tool was used (just check the first in the list for now)
    # if so, call the tool and run another call to generate a message given the tool's result
    if message.tool_calls and len(message.tool_calls) > 0:
        # get the tool's name and arguments
        tool_name = message.tool_calls[0].function.name
        tool_arguments = json.loads(message.tool_calls[0].function.arguments, strict=False)

        # find the tool that matches the name
        tool = [t for t in tools if t.schema["function"]["name"] == tool_name][0]

        # execute the tool
        if not hide_excess_logs:
            print("[INTERNAL]", name + ":", "Using tool", tool_name, "with arguments:", tool_arguments)
        result = str(tool.execute(tool_arguments))
        if not hide_excess_logs:
            print("[INTERNAL]", name + ":", "Tool output --", result)

        # call openai api again to generate a message given the tool's result
        context = context + [{"role": "assistant", "content": "This is the result of running the " + tool_name + " tool: '" + result + "'. Given this result, report back with the answer (or lack thereof)."}]
        response = client.chat.completions.create(
            model=model,
            messages=context,
            tools=tool_schemas,
            tool_choice="none"
        )
        message = response.choices[0].message

        # return the message content
        return message.content
    # if not, just return the openai response content
    else:
        return message.content

# define an openai supervisor utility that manages the openai interaction for supervisors
# - requires a supervisor object and the message history
def supervisor_openai(supervisor, message_history):
    # get the supervisor's prompt and available agents
    prompt = supervisor.prompt
    agents = supervisor.agents

    # get the most recent message
    last_message = message_history[-1]
    # get the actor that sent the last message
    last_actor = None
    if last_message["role"] == "assistant":
        last_actor = last_message["content"].split(": ")[0]
    # remove last_actor from options if not allowing repeat runs
    if not allow_repeat_runs:
        agents = [agent for agent in agents if agent.name != last_actor]

    # next actor selection prompt
    next_actor_prompt = "Given the conversation above, who should act next? Or should we FINISH? Select one of: " + ", ".join([agent.name for agent in agents]) + ", FINISH" + ". Use the select_next_worker function to decide."

    # turn the agent options into an openai function calling schema
    tools = [
        {
            "type": "function",
            "function": {
                "name": "select_next_worker",
                "description": "Select the worker to act next.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "worker": {
                            "type": "string",
                            "description": "The name of the worker to act next, or 'FINISH' if the goal is complete or the answer has been found.",
                        },
                    },
                    "required": ["worker"],
                },
            },
        }
    ]

    # construct the chat context (prompt + message history + next actor prompt)
    prompt_message = {
        "role": "system",
        "content": prompt,
    }
    next_actor_prompt_message = {
        "role": "system",
        "content": next_actor_prompt,
    }
    context = [prompt_message] + message_history + [next_actor_prompt_message]

    # call openai api, require the select_next_worker function to be used
    response = client.chat.completions.create(
        model=model,
        messages=context,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "select_next_worker"}}
    )
    message = response.choices[0].message
    arguments = json.loads(message.tool_calls[0].function.arguments, strict=False)
    # if nothing was selected, return FINISH
    if arguments["worker"] == "":
        return "FINISH"
    # otherwise return the selected agent or FINISH
    return arguments["worker"]

# define the tool class
# - requires an openai-style tool schema and a "run" function that takes the tool's arguments (from the schema) and returns the result
class Tool:
    def __init__(self, schema, run_function):
        self.schema = schema
        self.run_function = run_function
    
    def execute(self, arguments):
        # get required arguments from schema
        schema_args = self.schema["function"]["parameters"]["required"]

        # check that all required arguments are present
        if not all(arg in arguments for arg in schema_args):
            missing_args = [arg for arg in schema_args if arg not in arguments]
            return "Error -- Missing required arguments: " + ", ".join(missing_args)

        # run the tool's run_function and return the result
        return self.run_function(**arguments)


# define the agent class
# - requires a name, a prompt defining the agent's role, and a list of tools
class Agent:
    def __init__(self, name, prompt, tools=[]):
        self.name = name
        self.prompt = prompt
        self.tools = tools

    def invoke(self, message_history):
        return agent_openai(self, message_history)

# define the supervisor class
# - requires a prompt defining the supervisor's role, and a list of agents
class Supervisor:
    def __init__(self, prompt, agents):
        self.prompt = prompt
        self.agents = agents
        self.loop_count = 0

    def invoke(self, message_history):
        # check if over maximum loop count
        if self.loop_count >= max_loop_count:
            print("Supervisor: Reached maximum loop count.")
            return "FINISH"
        else:
            result = supervisor_openai(self, message_history)
            if result == "FINISH":
                selection = "FINISH"
            else:
                selection = [agent for agent in self.agents if agent.name == result][0]
        # increment loop count
        self.loop_count += 1
        return selection

# define the ensemble class
# - requires a prompt defining the ensemble's goals/capabilities, and a supervisor
# (if the option selected is "FINISH", the ensemble should stop running and generate a final answer to return)
class Ensemble:
    def __init__(self, prompt, supervisor):
        self.prompt = prompt
        self.supervisor = supervisor
        self.message_history = []

    def invoke(self):
        #add the user prompt to message history if empty
        if len(self.message_history) == 0:
            self.message_history.append({"role": "user", "content": self.prompt})
        # loop that calls either supervisor.invoke or agent.invoke, alternating
        while True:
            # get the current actor
            actor = self.supervisor.invoke(self.message_history)
            # if the actor is "FINISH", return the result
            if actor == "FINISH":
                print("[FINISH]")
                # remove the actor name and colon from the final result
                final_result = self.message_history[-1]["content"].split(": ")[1]
                return final_result
            # otherwise, invoke the actor and add the result to the message history
            else:
                print("[" + actor.name + "]")
                result = actor.invoke(self.message_history)
                self.message_history.append({"role": "assistant", "content": actor.name + ": " + result})
                if not hide_excess_logs:
                    print(actor.name + ": " + result)



"""
if __name__ == "__main__":
    # example of a simple ensemble
    weather_tool_schema = {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string", 
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. By default, it should be celsius.",
                    },
                },
                "required": ["location", "unit"],
            },
        },
    }
    def get_current_weather(location, unit):
        return f"The current weather in {location} is 17 degrees {unit}."
    weather_tool = Tool(weather_tool_schema, get_current_weather)
    # define an example "meteorologist" agent
    meteorologist = Agent(
        "Meteorologist",
        "You are a meteorologist. Your job is to use the tools at your disposal to find the current weather in a given location.",
        [weather_tool]
    )
    weather_supervisor = Supervisor(
        "You are a supervisor. Your job is to manage the conversation between the meteorologist and the user.",
        [meteorologist]
    )
    weather_team = Ensemble(
        "Your goal is to figure out the current weather in San Francisco.",
        weather_supervisor
    )
    # test running the weather team ensemble
    weather_team.invoke()
"""
