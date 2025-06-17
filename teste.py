import json

with open("/home/toma/Downloads/Demo__My_first_AI_Agent_in_n8n.json ", "r") as file:
    model_data = json.load(file)

print(model_data)
