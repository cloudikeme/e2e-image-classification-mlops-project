import requests
import json

input_path = "inference-test.json"

with open(input_path) as json_file:
	data = json.load(json_file)

r = requests.post(url="http://localhost:8080/v1/models/fufu-sample:predict", data=json.dumps(data), headers={'Host': 'fufu-sample.kubeflow.example.com'})
print(r.text)