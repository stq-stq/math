import requests

url = "https://api.siliconflow.cn/v1/chat/completions"

payload = {
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [
        {
            "role": "user",
            "content": "中国大模型行业2025年将会迎来哪些机遇和挑战？"
        }
    ],
    "stream": True,
    "max_tokens": 512,
    "stop": ["null"],
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 50,
    "frequency_penalty": 0.5,
    "n": 1,
    "response_format": {"type": "text"},
    "tools": [
        {
            "type": "function",
            "function": {
                "description": "<string>",
                "name": "<string>",
                "parameters": {},
                "strict": False
            }
        }
    ]
}
headers = {
    "Authorization": "Bearer sk-awgonomhddmxylhnsjnbsbbjqommtkgtmjybazkobtosybgf",
    "Content-Type": "application/json"
}

response = requests.request("POST", url, json=payload, headers=headers)

print(response.text)

# import requests
#
# url = "https://api.siliconflow.cn/v1/chat/completions"
#
# payload = {
#     "model": "deepseek-ai/DeepSeek-V2.5",  # 替换成你的模型
#     "messages": [
#         {
#             "role": "user",
#             "content": "SiliconCloud公测上线，每用户送3亿token 解锁开源大模型创新能力。对于整个大模型应用领域带来哪些改变？"
#         }
#     ],
#     "stream": True  # 此处需要设置为stream模式
# }
#
# headers = {
#     "accept": "application/json",
#     "content-type": "application/json",
#     "authorization": "Bearer your-api-key"
# }
#
# response = requests.post(url, json=payload, headers=headers, stream=True)  # 此处request需要指定stream模式
#
# # 打印流式返回信息
# if response.status_code == 200:
#     for chunk in response.iter_content(chunk_size=8192):
#         if chunk:
#             decoded_chunk = chunk.decode('utf-8')
#             print(decoded_chunk, end='')
# else:
#     print('Request failed with status code:', response.status_code)