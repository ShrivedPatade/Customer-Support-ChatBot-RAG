import request

url = 'http://127.0.0.1:5000/chatbot'

# Define the data to send in the POST request
data = {
    "query": "ai",
    "k": 3
}

# Send the POST request
response = request.post(url, json=data)

# Print the response from the API
print(response.json())