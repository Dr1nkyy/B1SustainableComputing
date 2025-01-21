import requests

headers = {
  'Accept': 'application/json'
}

# Obtain current CPU and GPU power

# Make a request to the API
response = requests.get('http://10.19.27.119:8085/data.json', headers=headers)
data = response.json()

# Navigate through the JSON structure to retrieve the CPU Package Power value
cpu_package_power = data['Children'][0]['Children'][1]['Children'][1]['Children'][0]['Value']
gpu_power = data['Children'][0]['Children'][3]['Children'][0]['Children'][0]['Value']

# Convert the string to a float
cpu_package_power=float(cpu_package_power[:-1])
gpu_power=float(gpu_power[:-1])

# Obtain current Carbon Intensity
headers = {
  'Accept': 'application/json'
}
 
r = requests.get('https://api.carbonintensity.org.uk/generation', params={}, headers = headers)
CIcurrent = r.json()

# Print the current CPU and GPU power
print("Current CPU Package Power: ", cpu_package_power, "W")
print("Current GPU Power: ", gpu_power, "W")  

# Print the current Carbon Intensity
print("Current Carbon Intensity: ", CIcurrent['data'][0]['intensity']['actual'], "gCO2/kWh")  