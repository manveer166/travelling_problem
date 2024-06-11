# app.py
from flask import Flask, request, jsonify, render_template
import pulp
import googlemaps

app = Flask(__name__)

# Function to calculate distances using Google Maps API
def get_distance_matrix(api_key, addresses):
    gmaps = googlemaps.Client(key=api_key)
    distance_matrix = gmaps.distance_matrix(addresses, addresses, mode='driving')
    distance_matrix_km = [
        [element['distance']['value'] / 1000 for element in row['elements']]
        for row in distance_matrix['rows']
    ]
    return distance_matrix_km

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.get_json()
    api_key = data['api_key']
    patient_addresses = data['patient_addresses']
    worker_names = data['worker_names']
    num_workers = len(worker_names)
    
    # Combine base address and patient addresses
    addresses = ["CV2 2TE, UK"] + patient_addresses
    num_locations = len(addresses)
    distance_matrix = get_distance_matrix(api_key, addresses)

    # Define the problem
    prob = pulp.LpProblem("VehicleRoutingProblem", pulp.LpMinimize)
    
    # Decision variables
    x = pulp.LpVariable.dicts('route', (range(num_workers), range(num_locations), range(num_locations)), cat='Binary')
    u = pulp.LpVariable.dicts('u', (range(num_workers), range(num_locations)), lowBound=0, cat='Continuous')

    # Initializing the patient-worker assignment preferences
    patient_worker_preferences = {i: i % num_workers for i in range(1, num_locations)}

    # Objective function
    prob += pulp.lpSum(distance_matrix[i][j] * x[k][i][j] for k in range(num_workers) for i in range(num_locations) for j in range(num_locations))

    # Constraints to ensure each patient is visited exactly once by one worker
    for j in range(1, num_locations):
        prob += pulp.lpSum(x[k][i][j] for k in range(num_workers) for i in range(num_locations) if i != j) == 1

    # Constraints to ensure each worker starts and ends at the base location
    for k in range(num_workers):
        prob += pulp.lpSum(x[k][0][j] for j in range(1, num_locations)) == 1
        prob += pulp.lpSum(x[k][i][0] for i in range(1, num_locations)) == 1

    # Constraints to ensure that each worker leaves and returns to the depot
    for k in range(num_workers):
        for i in range(num_locations):
            prob += pulp.lpSum(x[k][i][j] for j in range(num_locations) if i != j) == pulp.lpSum(x[k][j][i] for j in range(num_locations) if i != j)

    # Subtour elimination constraints
    for k in range(num_workers):
        for i in range(1, num_locations):
            for j in range(1, num_locations):
                if i != j:
                    prob += u[k][i] - u[k][j] + num_locations * x[k][i][j] <= num_locations - 1

    for k in range(num_workers):
        for i in range(1, num_locations):
            prob += u[k][i] >= 0
            prob += u[k][i] <= num_locations - 1

    # Preferences for worker-patient assignments
    for j in range(1, num_locations):
        preferred_worker = patient_worker_preferences[j]
        prob += pulp.lpSum(x[preferred_worker][i][j] for i in range(num_locations) if i != j) >= 0.5  # Ensuring preferred assignments

    # Solve the problem
    prob.solve()

    # Extracting the solution data
    routes = [[] for _ in range(num_workers)]
    for k in range(num_workers):
        current_location = 0
        while True:
            next_location = None
            for j in range(num_locations):
                if pulp.value(x[k][current_location][j]) == 1:
                    routes[k].append(current_location)
                    next_location = j
                    break
            if next_location is None or next_location == 0:
                routes[k].append(0)
                break
            current_location = next_location

    result = []
    for worker_id, route in enumerate(routes):
        worker_result = {
            "worker": worker_names[worker_id],
            "route": route,
            "distance": sum(distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
        }
        result.append(worker_result)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
