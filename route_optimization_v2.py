# Route Optimization with RL in Python
import openrouteservice
import folium
import numpy as np
import random

# Initialize OpenRouteService client
client = openrouteservice.Client(key='5b3ce3597851110001cf6248f93e429c77804878aaf96a6cc10f4d86')  # Replace with your key

# Define coordinates for origin and destination
origin = (80.92238882182758461, 26.83293623332243)
destination = (80.95041842860792, 26.914135557686038)

# Step 1: Generate alternative routes
routes = []
num_routes = 8  # Number of routes to create
for _ in range(num_routes):
    try:
        # Add variation by creating random waypoints
        waypoints = [
            (origin[0] + random.uniform(-0.008, 0.008), origin[1] + random.uniform(-0.008, 0.008)),
            (destination[0] + random.uniform(-0.008, 0.008), destination[1] + random.uniform(-0.008, 0.008)),
        ]
        route = client.directions(
            coordinates=[origin] + waypoints + [destination],
            profile='driving-car',
            format='geojson'
        )
        routes.append(route)
    except openrouteservice.exceptions.ApiError as error:
        print(f"API Error: {error}")
    except Exception as error:
        print(f"Error encountered: {error}")

# Step 2: Q-learning setup
states = len(routes)
q_matrix = np.zeros((states, states))  # Initialize Q-matrix

# Parameters for reinforcement learning
learning_rate = 0.15
discount_factor = 0.8
exploration_chance = 0.2
training_episodes = 800

# Rewards configuration
goal_reward = 120  # Reward for the optimal path
penalty = -2  # Penalty for non-optimal actions

# For tracking visited routes
visited_routes = []

# Step 3: Training the agent
for episode in range(training_episodes):
    current_state = random.randint(0, states - 1)
    cumulative_reward = 0
    print(f"\nTraining Episode {episode + 1} begins...")

    for _ in range(states - 1):
        # Choose action: Exploration vs Exploitation
        if random.random() < exploration_chance:
            selected_action = random.choice(range(states))
        else:
            selected_action = np.argmax(q_matrix[current_state])

        visited_routes.append(selected_action)
        print(f"  Evaluating Route {selected_action + 1}.")

        # Compute rewards based on distance
        route_distance = routes[selected_action]['features'][0]['properties']['segments'][0]['distance']
        reward = goal_reward - route_distance if selected_action == states - 1 else penalty

        # Update Q-matrix
        old_q_value = q_matrix[current_state, selected_action]
        next_max = np.max(q_matrix[selected_action])
        q_matrix[current_state, selected_action] = (1 - learning_rate) * old_q_value + \
                                                   learning_rate * (reward + discount_factor * next_max)

        # Move to the next state and accumulate rewards
        current_state = selected_action
        cumulative_reward += reward
        if current_state == states - 1:
            break

    print(f"  Total Reward in Episode {episode + 1}: {cumulative_reward}")

# Step 4: Determining the best route
optimal_route_index = np.argmax(np.sum(q_matrix, axis=1))
optimal_route_coords = routes[optimal_route_index]['features'][0]['geometry']['coordinates']
print(f"\nOptimal Route Found: Route {optimal_route_index + 1}")

# Step 5: Print all routes
print("\nDetails of All Routes:")
for idx, route in enumerate(routes):
    print(f"Route {idx + 1}:")
    for coord in route['features'][0]['geometry']['coordinates']:
        print(f"  {coord}")
    print()

# Step 6: Visualizing routes with Folium
route_map = folium.Map(location=(origin[1], origin[0]), zoom_start=13)

# Add markers for origin and destination
folium.Marker((origin[1], origin[0]), tooltip="Origin", icon=folium.Icon(color="green")).add_to(route_map)
folium.Marker((destination[1], destination[0]), tooltip="Destination", icon=folium.Icon(color="red")).add_to(route_map)

# Add all routes in gray for visualization
for idx, route in enumerate(routes):
    route_coordinates = route['features'][0]['geometry']['coordinates']
    folium.PolyLine(
        [(coord[1], coord[0]) for coord in route_coordinates],
        color="gray", weight=2, opacity=0.6
    ).add_to(route_map)

# Highlight the optimal route in blue
folium.PolyLine(
    [(coord[1], coord[0]) for coord in optimal_route_coords],
    color="blue", weight=5, opacity=1
).add_to(route_map)

# Highlight all visited routes in orange
for route_idx in visited_routes:
    explored_route_coords = routes[route_idx]['features'][0]['geometry']['coordinates']
    folium.PolyLine(
        [(coord[1], coord[0]) for coord in explored_route_coords],
        color="orange", weight=3, opacity=0.8
    ).add_to(route_map)

# Save and show the map
route_map.save("optimized_routes_map.html")
print("\nMap of routes saved as 'optimized_routes_map.html'.")
