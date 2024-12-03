import tkinter as tk
import time
import numpy as np
import random

# Parameters for Q-learning
alpha = 0.1  # Learning rate
gamma = 0.85  # Discount factor
epsilon = 0.1  # Exploration rate
episodes = 1000  # Number of training episodes


class RouteOptimizer:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=900, height=400, bg="lightblue")
        self.canvas.pack()

        # Create the vehicle and destination on the canvas
        self.vehicle = self.canvas.create_rectangle(40, 280, 90, 330, fill="green")
        self.destination = self.canvas.create_oval(830, 50, 880, 100, fill="gold")
        self.canvas.create_text(855, 75, text="Target", fill="black")

        # Define the coordinates for each route
        self.paths = [
            [(60, 300), (150, 300), (250, 220), (400, 220), (600, 180), (830, 75)],  # Path 1
            [(60, 300), (170, 270), (270, 250), (450, 200), (700, 140), (830, 75)],  # Path 2
            [(60, 300), (190, 290), (310, 230), (500, 210), (750, 160), (830, 75)],  # Path 3
            [(60, 300), (210, 310), (370, 260), (550, 200), (800, 130), (830, 75)],  # Path 4
            [(60, 300), (230, 320), (450, 270), (700, 210), (830, 75)],  # Path 5 (optimal)
            [(60, 300), (250, 330), (500, 300), (750, 220), (830, 75)],  # Path 6
            [(60, 300), (270, 310), (550, 250), (830, 100)],  # Path 7
        ]
        self.draw_paths()

        # Initialize Q-table with zeros
        self.q_table = np.zeros((len(self.paths), 2))  # 2 actions: Move forward or Stay

    def draw_paths(self):
        # Draw each path with unique styles
        colors = ['blue', 'purple', 'red', 'green', 'orange', 'pink', 'cyan']
        for i, path in enumerate(self.paths):
            for j in range(len(path) - 1):
                self.canvas.create_line(
                    path[j][0], path[j][1], path[j + 1][0], path[j + 1][1],
                    fill=colors[i], width=3
                )
            # Label each path
            self.canvas.create_text(path[0][0] + 30, path[0][1] - 15, text=f"Path {i + 1}",
                                    fill=colors[i], font=("Helvetica", 10, "bold"))

    def reset_vehicle(self):
        # Reset vehicle position to the starting point
        self.canvas.coords(self.vehicle, 40, 280, 90, 330)
        self.canvas.update()

    def display_best_path(self, path_index):
        # Display the vehicle moving along the best path
        path = self.paths[path_index]
        for position in path:
            self.canvas.coords(
                self.vehicle, position[0] - 25, position[1] - 25, position[0] + 25, position[1] + 25
            )
            self.canvas.update()
            time.sleep(0.1)

    def train_agent(self):
        for episode in range(episodes):
            path_index = random.randint(0, len(self.paths) - 1)
            print(f"Episode {episode + 1}: Exploring Path {path_index + 1}")

            # Initialize state and cumulative reward
            state = path_index
            total_reward = 0

            for _ in range(len(self.paths[path_index])):
                # Choose action: Explore or Exploit
                if random.uniform(0, 1) < epsilon:
                    action = random.choice([0, 1])  # 0 = Stay, 1 = Move Forward
                else:
                    action = np.argmax(self.q_table[state])

                # Take action and calculate reward
                if action == 1:  # Move Forward
                    reward = 15 if path_index == 4 else -2  # Higher reward for optimal path (Path 5)
                    next_state = path_index
                else:
                    reward = -5  # Penalty for staying
                    next_state = state

                # Update Q-value
                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                self.q_table[state, action] = new_value

                # Update state and reward
                state = next_state
                total_reward += reward

            print(f"Total reward for Episode {episode + 1}: {total_reward}")

        print("Training completed!")
        print("Optimal Q-values:\n", self.q_table)

    def run(self):
        # Train the agent and display the best path
        self.train_agent()
        best_path = np.argmax(np.max(self.q_table, axis=1))
        print(f"Best Path Identified: Path {best_path + 1}")
        self.display_best_path(best_path)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Route Optimization Simulator")
    optimizer = RouteOptimizer(root)
    optimizer.run()
    root.mainloop()
