import os
os.environ["TORCH_CUDA_ARCH_LIST"] = ""
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import curses

# Constants
BATCH_SIZE = 32
NUM_EPISODES = 1000

# Crop class
class Crop:
    def __init__(self, crop_type):
        self.crop_type = crop_type
        self.state = "empty"
        self.growth_time = 0
        self.time_to_grow = random.randint(crop_type["min_grow_time"], crop_type["max_grow_time"])

    def get_symbol(self):
        if self.state == "empty":
            return "E"
        elif self.state == "growing":
            return "G"
        else:
            return "M"

# Field class
class Field:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.field = [[Crop(random.choice(crop_types)) for _ in range(width)] for _ in range(height)]
        self.current_position = (0, 0)
        self.soil_nutrients = [[random.uniform(0.3, 0.7) for _ in range(width)] for _ in range(height)]
        self.crop_inventory = {}  # Track the number of crops harvested
        self.seed_inventory = {crop["name"]: 10 for crop in crop_types}  # Start with 10 seeds for each crop type
        self.money = 10.0  # Initial money for the field
        self.temperature = 20  # Initial temperature in degrees Celsius

    def move(self, dx, dy):
        x, y = self.current_position
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < self.width and 0 <= new_y < self.height:
            self.current_position = (new_x, new_y)

    def water(self):
        x, y = self.current_position
        crop = self.field[y][x]
        if crop.state == "growing":
            self.soil_nutrients[y][x] += crop.crop_type["nutrient_value"] * 0.5

    def plant_crop(self, crop_type_name):
        x, y = self.current_position
        crop_type = next((crop for crop in crop_types if crop["name"] == crop_type_name), None)
        if crop_type:
            crop = self.field[y][x]
            if crop.state == "empty" and self.seed_inventory[crop_type_name] > 0:
                crop.crop_type = crop_type
                crop.state = "growing"
                crop.growth_time = 0
                crop.time_to_grow = random.randint(crop_type["min_grow_time"], crop_type["max_grow_time"])
                self.seed_inventory[crop_type_name] -= 1

    def buy_seeds(self, crop_type_name, amount):
        crop_type = next((crop for crop in crop_types if crop["name"] == crop_type_name), None)
        if crop_type and amount > 0:
            cost = crop_type["nutrient_value"] * 0.2 * amount  # Arbitrary cost calculation
            if cost <= self.money:
                # Assume the purchase is successful, and add the seeds to the inventory
                self.seed_inventory[crop_type_name] = self.seed_inventory.get(crop_type_name, 0) + amount
                self.money -= cost

    def sell_crops(self, crop_type_name, amount):
        crop_type = next((crop for crop in crop_types if crop["name"] == crop_type_name), None)
        if crop_type and amount > 0:
            if crop_type_name in self.crop_inventory and self.crop_inventory[crop_type_name] >= amount:
                # Assume the sale is successful, and deduct the sold crops from the inventory
                self.crop_inventory[crop_type_name] -= amount
                earned = crop_type["nutrient_value"] * 0.1 * amount  # Arbitrary earning calculation
                self.money += earned

    def display(self):
        # Print the crop information for all crops
        for y in range(self.height):
            for x in range(self.width):
                crop = self.field[y][x]
                print(
                    f"X: {x}, Y: {y}, Crop: {crop.crop_type['name']}, Nutrient Value: {crop.crop_type['nutrient_value']}, "
                    f"% Grown: {crop.growth_time * 100 / crop.time_to_grow:.1f}%, State: {crop.state}"
                )

        # Print AI's position and last action
        x, y = self.current_position
        print(f"AI Position: X: {x}, Y: {y}")

    def update_crop_growing(self):
        for y in range(self.height):
            for x in range(self.width):
                crop = self.field[y][x]
                if crop.state == "growing":
                    crop.growth_time += 1
                    if crop.growth_time >= crop.time_to_grow:
                        crop.state = "mature"

    def check_crop_death(self):
        for y in range(self.height):
            for x in range(self.width):
                crop = self.field[y][x]
                if crop.state == "growing":
                    death_chance = crop.crop_type["pest_resistance"] * 0.05  # Arbitrary death chance
                    if random.random() < death_chance:
                        crop.state = "empty"

    def get_reward(self):
        reward = 0
        for y in range(self.height):
            for x in range(self.width):
                crop = self.field[y][x]
                if crop.state == "mature":
                    reward += crop.crop_type["nutrient_value"] * 0.1  # Arbitrary reward calculation
                    crop.state = "empty"
                    self.crop_inventory[crop.crop_type["name"]] = self.crop_inventory.get(crop.crop_type["name"], 0) + 1
        return reward

# Weather module
class Weather:
    def __init__(self):
        self.temperature = 20  # Initial temperature in degrees Celsius
        self.weather_effects = {
            "Sunny": {"temperature_change": 2, "description": "A sunny day with warm weather."},
            "Cloudy": {"temperature_change": 0, "description": "A cloudy day with mild weather."},
            "Rainy": {"temperature_change": -2, "description": "A rainy day with cool weather."},
            "Stormy": {"temperature_change": -5, "description": "A stormy day with very cool weather."},
            "Snowy": {"temperature_change": -8, "description": "A snowy day with cold weather."},
            "Foggy": {"temperature_change": -1, "description": "A foggy day with slightly cool weather."},
            "Heatwave": {"temperature_change": 5, "description": "A scorching heatwave with extremely hot weather."},
            "Coldwave": {"temperature_change": -5, "description": "A coldwave with freezing temperatures."},
        }

    def get_weather(self):
        weather_type = random.choice(list(self.weather_effects.keys()))
        return weather_type

    def update_weather(self):
        weather_type = self.get_weather()
        temperature_change = self.weather_effects[weather_type]["temperature_change"]
        self.temperature += temperature_change

    def display_weather(self):
        weather_type = self.get_weather()
        print(f"Weather: {weather_type}")
        print(self.weather_effects[weather_type]["description"])
        print(f"Temperature: {self.temperature}°C")

# DQN model
class DQNModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# DQNAgent class
class DQNAgent:
    def __init__(self, input_size, output_size, crop_types):
        self.model = DQNModel(input_size, output_size)
        self.replay_buffer = []
        self.epsilon = 1.0  # Initial exploration rate (1.0 means always explore)
        self.epsilon_decay = 0.99  # Decay rate for exploration rate
        self.gamma = 0.99  # Discount factor for future rewards
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.crop_types = crop_types  # List of available crop types
        self.crop_indices = {crop["name"]: idx for idx, crop in enumerate(crop_types)}

    def choose_action(self, state, field):
        if random.random() < self.epsilon:
            # Randomly explore with probability epsilon
            return random.randint(0, state.size(1) - 1)
        else:
            # Choose the action with the highest Q-value according to the model
            with torch.no_grad():
                q_values = self.model(state)
            actions = self.get_available_actions(field)
            valid_q_values = q_values[0][actions]
            return actions[torch.argmax(valid_q_values)].item()

    def get_available_actions(self, field):
        actions = []
        x, y = field.current_position
        crop = field.field[y][x]
        if crop.state == "empty":
            actions.extend([5, 6])  # Plant and buy seeds actions are available
        elif crop.state == "mature":
            actions.extend([4, 7])  # Water and sell crops actions are available
        return torch.tensor(actions, dtype=torch.long)

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        self.epsilon *= self.epsilon_decay

    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        batch = random.sample(self.replay_buffer, BATCH_SIZE)

        states = torch.cat([experience[0] for experience in batch])
        actions = torch.tensor([experience[1] for experience in batch], dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor([experience[2] for experience in batch], dtype=torch.float32).unsqueeze(1)
        next_states = torch.cat([experience[3] for experience in batch])
        dones = torch.tensor([experience[4] for experience in batch], dtype=torch.float32).unsqueeze(1)

        q_values = self.model(states).gather(1, actions)
        next_q_values = self.model(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Crop types
crop_types = [
    {"name": "Wheat", "min_grow_time": 5, "max_grow_time": 10, "pest_resistance": 0.7, "nutrient_value": 0.5},
    {"name": "Corn", "min_grow_time": 7, "max_grow_time": 12, "pest_resistance": 0.8, "nutrient_value": 0.6},
    {"name": "Tomato", "min_grow_time": 8, "max_grow_time": 14, "pest_resistance": 0.6, "nutrient_value": 0.7},
    {"name": "Potato", "min_grow_time": 6, "max_grow_time": 11, "pest_resistance": 0.9, "nutrient_value": 0.8},
    {"name": "Carrot", "min_grow_time": 6, "max_grow_time": 10, "pest_resistance": 0.85, "nutrient_value": 0.7},
]

# Function to apply weather effects
def apply_weather_effects(field, weather_module):
    weather_type = weather_module.get_weather()
    temperature_change = weather_module.weather_effects[weather_type]["temperature_change"]
    field.temperature += temperature_change

# Main AI loop with training
def run_agent(field, agent, weather_module, win):
    device = torch.device("cpu")  # Explicitly set the device to CPU
    agent.model.to(device)

    for episode in range(NUM_EPISODES):
        field = Field(10, 10)  # Initialize the field for each episode
        agent.epsilon = 0.9  # Reset epsilon for each episode
        total_reward = 0

        while True:
            state = [crop.state == "mature" for row in field.field for crop in row]
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action = agent.choose_action(state, field)

            x, y = field.current_position
            if action == 0:
                if y > 0:
                    field.move(0, -1)
            elif action == 1:
                if y < field.height - 1:
                    field.move(0, 1)
            elif action == 2:
                if x > 0:
                    field.move(-1, 0)
            elif action == 3:
                if x < field.width - 1:
                    field.move(1, 0)
            elif action == 4:
                field.water()
            elif action == 5:
                crop_type_name = agent.crop_types[0]["name"]  # Get the name of the first crop type in the list
                field.plant_crop(crop_type_name)  # Plant the chosen crop type
            elif action == 6:
                crop_type_name = agent.crop_types[0]["name"]  # Get the name of the first crop type in the list
                field.buy_seeds(crop_type_name, 5)  # Buy seeds of the chosen crop type
            elif action == 7:
                crop_type_name = agent.crop_types[0]["name"]  # Get the name of the first crop type in the list
                field.sell_crops(crop_type_name, 5)  # Sell crops of the chosen crop type
            else:
                print("Invalid action. Skipping...")

            apply_weather_effects(field, weather_module)  # Apply weather effects on the field
            field.update_crop_growing()
            field.check_crop_death()
            next_state = [crop.state == "mature" for row in field.field for crop in row]
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
            reward = field.get_reward()  # Get the reward based on the current state
            agent.store_experience(state, action, reward, next_state, done=False)
            agent.train()  # Train the agent after each action
            total_reward += reward
            field.display()
            weather_module.display_weather()

            time.sleep(5)  # Adjust the time interval between AI actions

            # Check for episode termination conditions (you can customize this)
            if total_reward <= -10 or field.money <= 0:
                break

# Main display loop
def main(stdscr):
    # Initialize the field, agent, and weather module
    field = Field(10, 10)
    agent = DQNAgent(10 * 10, 8, crop_types)  # 8 possible actions: move up, move down, move left, move right, water, plant crop, buy seeds, sell crops
    weather_module = Weather()

    # Run the AI agent in a separate thread
    ai_thread = threading.Thread(target=run_agent, args=(field, agent, weather_module, stdscr))
    ai_thread.daemon = True
    ai_thread.start()

    # Main display loop
    while True:
        stdscr.clear()
        field.display()
        weather_module.display_weather()
        time.sleep(5)  # Adjust the display update interval

if __name__ == "__main__":
    curses.wrapper(main)
