# Forklift Route Optimization Using Contextual Thompson Sampling

This project was developed to enhance real-time decision-making in forklift operations at Uline’s high-volume distribution centers. By applying a **Contextual Thompson Sampling algorithm with softmax scaling**, we enable adaptive route selection that balances efficiency and exploration across dynamic warehouse conditions.

## 🚀 Objective

Forklift travel efficiency can vary due to temporary congestion, task types, and time-based constraints. Instead of relying on static routes or deterministic heuristics, this model:
- Learns from past routing outcomes
- Adapts to contextual information like zone/task
- Continuously improves via feedback loops

## 🧠 Model Summary

- **Contextual Thompson Sampling** agent
- **Softmax temperature scaling** to balance exploration vs. exploitation
- **Short-term memory buffer** to capture recent performance patterns
- **Beta-distributed belief update** for each possible route decision

## ⚙️ How It Works

For each route decision:
1. Context (e.g., zone, time of day) is used as a key
2. Bid/route options are sampled from Beta distributions
3. Probabilities are scaled via softmax (temperature `τ`)
4. The selected route is executed
5. Feedback (e.g., win/loss or delay/success) is used to update beliefs

## 📁 Files

- `thompson-stm.py`: Main model class
- `example_simulation.py`: Example usage script (to be added)
- `README.md`: Project overview
- `requirements.txt`: Required packages

## 🏭 Uline Use Case

This model can be embedded in a warehouse management system (WMS) to:
- Select optimal forklift paths based on historical performance
- Adapt routing strategies in real-time to minimize travel time
- Improve warehouse throughput and reduce fuel usage

## 🔧 Future Directions

- Integration with IoT sensor data and forklift telemetry
- Multi-agent coordination and collision avoidance
- Hybrid with reinforcement learning for long-term planning

---

