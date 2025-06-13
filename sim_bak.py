import traci
import sumolib
import os
import sys
import time
import matplotlib.pyplot as plt
from q_agent import QLearningAgent

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please set the SUMO_HOME environment variable")

SUMO_BINARY = "sumo-gui"
sumo_cmd = [SUMO_BINARY, "-c", "grid.sumocfg"]

MODE = "before"  # "before" 또는 "after" 선택 가능
Q_TABLE_PATH = "q_table.pkl"
SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

def get_state(tls_id):
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    car_wait = sum(traci.lane.getLastStepVehicleNumber(l) for l in lanes)
    ped_wait = get_pedestrian_waiting(tls_id)
    current_phase = traci.trafficlight.getPhase(tls_id)
    state = (car_wait, ped_wait, current_phase)
    return state

def get_pedestrian_waiting(tls_id):
    crossings = traci.trafficlight.getControlledLinks(tls_id)
    ped_edges = set()
    for entry in crossings:
        for conn in entry:
            if len(conn) > 3 and conn[3]:
                ped_edges.add(conn[3])
    total = 0
    for edge in ped_edges:
        for l in traci.edge.getLanes(edge):
            total += traci.lane.getLastStepPersonNumber(l)
    return total

def get_reward(tls_id):
    lanes = traci.trafficlight.getControlledLanes(tls_id)

    car_wait = sum(traci.lane.getLastStepVehicleNumber(l) for l in lanes)
    ped_wait = get_pedestrian_waiting(tls_id)
    reward = - (car_wait + 0.5 * ped_wait)
    return reward

def run():
    print("Starting SUMO...")
    traci.start(sumo_cmd)
    # traci.start(["sumo-gui", "-c", "grid.sumocfg"])
    print("Starting SUMO simulation...")
    step = 0
    max_steps = 500
    switch_interval = 10

    tls_ids = traci.trafficlight.getIDList()
    print("Available TLS IDs:", tls_ids)

    agents = {}
    logic_map = {}
    actions = [0, 1]  
    phase_record = {}
    rewards_over_time = []
    rewards_by_agent = {tls_id: [] for tls_id in tls_ids}

    for tls_id in tls_ids: #신호 별 에이전트
        logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        if len(logic.phases) >= 2:
            agent = QLearningAgent(actions)
            if MODE == "after" and os.path.exists(Q_TABLE_PATH):
                agent.load(Q_TABLE_PATH)
                print(f"Loaded Q-table for {tls_id} from {Q_TABLE_PATH}")
                agent.epsilon = 0.05

            agents[tls_id] = agent
            logic_map[tls_id] = logic
            phase_record[tls_id] = 0
        else:
            print(f"Skipping {tls_id} (only {len(logic.phases)} phase)")
 
    while step < max_steps:
        total_reward = 0

        if step % switch_interval == 0:
            for tls_id in agents:
                state = get_state(tls_id)
                action = agents[tls_id].choose_action(state)
                phase_record[tls_id] = action
                max_phase = len(logic_map[tls_id].phases)
                if action < max_phase:
                    traci.trafficlight.setPhase(tls_id, action)

        traci.simulationStep()

        # 스크린샷 저장
        # traci.gui.screenshot(viewID="View #0", filename=f"{SCREENSHOT_DIR}/step_{step}.png")

        for tls_id in agents:
            state = get_state(tls_id)
            reward = get_reward(tls_id)
            total_reward += reward
            action = phase_record[tls_id]
            next_state = get_state(tls_id)
            rewards_by_agent[tls_id].append(reward) 
            if MODE == "before":
                agents[tls_id].update_q(state, action, reward, next_state)
                agents[tls_id].decay_epsilon()

        rewards_over_time.append(total_reward)
        print(f"Step {step}: Total reward = {total_reward}")
        step += 1

    traci.close()

    if MODE == "before":
        for tls_id, agent in agents.items():
            agent.save(Q_TABLE_PATH)
        print("Saved Q-table after learning.")

    print("\n===== Q-Table Summary =====")
    for tls_id, agent in agents.items():
        print(f"Q-table for {tls_id}:")
        for state, actions_q in agent.q_table.items():
            print(f"  State {state} → {actions_q}")

    plt.plot(rewards_over_time)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title(f'Learning Progress')
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(12, 6))
    for tls_id, rewards in rewards_by_agent.items():
        plt.plot(rewards, label=f"{tls_id}")
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title(f'Reward Traffic Light')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    # run()
    NUM_EPISODES = 3 if MODE == "before" else 1
    for episode in range(NUM_EPISODES):
        print(f"\n==== EPISODE {episode+1} ({MODE.upper()}) ====")
        run()
