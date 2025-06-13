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

SUMO_BINARY = "sumo"
sumo_cmd = [SUMO_BINARY, "-c", "grid.sumocfg"]

def get_state(tls_id):
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    ew_lanes = [l for l in lanes if 'E' in l or 'W' in l]
    ns_lanes = [l for l in lanes if 'N' in l or 'S' in l]
    ew_waiting = sum(traci.lane.getLastStepVehicleNumber(l) for l in ew_lanes)
    ns_waiting = sum(traci.lane.getLastStepVehicleNumber(l) for l in ns_lanes)
    ped_waiting = get_pedestrian_waiting(tls_id)
    return (ew_waiting, ns_waiting, ped_waiting)

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
    return -(car_wait + ped_wait)

def run():
    traci.start(sumo_cmd)
    step = 0
    max_steps = 500
    switch_interval = 10

    tls_ids = traci.trafficlight.getIDList()
    print("Available TLS IDs:", tls_ids)

    agents = {}
    logic_map = {}
    actions = [0, 1]  # phase indices
    phase_record = {}
    rewards_over_time = []

    for tls_id in tls_ids:
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
        if len(logic.phases) >= 2:
            agents[tls_id] = QLearningAgent(actions)
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

        for tls_id in agents:
            state = get_state(tls_id)
            reward = get_reward(tls_id)
            total_reward += reward
            action = phase_record[tls_id]
            next_state = get_state(tls_id)
            agents[tls_id].update_q(state, action, reward, next_state)
            agents[tls_id].decay_epsilon()

        rewards_over_time.append(total_reward)
        print(f"Step {step}: Total reward = {total_reward}")
        step += 1

    traci.close()

    print("\n=== Q-Tables ===")
    for tls_id, agent in agents.items():
        print(f"Agent at {tls_id}")
        for state, actions in agent.q_table.items():
            print(f"  State {state}:")
            for action, q in actions.items():
                print(f"    Phase {action}: {round(q, 2)}")

    plt.plot(rewards_over_time)
    plt.xlabel('Time Step')
    plt.ylabel('Total Reward (vehicles + pedestrians)')
    plt.title('Learning Progress with Pedestrians')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run()