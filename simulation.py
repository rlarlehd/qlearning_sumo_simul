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

            agents[tls_id] = agent
            logic_map[tls_id] = logic
            phase_record[tls_id] = 0
        else:
            print(f"Skipping phase({tls_id}))")
 
    while step < max_steps:
        total_reward = 0

        if step % switch_interval == 0:
            for tls_id in agents:
                state = get_state(tls_id) #상태체크
                action = agents[tls_id].choose_action(state) #정책에 따른 행동 선택
                phase_record[tls_id] = action
                max_phase = len(logic_map[tls_id].phases)
                if action < max_phase:
                    traci.trafficlight.setPhase(tls_id, action)

        traci.simulationStep() #Sumo 시뮬레이션 단계 진행

        for tls_id in agents:
            state = get_state(tls_id)
            reward = get_reward(tls_id) #보상
            total_reward += reward
            action = phase_record[tls_id]
            next_state = get_state(tls_id)
            rewards_by_agent[tls_id].append(reward) #에이전트 별 리워드 합산 (시각화)
            agents[tls_id].update_q(state, action, reward, next_state) #Q값 업데이트
            agents[tls_id].decay_epsilon() #탐험률

        rewards_over_time.append(total_reward)
        print(f"단계 {step}: Total reward = {total_reward}")
        step += 1

    traci.close()

    print("\n===== Q-Table =====")
    for tls_id, agent in agents.items():
        print(f"Q-table {tls_id}:")
        for state, actions_q in agent.q_table.items():
            print(f"  State {state} => {actions_q}")

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
    run()
