from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel\
                             import EngineConfigurationChannel

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from collections import deque
import random
import copy
import numpy as np

env_name = f"./new_env\KnightsAdventure" #유니티 환경 경로

action_size = 3
print("action_size: ", action_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device: ", device)

load_model = False
train_mode = True

batch_size = 128
mem_maxlen = 50000
discount_factor = 0.999
learning_rate = 0.00025

run_step = 1000000 if train_mode else 0
test_step = 50000
train_start_step = 5000
target_update_step = 1000

print_interval = 10
save_interval = 30

epsilon_eval = 0.05
epsilon_init = 1.0 if train_mode else epsilon_eval
epsilon_min = 0.1
explore_step = run_step * 0.9
eplsilon_delta = (epsilon_init - epsilon_min)/explore_step if train_mode else 0.

save_path = f"./saved_models/models/DQN" #모델을 저장할 폴더
load_path = f"./saved_models/DQN/11-29_2" #저장된 모델 위치

class DQN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(DQN, self).__init__(**kwargs)
        self.conv1 = torch.nn.Linear(438, 32)
        self.conv2 = torch.nn.Linear(32, 64)
        self.conv3 = torch.nn.Linear(64, 64)
        self.flat = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(64, 512)
        self.q = torch.nn.Linear(512, action_size)

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)
        print("input x.shape", x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        print("conv x.shape", x.shape)
        x = self.flat(x)
        print("flat x.shape", x.shape)
        x = F.relu(self.fc1(x))
        return self.q(x)
    
# DQNAgent 클래스 -> DQN 알고리즘을 위한 다양한 함수 정의 
class DQNAgent:
    def __init__(self):
        self.network = DQN().to(device)
        self.target_network = copy.deepcopy(self.network)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=mem_maxlen)
        self.epsilon = epsilon_init
        self.writer = SummaryWriter(save_path)

        if load_model == True:
            print(f"... Load Model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path+'/ckpt', map_location=device)
            self.network.load_state_dict(checkpoint["network"])
            self.target_network.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        
    # Epsilon greedy 기법에 따라 행동 결정 
    def get_action(self, state, training=True):
        #  네트워크 모드 설정
        self.network.train(training)
        epsilon = self.epsilon if training else epsilon_eval
        # 랜덤하게 행동 결정
        if epsilon > random.random():  
            action = np.random.randint(0, action_size, size=(state.shape[0],1))
            
        # 네트워크 연산에 따라 행동 결정
        else:
            q = self.network(torch.FloatTensor(state).to(device))
            action = torch.argmax(q, axis=-1, keepdim=True).data.cpu().numpy()
        return action

    # 리플레이 메모리에 데이터 추가 (상태, 행동, 보상, 다음 상태, 게임 종료 여부)
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 학습 수행
    def train_model(self):
        batch = random.sample(self.memory, batch_size)
        state      = np.stack([b[0] for b in batch], axis=0)
        action     = np.stack([b[1] for b in batch], axis=0)
        reward     = np.stack([b[2] for b in batch], axis=0)
        next_state = np.stack([b[3] for b in batch], axis=0)
        done       = np.stack([b[4] for b in batch], axis=0)

        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device),
                                                        [state, action, reward, next_state, done])

        eye = torch.eye(action_size).to(device)
        one_hot_action = eye[action.view(-1).long()]
        q = (self.network(state) * one_hot_action).sum(1, keepdims=True)

        with torch.no_grad():
            next_q = self.target_network(next_state)
            target_q = reward + next_q.max(1, keepdims=True).values * ((1 - done) * discount_factor)

        loss = F.smooth_l1_loss(q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 엡실론 감소
        self.epsilon = max(epsilon_min, self.epsilon - eplsilon_delta)

        return loss.item()

    # 타겟 네트워크 업데이트
    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())

    # 네트워크 모델 저장 
    def save_model(self, epi):
        print(f"... Save Model to {save_path}/ckpt ...")
        torch.save({
            "network" : self.network.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
        }, save_path+'/epi{}_ckpt'.format(epi))

    # 학습 기록 
    def write_summray(self, score, loss, epsilon, step, duration):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/loss", loss, step)
        self.writer.add_scalar("model/epsilon", epsilon, step)
        self.writer.add_scalar("duration/episode", duration, step)

# Main 함수 -> 전체적으로 DQN 알고리즘을 진행 
if __name__ == '__main__':
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_name,
                           side_channels=[engine_configuration_channel])
    env.reset()
    
    behavior_name = list(env.behavior_specs.keys())[0]
    for i in env.behavior_specs.keys():
        print("behavior", i)
    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=1.0)
    dec, term = env.get_steps(behavior_name)
    print("DEC: ", dec.obs)
    print("term: ", term.obs)
    
    agent = DQNAgent()
    max_epi = 2000
    current_step = 0
    
    losses, scores, duration, episode, score = [], [], [], 0, 0
    for step in range(run_step + test_step):
        if step == run_step:
            if train_mode:
                agent.save_model()
            print("TEST START")
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(time_scale=1.0)

        preprocess = lambda ray1, ray2, ray3, ray4, ray5, ray6: np.concatenate((ray1, ray2, ray3, ray4, ray5, ray6), axis=-1) 
        state = preprocess(dec.obs[0], dec.obs[1], dec.obs[2], dec.obs[3],dec.obs[4], dec.obs[6])
        action = agent.get_action(state, train_mode)
        real_action = action# + 1
        action_tuple = ActionTuple()
        action_tuple.add_discrete(real_action)
        env.set_actions(behavior_name, action_tuple)
        env.step()

        dec, term = env.get_steps(behavior_name)
        done = len(term.agent_id) > 0
        reward = term.reward if done else dec.reward
        if done:
            next_state = preprocess(term.obs[0], term.obs[1], term.obs[2], term.obs[3],term.obs[4],term.obs[6])
        else:
            next_state = preprocess(dec.obs[0], dec.obs[1], dec.obs[2], dec.obs[3],dec.obs[4], dec.obs[6])

        score += reward[0]

        if train_mode:
            agent.append_sample(state[0], action[0], reward, next_state[0], [done])

        if train_mode and step > max(batch_size, train_start_step):
            # 학습 수행
            loss = agent.train_model()
            losses.append(loss)

            # 타겟 네트워크 업데이트 
            if step % target_update_step == 0:
                agent.update_target()

            
        current_step+=1
        
        if done:
            print("done")
            duration.append(current_step)
            episode +=1
            scores.append(score)
            print(score)
            score = 0
            current_step = 0
            env.reset()
            behavior_name = list(env.behavior_specs.keys())[0]
            spec = env.behavior_specs[behavior_name]
            engine_configuration_channel.set_configuration_parameters(time_scale=1.0)
            dec, term = env.get_steps(behavior_name)

            # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록 
            if episode % print_interval == 0:
                mean_score = np.mean(scores)
                mean_loss = np.mean(losses)
                mean_duration = np.mean(duration)
                agent.write_summray(mean_score, mean_loss, agent.epsilon, episode, mean_duration)
                losses, scores, duration = [], [], []

                print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / duration: {mean_duration} / " +\
                      f"Loss: {mean_loss:.4f} / Epsilon: {agent.epsilon:.4f}")

            # 네트워크 모델 저장 
            if train_mode and episode % save_interval == 0:
                agent.save_model(episode)

    env.close()
