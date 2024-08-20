# Unity_DQN

강화학습을 적용시켜볼 게임은 Knights-Adventure라는 게임입니다.

깃허브에서 유저가 만들어 놓은 2D게임에 적용시켜보았습니다.

(책 파이토치와 유니티 ML-Agents로 배우는 강화학습을 참고했습니다.)

[Knights-Addventure Github link](https://github.com/BatuhanDemiray/Knights-Adventure)

---

게임 설명

Knights-Adventure는 몬스터나 장애물을 피해가며 목표지점에 도착하는 게임입니다.

몬스터나 장애물에 부딪힌다면 체력이 소모되고, 체력이 0이 되면 플레이어는 사망합니다.

플레이어는 2단점프로 몬스터를 밟으면 몬스터를 처치할 수 있습니다. 

필드 곳곳에 코인과 체력 포션이 존재하는데, 플레이어가 체력 포션에 닿으면, 플레이어는 체력을 회복하고, 코인에 닿으면 보유 코인량이 올라갑니다. (코인의 쓰임새는 아직 없는것으로 보입니다.)



정리하자면 오브젝트들의 종류는 다음과 같습니다.

#### 적(Enemies)

1. 움직이는 몬스터

2. 고정형 장애물(닿이면 체력 소모)
<p align="center">
  <img src="https://github.com/user-attachments/assets/90f9e000-4d93-4a52-92f4-8f2b0d3ca13d">
</p>
   ![Enermies](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Enermies.png)

#### 수집형 오브젝트(Collectibles)

 1. 코인

 2. 체력 포션

    ![Collectibles](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Collectibles.png)



#### 장애물(Obstacles)

장애물은 플레이어의 체력을 소모시키지 않지만, 진로를 방해합니다.

![Obstacles](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Obstacles.png)



#### 목표 지점(Goal)

다음 Stage로 넘어가는 포탈이지만, 학습할때 목표지점(Goal)로 사용했습니다.

![Portal](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Portal.png)

---



### 유니티 설정

해당 게임은 2D게임으로 연속적인 행동보다는 이산적인 행동에 더 가깝다고 판단되어 DQN을 적용시켜 학습했습니다.

유니티 ML-Agents로 학습 환경을 구성했습니다.



1. OnEpisodeBegin()

   새로운 에피소드가 시작될 때 이전 에피소드에서 바뀐 값들을 초기화합니다.

```c#
public override void OnEpisodeBegin()
    {
        currentPlayerHealth = 100; //Player 체력
        this.transform.localPosition = InitTrans; //Player 위치 초기화
        distance = area.goal.transform.localPosition.x - this.transform.localPosition.x; //학습할 때 Observation으로 사용한 목표지점과 Player의 거리를 초기화
        preDist = distance; //Player가 목표지점에 도달하기 위해 올바른 방향으로 진행중인지 확인하기 위한 변수 초기화
    }
```

2. CollectObservations()

   학습을 위해 관측할 정보들입니다.

   ```c#
   public override void CollectObservations(VectorSensor sensor)
       {
           sensor.AddObservation(area.goal.localPosition); //목표지점의 위치
           sensor.AddObservation(this.transform.localPosition); //Player의 현재 위치
           sensor.AddObservation(distance); //목표지점과 Player의 거리
           sensor.AddObservation(body2D.velocity.x); //Player의 x축 가속도
           sensor.AddObservation(body2D.velocity.y); //Player의 y축 가속도
           sensor.AddObservation(currentPlayerHealth); //Player의 현재 체력
       }
   ```

   스크립트에서 지정한 벡터 정보외에 유니티 내부에서 4개의 Ray를 추가적으로 사용했습니다.

   1. Player가 서 있을수 있는 지형과 장애물을 탐지하는 Ray
   2. 몬스터를 탐지하는 Ray
   3. Player에게 피해를 입힐 수 있는 고정형 Enemy를 탐지하는 Ray
   4. Player가 낙사할때 충돌하는 Object를 탐지하는 Ray

   ![Ground+Enemy_Ray](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Ground+Enemy_Ray.png)

3. OnActionReceived()

   Player의 Action과 그에 맞는 Reward를 책정하는데, Reward가 -10점 이하로 떨어질 경우 Player를 죽이고 Scene을 다시 시작했습니다. (Player가 죽으면 Scene이 Restart됩니다.)

   

   감점

   1. 항상 (-0.01)
   2. Player가 죽었을때 (-5)
   3. Player가 Goal의 반대방향으로 이동할 때 ((preDist - distance)*0.05)
   4. Scene이 시작할때 Player와 Goal의 초기 거리보다 먼 거리에서 죽었을때 (-10)

   

   가점

   1. Player가 Goal의 방향으로 이동할 때 ((preDist - distance)*0.05)
   2. Player가 코인을 획득했을 때 (1)
   3. Player가 Goal에 도달했을 때 (10)

   

   움직임

   Discrete action을 적용했습니다.

   본 게임은 정지(아무것도 하지 않음), 좌, 우, 점프로 총 4개의 action으로 분류할 수 있습니다.

   학습에서는 학습속도를 위해 action을 지정할때 정지를 제외한 뒤 학습했습니다.

   ```c#
   public override void OnActionReceived(ActionBuffers actionsBuffers)
       {
           UpdateAnimations();
           ReduceHealth();
           BoostHealth();
           AddCoin();
   
           distance = area.goal.transform.localPosition.x - this.transform.localPosition.x;
       
           if (currentPlayerHealth <= 0)
           {
               isDead = true;
           }
   
           if (currentPlayerHealth > maxPlayerHealth)
               currentPlayerHealth = maxPlayerHealth;
   
           if (transform.position.y <= -6)
               isDead = true;
   
           if (isDead)
           {
               Debug.Log("Dead");
               KillPlayer();
               AddReward(-5.0f);
               if (distance > Init_distance)
               {
                   AddReward(-10.0f);
               }
               Debug.Log("Final_Reward: " + GetCumulativeReward());
               EndEpisode();
           }
           AddReward(-0.01f);
   
           float reward = (preDist - distance);
           Debug.Log(reward);
           AddReward(reward * 0.05f);
           preDist = distance;
       
           var actions = actionsBuffers.DiscreteActions[0];
           switch (actions)
           {
               case k_Right:
                   body2D.velocity = new Vector2(playerSpeed, body2D.velocity.y);
                   break;
               case k_Left:
                   body2D.velocity = new Vector2(-playerSpeed, body2D.velocity.y);
                   break;
               case k_Up:
                   if (isGround)
                   {
                       body2D.AddForce(new Vector2(0, jumpPower));
                       audioSource.PlayOneShot(audioJump);
                       isGround = false;
                       canDoubleJump = true;
                   }
                   else if (!isGround && canDoubleJump)
                   {
                       body2D.AddForce(new Vector2(0, doubleJumpPower));
                       canDamage = true;
                       audioSource.PlayOneShot(audioJump);
                       canDoubleJump = false;
                   }
                   break;
               default:
                   throw new ArgumentException("Invalid action value");
           }
           Debug.Log("Reward: " + GetCumulativeReward());
       }
   ```
   
   ```c#
   void AddCoin()
       {
           if (earnCoin)
           {
               currentCoin += addCoin.coin;
               AddReward(1.0f);
               coinText.text = currentCoin.ToString();
               earnCoin = false;
               audioSource.PlayOneShot(audioCoin);
           }
       }
   ```
   
   ```c#
   private void OnTriggerEnter2D(Collider2D other)
       {
           if (other.tag == "Goal") 
           {
               AddReward(10f);
               EndEpisode();
           }
       }
   ```

유니티 환경을 세팅한 뒤, 파이썬에서 DQN을 적용시켜 학습했습니다.



학습에 사용한 파라미터는 다음과 같습니다.

```python
state_size = 363
action_size = 3

batch_size = 128
mem_maxlen = 50000
discount_factor = 0.9
learning_rate = 0.00025

run_step = 250000 if train_mode else 0
test_step = 10000
train_start_step = 500
target_update_step = 2000

epsilon_eval = 0.05
epsilon_init = 0.9 if train_mode else epsilon_eval
epsilon_min = 0.1
explore_step = run_step * 0.85
eplsilon_delta = (epsilon_init - epsilon_min)/explore_step if train_mode else 0.
```



state_size는 학습을 위해 유니티 스크립트에서 지정한 Observation Vector와, Ray sensor의 Vector입니다.

![term_obs](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\term_obs.png)

(4개의 벡터와 유니티 스크립트에서 지정했던 Observation Vector)



action_size는 좌, 우, 점프로 총 3종류의 action을 가지고 있습니다.

mem_maxlen은 딥마인드에서 제안한 Capture&Replay의 최대 메모리크기를 담당하는 파라미터입니다.

target_update_step은 딥마인드에서 제안한 Separate-Network, 즉 target_network를 업데이트하는 빈도입니다. 

나머지 파라미터들은 Q-Network와 동일합니다.



DQN()

Input을 이미지로 사용하지 않고 벡터로 사용했기 때문에, CNN이 아닌 Linear를 사용했습니다.

```python
class DQN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(DQN, self).__init__(**kwargs)
        self.conv1 = torch.nn.Linear(state_size, 32)
        self.conv2 = torch.nn.Linear(32, 64)
        self.conv3 = torch.nn.Linear(64, 64)

        self.flat = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(64, 512)
        self.q = torch.nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        return self.q(x)
```



DQNAgent()

```python
class DQNAgent:
    def __init__(self):
        self.network = DQN().to(device)
        self.target_network = copy.deepcopy(self.network)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=mem_maxlen)
        self.epsilon = epsilon_init
        self.writer = SummaryWriter(save_path)
        
    # Epsilon greedy 기법에 따라 행동 결정 
    def get_action(self, state, training=True):
        epsilon = self.epsilon
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
    def save_model(self):
        print(f"... Save Model to {save_path}/ckpt ...")
        torch.save({
            "network" : self.network.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
        }, save_path+'/ckpt')

    # 학습 기록 
    def write_summray(self, score, loss, epsilon, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/loss", loss, step)
        self.writer.add_scalar("model/epsilon", epsilon, step)
```



Main()

Unity환경을 불러오고, DQN 알고리즘을 학습시킵니다.

```python
if __name__ == '__main__':
    # 유니티 환경 경로 설정 (file_name)
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_name,
                           side_channels=[engine_configuration_channel])
    env.reset()

    # 유니티 브레인 설정 
    behavior_name = list(env.behavior_specs.keys())[0]
    for i in env.behavior_specs.keys():
        print("behavior", i)
    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=1.0)
    dec, term = env.get_steps(behavior_name)
    print("DEC: ", dec.obs) # DEC는 에피소드가 진행중일때의 상태
    print("term: ", term.obs) # term은 에피소드가 done일때의 상태

    # DQNAgent 클래스를 agent로 정의 
    agent = DQNAgent()
    
    losses, scores, episode, score = [], [], 0, 0
    for step in range(run_step + test_step):
        if step == run_step:
            if train_mode:
                agent.save_model()
            print("TEST START")
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(time_scale=1.0)
			#Unity환경의 시간(배속)을 설정
        preprocess = lambda level_ray, monster_ray ,enemy_ray, dead_line_ray, obs: np.concatenate((level_ray, monster_ray, enemy_ray, dead_line_ray, obs), axis=-1) 
        #앞서 지정한 Ray벡터들과 Unity 스크립트의 백터를 concatenate
        state = preprocess(dec.obs[0], dec.obs[1], dec.obs[2], dec.obs[3], dec.obs[5])
        action = agent.get_action(state, train_mode)
        real_action = action
        action_tuple = ActionTuple()
        action_tuple.add_discrete(real_action)
        env.set_actions(behavior_name, action_tuple)
        env.step()

        dec, term = env.get_steps(behavior_name)
        print(term.obs)
        done = len(term.agent_id) > 0
        reward = term.reward if done else dec.reward
        if done:
            next_state = preprocess(term.obs[0], term.obs[1], term.obs[2], term.obs[3], term.obs[5])
        else:
            next_state = preprocess(dec.obs[0], dec.obs[1], dec.obs[2], dec.obs[3], dec.obs[5])
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

        if step % 1000 == 0:
            print("current_step: ",step)
            print("current_reward: ",score)
        
        if done:
            print("done")
            episode +=1
            scores.append(score)
            print(score)
            score = 0
            
            env.reset()
            behavior_name = list(env.behavior_specs.keys())[0]
            spec = env.behavior_specs[behavior_name]
            engine_configuration_channel.set_configuration_parameters(time_scale=1.0)
            dec, term = env.get_steps(behavior_name)

            # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록 
            if episode % print_interval == 0:
                mean_score = np.mean(scores)
                mean_loss = np.mean(losses)
                agent.write_summray(mean_score, mean_loss, agent.epsilon, step)
                losses, scores = [], []

                print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " +\
                      f"Loss: {mean_loss:.4f} / Epsilon: {agent.epsilon:.4f}")

            # 네트워크 모델 저장 
            if train_mode and episode % save_interval == 0:
                agent.save_model()

    env.close()

```



#### 학습 결과

![Score_graph](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Knight-adventure-Score.png)

[유튜브Link](https://www.youtube.com/watch?v=Up7Cn4rauiQ)



학습 중 설정한 Reward를 토대로 의도한 방식으로 학습을 하는지 확인하기 위해 Reward값을 수정해 학습한 결과

##### 수정한 Reward

1. (New) Monster를 처치시 +2
2. Coin을 획득할 시 +1 -> -1

※ Coin을 최대한 피하며, Monster를 처치하도록 유도

![Score2](C:\Users\dydyz\OneDrive\바탕 화면\GitHub_blog\YeongJin96.github.io\assets\img\Knight-adventure2-Score.png)

[유튜브Link](https://youtu.be/QLL2W_ohny0)
