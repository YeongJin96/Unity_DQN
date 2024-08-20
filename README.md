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
