 







<img src="https://www.dropbox.com/s/z9i4l0ha6emxn7d/Screenshot%202018-11-15%2012.29.52.png?raw=1">





안녕하세요. 정원석입니다.



오늘은 강화학습의 일반적인 정의와 학습 알고리즘에 대해서 전체적으로 알아보도록 하겠습니다. 



<img src="https://www.dropbox.com/s/d9hcv8saxkq8lgc/Screenshot%202018-11-15%2012.42.55.png?raw=1">





오늘 다룰 내용은, 

1. 강화학습 알고리즘에 근간이 되는 Markov decision process 의 정의

2. 강화학습에서 어떠한 것을 최적화할 것인지에 대한 문제정의 
3. 강화학습 알고리즘의 전체적인 개요 (세가지)

입니다. 



<img src="https://www.dropbox.com/s/mh81vlyg3ph7tth/Screenshot%202018-11-15%2012.43.14.png?raw=1">



먼저 정의 부터 알아보도록 하겠습니다. 



지난시간에 imitation learning에서 순차적인 의사결정의 요소에 대해서 알아보았습니다. 

 ## 1. Markov Decision Problem 



<img src="https://www.dropbox.com/s/aa7zarn7ya5zsa9/Screenshot%202018-11-15%2012.45.20.png?raw=1">

 여기서 정책(policy)는  관찰값(observation)이 주어졌을때 action의 분포를 출력하며, 

이 policy는 파라메터로서 목표를 달성하기 위한 action을 출력하기 위해 학습되어진다고 하였습니다. 

위의 수식중 세타는 인공신경망의 파레메터를 의미합니다. 



강화학습 알고리즘 중에서는 이 policy를 바로 업데이트 하지 않고 다른 function을 업데이트 하는 value function 혹은 q function 방법 또한 있습니다. 



그렇다면 전문가 데이터가 없을때는 어떻게 Policy를 학습시킬수 있을까요? 

## 2. Reward functions 





<img src="https://www.dropbox.com/s/l97eo8nyam6j2zj/Screenshot%202018-11-17%2012.53.31.png?raw=1">



전문가 데이터를 사용하여 전문가처럼 행동을 하는 Imitation learning 방법이 아닌, 

reward function을 이용한 학습 방법에 대해서 알아보도록 하겠습니다. 

Reward function을 통하여 어떠한 행동(action)이 좋은 행동인지 나쁜 행동인지 알수 있습니다. 

자동차의 예를 들으면, 안전하게 운전을 하는 행동을 할때는 높은 리워드를 받으며 반대로 교통사고를 내는 행동을 했을때는 낮은 리워드 혹은 페널티를 받습니다. 

하지만 여기서 중요한점은, 행동을 선택할때 지금 당장 좋은 행동을 선택하는 것이 아닌 앞으로 일어날 모든 시간들을 고려하여 좋은 행동을 선택한다는 것입니다. 

즉, 현재는 좋은 행동이 아닐지라도 그 행동으로 인해 추후에 더 많은 리워드를 받을수 있다면 

좋지 않은 행동이지만 선택하는 것 입니다. 





위에서 말한대로 우리는 reward function을 사용하여 Markov decision process 를 통하여 높은 리워드를 받기 위해서 행동을 선택한다고 하였는데요. 

 Markov decision process의 정의와 요소에 대해서 더 자세히 알아보도록 하겠습니다. 



## 3. Markov chain 



<img src="https://www.dropbox.com/s/t8q22gtq64a0tau/Screenshot%202018-11-18%2000.26.38.png?raw=1">



위 사진에서의 Andrey Markov가 markov decision process를 만들진 않았지만, 

Markov chains으로 잘 알려져 있습니다. 



Markov chain은 temporal process 를 위한 아주 간단한 모델입니다. 
Markov chain이 control temporal process가 아니라 그냥 temporal process인 이유는 

Markov chain이 state 정보만 포함하기 때문입니다. 



<img src="https://www.dropbox.com/s/nrzrltcbcb3ruvc/Screenshot%202018-11-18%2011.36.37.png?raw=1">



Markov chain은 두 Object인 state space와 transition probability를 이용하여 정의할 수 있습니다



$$M=\{S,T\}$$

$S $는 State space이며 $T$ 는 tranistion operator 입니다.

Transition operation은  주어진  $S_t$에서 어떠한 특정한 State로 넘어갈 조건부 확률입니다.

$\mu_{t,i}$ 는 time step $t$에 state가 $i$일 확률로 정의한다면, 

$\rightarrow_{\mu_t}$ vector $\mu_t$는 time step $t$의 state 확률의 vector 입니다. 

예를 들어 state가 다섯개 있다면,  $\mu_t$는 다섯개의 숫자가 있는 vector 입니다. 

그리고 또한 



state $j$가 주어졌을때 state가 $i$ 가 될 확률 메트릭스 $T_{i,j}$를 생성한다면, 

$\rightarrow_{\mu_t+1} =  T_{\rightarrow{\mu_t} } $라고 정의할 수 있습니다. 

또한, 이러한 Markov chain을 그래픽으로 아주 심플하게 표현할수도 있습니다 :) 

 이 Markov chain은 markov property를 충족하기 때문에 전의 state ( $s_{t-1}$ )과는 독립적입니다. 

## 4. Markov decision process



<img src="https://www.dropbox.com/s/ia8iam0kd6ms02z/Screenshot%202018-11-20%2011.47.43.png?raw=1">



Markov decisio process는 Markov chain에서 행동 (action)과 보상(reward)이 추가됩니다. 

이 Markov decision process의 많은 부분이 Richard bellman이 만든 다이나믹 프로그래밍(dynamic programming) 을 기반으로 하였습니다. 

Markov decisio process는 행동 스페이스 (action space)와 보상함수 (reward function)가 추가됩니다. 



<img src="https://www.dropbox.com/s/8lp1vm516wh8jp1/Screenshot%202018-11-20%2011.48.48.png?raw=1">

$\mu_{t,j}$ 는 마찬가지로 time step $t$에 state가 $j$일 확률입니다. 

여기서 $\xi_{t,k}$ 를 하나 더 정의하는데 이것은 time step $t$에서 행동이 k 일 확률 입니다. 

그렇다면 Transition operation $T_{i,j,k}$은 주어진 $S_t$가 j 이고  $a_t$ 가 k 일때 다음 state가 i 일때의 확률입니다. 



$\mu_{t,i}$은 $\mu_{t,j}$  $\xi_{t,k}$ $T_{i,j,k}$ 을 모두 고려한 확률 일 것입니다. 



## 5. Partial Obseved Markov decision process 

<img src="https://www.dropbox.com/s/eilcxl7j7jz3tfg/Screenshot%202018-11-20%2011.51.52.png?raw=1">



 지금까지 Markov chain과 Markov decision process에 대하여 알아보았습니다. 

보통 우리가 풀어야할 문제들은 Markov chain과 Markov decision process보다 Partial observed Markov decision process인 케이스가 대부분 입니다. 

PoMDP 에서는 Markov decision process에서 oberservation과 Emission을 추가해 줍니다. 

예를들어 state가 좌표, 속도 등이고 action이 브레이크, 가속패달 이라면 observation space란 카메라에서부터 받는 이미지 입니다. 

그리고 emission 확률이란 state가 $s_t$일때 observation이 $o_t$일 확률 입니다. 

Observation o를 추가하여 그래픽 모델로 표현할 수 있습니다. 



## 6. The goal of reinforcement learning 



<img src="https://www.dropbox.com/s/5hrx7l7fr13q8wx/Screenshot%202018-11-20%2011.52.13.png?raw=1">

MDP, POMDP 이외에도 강화학습의 Objective 함수를 정의할 수 있습니다.  

Policy가 있다면, 이 policy는 인공신경망 또는 value function(가치 함수) 등 무엇이든 될 수 있습니다. 

이 policy가 무엇이든 parameter $\theta$ 로 이루어져 있습니다. 

Policy $\pi$는 state를 입력으로 받고 action을 출력하는 인공신경망이라면, 

$\theta$  는  신경망의 weight(웨이트) 입니다. 

polic의 출력은 $\theta$ 변하며, 각 상황에 맞는 action을 선택하도록  $\theta$ 를 강화학습 알고리즘을 통해 학습을 시킵니다. 

Policy에 의해 출력된 행동으로인하여   Environment (환경) 이 영향을 받고 다음 state를 리턴합니다.

이러한 과정을 통해 state와 action가 순차적으로 발생하면서 trajectory를 생성합니다.    

<img src="https://www.dropbox.com/s/u1v5q7gab6z1coa/Screenshot%202018-11-20%2011.56.26.png?raw=1">



어떠한 trajectory의 확률을 $$p_{\theta}(s_1,a_1,.....,S_T,a_T)$$ 이라고 정의한다면, 

이 확률은 처음 state가 $s_1$ 확률에  

policy가 state $s_t$를 받고 $\theta$  에 의해 aciotn $a_t$ 를 출력할 확률 $$\pi_{\theta}(a_t \mid s_t) $$  

state $s_t$에서 action $a_t$를 선택했을때 state가 $s_{t+1} $ 일 확률   $$p(s_{t+1} \mid s_t,a_t)$$

을 끝날때까지 product해준 확률일 것입니다. 



강화학습에서 objective는 이 trajectory의 분포를 이용하여 정의합니다. 

Objective는 한  trajectory에서 받을 총 reward의 기대값 (expectation value)를 최대화 하는 파라메터 $\theta^*$ 를 찾는것 입니다. 



여기서  기대값이라고 하는 이유는 trajectory 에서의 state와 action은 파라메터 $\theta$에 의한  random variable이기 때문입니다. 



## 7. The anatomy of a reinforcement learning algorithm 



<img src="https://www.dropbox.com/s/kp2dbb78blqn2z5/Screenshot%202018-11-20%2011.58.18.png?raw=1">

 지금부터 나올 강화학습 알고리즘은 거의 대부분 세가지 파트로 나눌수 있습니다. 

1. 첫번째 파트는 Policy를 이용하여 sample을 생성합니다. 세상과 상호작용하며 데이터를 모읍니다. 

2. 이 모은 데이터를 이용하여 return값을 측정합니다. 
3. 이 return ( reward )를 이용하여 $\theta$를 수정합니다. 



예를 들면, 

1. 임의로 세번의 트라젝토리를 생성하고 
2. 그 중 가장 리워드를 많이 받은 트라젝토리 방향으로
3. policy를 업데이트 합니다. 
4. 업데이트 된 policy를 사용하여 다시 트라젝토리를 생성합니다. 
5. 이 과정을 반복하며 리워드를 가장 많이 받는 트라젝토리를 찾기위해 policy를 업데이트 합니다. 



이 구조는 여러가지 object에 사용될수 있는데요. 

유용하게 Object들을 살펴보도록 하겠습니다. 

## 8. Q- function and Value function

강화학습에서 많이 사용되는 Q-fucntion과 Value function에 대해서 알아보도록 하겠습니다. 



#### 8.1  Q-function



<img src="https://www.dropbox.com/s/0tnamwj9t2igyg2/Screenshot%202018-11-20%2011.58.46.png?raw=1">

$Q^{\pi} (s_t, a_t) = \sum_{t'=t}^T E_{\pi_\theta} [r(s_t', a_t') \mid s_t,a_t] $

 

위의 Q-function 식에서 $\pi$ 는 Q- function의 policy 입니다. 

Q-function은 State $s_t$, 어떠한 action $a_t$에서부터 끝날때까지(T) 받을수 있는 총 reward의 기대값을 계산합니다. 



#### 8.2 Value function 

<img src="https://www.dropbox.com/s/evbalj17qm0dysx/Screenshot%202018-11-20%2011.59.48.png?raw=1"> 

$V^{\pi} (s_t) = \sum_{t'=t}^T E_{\pi_\theta} [r(s_t', a_t') \mid s_t] $

  Value function은 Q-function의 state 버전이라고 볼 수 있습니다. 

state $s_t$ 에서부터  끝날때까지(T) 받을수 있는 총 reward의 기대값을 계산합니다. 



이렇게 Q-function과 Value function은 reward의 기대값을 계산하는데요. 

이 value 기반 알고리즘과 마찬가지로 강화학습에는 여러가지 학습 알고리즘이 존재합니다.

이에 대해서 전체적으로 알아보도록 하겠습니다. 



 ## 9. Types of RL algorithms  



<img src="https://www.dropbox.com/s/cvct003z60psnsa/Screenshot%202018-11-20%2012.00.10.png?raw=1">

$$\theta ^* = argmax_{\theta} E_{\tau ~ p_{\theta}(\tau)} [\sum_t r(s_t,a_t)]$$

여기서 모든 강화학습 알고리즘은 reward 의 기대값을 높이기 위한 $\theta^*$ 를 구합니다. 

#### 9.1 Policy gradients 

Policy gradients 방법은 위의 objective의 미분을 직접 계산합니다. 

#### 9.2 Value-based

Value function 혹은 Q-function의 기대값을 측정한뒤 optimal policy를 찾습니다. (argmax 합니다. )

#### 9.3 Actor-critic 

Actor-critic은 9.1,9.2 방법을 함께 사용한 것 입니다. 

Value function 혹은 Q-function의 기대값을 측정한뒤 optimial policy를 찾는것이 아닌 ( argmax하는것이 아닌) 이 값을 이용하여 policygradient 방법에서 더 나은 gradient를 찾습니다.

#### 9.4  Model-based RL 

state와 action이 선택되었을의 다음 state를 예측하는 것을 학습합니다. ( transition model을 학습 )

 <img src="https://www.dropbox.com/s/18z8u5m80y8esey/Screenshot%202018-11-20%2012.00.56.png?raw=1">

이렇게 많은 강화학습 알고리즘이 있는 이유는 

각 알고리즘마다 특징이 있기 때문입니다. 

1. 샘플의 량 

   좋은 policy를 얻기 위해 필요한 샘플량은 얼마나 되는가? 
   알고리즘이 off policy 인가 on policy 인가? 
   off policy : policy를 향상시킬때 sample을 사용하여 향상시킬수 있다. (그 Policy를 사용하여  생성된 샘플을 사용하지 않더라도) - 리플레이 메모리가 그 방법중 하나인것 같다. 

   on policy : policy가 바뀔때마다 새로운 sample을 새로 생성해야 한다. 

   그렇기 때문에 알고리즘이 on policy방법이라면 policy가 수정될때마다 sample을 새로 모아야 하기 때문에

   좋은 policy를 찾기 위해서 많은 샘플이 필요할 것이다. 

2. 하이퍼파라미터의 복잡성

3. 안정성 

   Converge 하는가? 
   그렇다면 무엇으로 Converge할 것인가? 
   항상 Converge하는가 ? 

4. MDP가 stochasstic,  deterministic

5. state space와 action space가 continous한지 discrete한지 

6.  MDP의 trajectory가 한정적인지 무한인지 





이번에는 MDP의 정의, 

1.Markov chain

2.Markov Decision Process

3.Partial Observed Markov Decision Process 

4.The goal of reinforcement learning

5.The anatomy of a reinforcement learning algorithm

6.Q-function and Value function

7.Type of RL algorithms 



에 대하여 알아보았습니다. 

다음시간은 강화학습 알고리즘중 하나인 정책을 바로 업데이트 하는 

policy gradients 알고리즘에 대하여 알아보도록 하겠습니다. 



<img src="https://www.dropbox.com/s/t5udpspzcg0q7yl/Screenshot%202018-11-20%2012.01.22.png?raw=1">



