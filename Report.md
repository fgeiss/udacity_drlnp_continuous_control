# Report 

## Baseline Evaluation
To evaluate the performance of the agent we performed a baseline test at the beginning of the project and run the environment N=100 times with random uniformly distributed actions. Below diagram shows the distribution with its mean and standard deviation from this experiment.

![Baseline 20 arms](reacher_20_arms_baseline.png)

## Learning Algorithm



### Neural Network Architecture

### Hyperparameters

| Hyperparameter       | Value   |
|----------------------|---------|
| BUFFER_SIZE          | 1000000 |
| BATCH_SIZE           | 256     | 
| GAMMA                | 0.99    |       
| TAU                  | 1e-3    |         
| Learning Rate Actor  | 1e-3    |     
| Learning Rate Critic | 1e-3    |    
| WEIGHT_DECAY         | 0.0     | 
| EPSILON_START        | 1.0     |
| EPSILON_DECAY        | 0.9995  |
| EPSILON_END          | 0.00001 |
| LEARN_EVERY          | 20      |
| LEARN_TIMES          | 10      |
| PRIO_ALPHA           | 0.1     |


## Results

![Results](learning_ddqn.png)

We trained the algorithm until an elevated threshold for the moving average score (dark blue line) of 35 was reached after approx. 183 episodes. The original threshold of 30 was exceeded after 143 training episodes. For the sake of interest the graph also shows the mean score per episode take over all 20 robot arms (light blue line) as well as the maximum and minumum scores from indiviual agents (blue shaded area). Interestingly the maximal scores are not exceeding a threshold around 40 althoug the theoretical optimal score is Timessteps * pos. reward = 1000 * 0.1 = 100.

## Ideas on Future Work and Possible Improvements

 - Implement prioritized replay to improve the learning performance (ongoing)
 - 
