# Research Goals




### 1. Compare the performance of random search and regularized evolution strategy on NAS. Which strategy has the better results?
  
The first part of the research would be to reproduce the results of NAS-Bench 101, and to extend this approach by adding the genetic algorithm into it.
We will tackle the task in the following steps:

- Apply random search and regularised evolution strategy on the search space of NAS. Train and test our approach on the dataset, e.g. CIFAR-10, and reproduce the results as NAS-Bench-101.
- Additionally, we will apply one more evolution strategy (ES) genetic algorithm to enhance this approach to better compare.
- After performing these experiments, we will have the results of random search, genetic algorithm, and regularised evolution strategy for the comparative study. 


### 2. Can the performance of NAS be estimated based on graph attributes without training the architectures?
  
In the second phase of our research, we enlarge the idea of a comparative study by adding some performance metrics, and we will be focused on getting results without having rigorous training by using several steps, which are followings:


- First, We will create a dataset in which attributes of the graphs will be stored, such as the number of nodes, edges, labels, density, etc. 
- After creating the database, it will be divided into the train and test sets. Then few regressors are trained on that training set and predict the performance of the remaining unseen data, test data.
- Then, we report an R-squared value showing how these data fit each regressor model.


