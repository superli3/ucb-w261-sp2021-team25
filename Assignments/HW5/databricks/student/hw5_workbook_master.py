# Databricks notebook source
# MAGIC %md
# MAGIC # HW 5 - Page Rank
# MAGIC __`MIDS w261: Machine Learning at Scale | UC Berkeley School of Information | Fall 2018`__
# MAGIC 
# MAGIC In Weeks 8 and 9 you discussed key concepts related to graph based algorithms and implemented SSSP.   
# MAGIC In this final homework assignment you'll implement distributed PageRank using some data from Wikipedia.
# MAGIC By the end of this homework you should be able to:  
# MAGIC * ... __compare/contrast__ adjacency matrices and lists as representations of graphs for parallel computation.
# MAGIC * ... __explain__ the goal of the PageRank algorithm using the concept of an infinite Random Walk.
# MAGIC * ... __define__ a Markov chain including the conditions underwhich it will converge.
# MAGIC * ... __identify__ what modifications must be made to the web graph inorder to leverage Markov Chains.
# MAGIC * ... __implement__ distributed PageRank in Spark.
# MAGIC 
# MAGIC __Please refer to the `README` for homework submission instructions and additional resources.__

# COMMAND ----------

# MAGIC %md
# MAGIC Jeff Li, Sonya Chen, Karthik Srinivasan, Justin Trobec

# COMMAND ----------

# MAGIC %md
# MAGIC # Notebook Set-Up
# MAGIC Before starting your homework run the following cells to confirm your setup.

# COMMAND ----------

# imports
import re
import ast
import time
import numpy as np
import pandas as pd

import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run the next cell to create your directory in dbfs
# MAGIC You do not need to understand this scala snippet. It simply dynamically fetches your user directory name so that any files you write can be saved in your own directory.

# COMMAND ----------

# RUN THIS CELL AS IS
# This code snippet reads the user directory name, and stores is in a python variable.
# Next, it creates a folder inside your home folder, which you will use for files which you save inside this notebook.
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
userhome = 'dbfs:/user/' + username
print(userhome)
hw5_path = userhome + "/HW5/" 
hw5_path_open = '/dbfs' + hw5_path.split(':')[-1] # for use with python open()
dbutils.fs.mkdirs(hw5_path)

# COMMAND ----------

# RUN THIS CELL AS IS. 
tot = 0
DATA_PATH = 'dbfs:/mnt/mids-w261/HW5/'
for item in dbutils.fs.ls(DATA_PATH):
  tot = tot+item.size
tot
# ~4.7GB

# COMMAND ----------

# RUN THIS CELL AS IS. You should see all-pages-indexed-in.txt, all-pages-indexed-out.txt and indices.txt in the results. If you do not see these, please let an Instructor or TA know.
display(dbutils.fs.ls(DATA_PATH))

# COMMAND ----------

# RUN THIS CELL AS IS - A test to make sure your directory is working as expected.
# You should see a result like:
# dbfs:/user/youremail@ischool.berkeley.edu/HW5/test.txt
dbutils.fs.put(hw5_path+'test.txt',"hello world",True)
display(dbutils.fs.ls(hw5_path))


# COMMAND ----------

sc = spark.sparkContext
spark

# COMMAND ----------

# MAGIC %md
# MAGIC # Question 1: Distributed Graph Processing
# MAGIC Chapter 5 from Lin & Dyer gave you a high level introduction to graph algorithms and concernts that come up when trying to perform distributed computations over them. The questions below are designed to make sure you captured the key points from this reading and your async lectures. 
# MAGIC 
# MAGIC ### Q1 Tasks:
# MAGIC 
# MAGIC * __a) short response:__ Give an example of a dataset that would be appropriate to represent as a graph. What are the nodes/edges in this dataset? Is the graph you describe 'directed' or 'undirected'? What would the average "in-degree" of a node mean in the context of your example? 
# MAGIC 
# MAGIC * __b) short response:__ Other than their size/scale, what makes graphs uniquely challenging to work with in the map-reduce paradigm? *(__HINT__: Do not respond in terms of any specific algorithm. Think in terms of the nature of the graph datastructure itself).*
# MAGIC 
# MAGIC * __c) short response:__ Briefly describe Dijskra's algorithm (goal/approach). What specific design component makes this approach hard to parallelize?
# MAGIC 
# MAGIC * __d) short response:__ How does parallel breadth-first-search get around the problem that you identified in part `c`? At what expense?

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q1 Student Answers:
# MAGIC 
# MAGIC > __1a)__ Give an example of a dataset that would be appropriate to represent as a graph. 
# MAGIC > What are the nodes/edges in this dataset? Is the graph you describe 'directed' or 'undirected'? 
# MAGIC > What would the average "in-degree" of a node mean in the context of your example?  
# MAGIC > * For example, the relationship of twitter users would be appropriate to represent as a graph. 
# MAGIC > * In the twitter users example, nodes are users, and their relationship is the edges. 
# MAGIC > * In the twitter example, the edge is directed. The reason is that in twitter, userA can follow userB, but userB does not necessarily follow userA. 
# MAGIC > * Say UserA follows userA, then the edge starts at UserA and has the arrow pointing toward userB. And userA is a follower of UserB in this case. 
# MAGIC > * In-degree represents the number of incoming neighbors, or followers. 
# MAGIC > * The average "in-degree" of a node is the average number of followers a user has. 
# MAGIC 
# MAGIC 
# MAGIC > __1b)__ Other than their size/scale, what makes graphs uniquely challenging to work with in the map-reduce paradigm?
# MAGIC > * It is challenging for map-reduce. The reason is a node could have many neighbors, and thus the results of a mapper might need to arrive at different reducers. 
# MAGIC > * Say if we have 10 mappers, and each mapper produces 10 results, and each of these 10 results need to arrive at different reducers.
# MAGIC > * With 10 mappers, we might have 100 results that need to do a large amount of shuffling and combing before we can do the next step of computation. 
# MAGIC > * And that shuffling and combining is expensive as data traffic is costly. 
# MAGIC > * In addition, for graph computation, we have to do matrix multiplication. And if the matrix is very big, and matrix multiplication is expensive, a single machine might not be able to fit the all of that. 
# MAGIC > * One thing that can make graphs challenging to work with is that there are not necessarily natural partitionings of them. In other words, paths an algorithm may take through the graph might make it impossible to assign some nodes to an individual partition of the dataset.
# MAGIC 
# MAGIC 
# MAGIC > __c)__ Briefly describe Dijskra's algorithm (goal/approach). What specific design component makes this approach hard to parallelize?
# MAGIC >> * Dijstka is the algorithm's goal is to find the shortest path from starting position to the destination/all other nodes. 
# MAGIC >> * The approach Dijkstra take is:
# MAGIC >> * The approach is we always pick the next node (Frontier) that is unvisited and has the shortest distance from the current node (with the info we have up until this point). 
# MAGIC >> * When we are at the next node, we again update the distance of all of its neighboring nodes. 
# MAGIC >> * Then again, we pick the next unvisited node with the shortest distance, and go there. 
# MAGIC >> * We do so until we reach our destination node. 
# MAGIC >> * When we pick which node to go next, we need to know the latest information of all the neighboring node. And the next node will update the information of neighbor nodes. And that update decides which node we go next again.
# MAGIC >> * In Dijkstra's algorithm, we proceed to find the shortest paths by traversing one node at a time. This is done sequentially using a priority queue and works well on a single node computer. However, this does not make it amenable to parallelization.
# MAGIC >> * In Dijkstra, the order of node visits matter! Because that ensures that we use the least cost to find the shortest path.
# MAGIC >> * Thus in this approach, we can ONLY VISIT one node at a time. That means we cannot compute the results for several nodes at the same time. 
# MAGIC 
# MAGIC 
# MAGIC >__d)__ How does parallel breadth-first-search get around the problem that you identified in part `c`? At what expense?
# MAGIC >> * The parallel breadth first search enables us to use a FIFO queue instead of the priority queue as seen in Dijkstra. In parallel BFS, the nodes on the same level can be processed at the same time.
# MAGIC >> * If visiting a node has some cost associated with it, then the parallel breath-first-search will not minimize the complexity of finding the lowest cost. The additional complexity arises when a visited node is put back into the frontier queue. This happens when the aforementioned visited node has a newly computed shorter/smaller cost. This is an expensive operation since it explores all paths.

# COMMAND ----------

# MAGIC %md
# MAGIC # Question 2: Representing Graphs 
# MAGIC 
# MAGIC In class you saw examples of adjacency matrix and adjacency list representations of graphs. These data structures were probably familiar from HW3, though we hadn't before talked about them in the context of graphs. In this question we'll discuss some of the tradeoffs associated with these representations. __`NOTE:`__ We'll use the graph from Figure 5.1 in Lin & Dyer as a toy example. For convenience in the code below we'll label the nodes `A`, `B`, `C`, `D`, and `E` instead of \\(n_1\\), \\(n_2 \\), etc but otherwise you should be able to follow along & check our answers against those in the text.
# MAGIC 
# MAGIC 
# MAGIC <img src="https://github.com/kyleiwaniec/w261_assets/blob/master/images/HW5/Lin-Dyer-graph-Q1.png?raw=true" width=50%>
# MAGIC 
# MAGIC ### Q2 Tasks:
# MAGIC 
# MAGIC * __a) short response:__ Relatively speaking, is the graph you described in Figure 5.1 in Lin & Dyer "sparse" or "dense"?  Explain how sparsity/density impacts the adjacency matrix and adjacency list representations of a graph.
# MAGIC 
# MAGIC * __b) short response:__ Run the provided code to create and plot our toy graph. Is this graph directed or undirected? Explain how the adjacency matrices for directed graphs will differ from those of undirected graphs.
# MAGIC 
# MAGIC * __c) code:__ Fill in the missing code to complete the function `get_adj_matr()`.
# MAGIC 
# MAGIC * __d) code:__ Fill in the missing code to complete the function `get_adj_list()`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q2 Student Answers:
# MAGIC > __a)__ This is relatively dense, compared to a graph like the world-wide-web. Sparse graphs will have a much smaller adjacency list representation compared to their matrix representation, as the matrix has to have an entry even for non-existent edges.
# MAGIC 
# MAGIC > __b)__ The graph is directed. An adjacency matrix for an undirected graph will be symetric, while a directed graph's adjacency matrix will not.

# COMMAND ----------

# part a - a graph is just a list of nodes and edges (RUN THIS CELL AS IS)
TOY_GRAPH = {'nodes':['A', 'B', 'C', 'D', 'E'],
             'edges':[('A', 'B'), ('A', 'D'), ('B', 'C'), ('B', 'E'), ('C', 'D'), 
                      ('D', 'E'), ('E', 'A'),('E', 'B'), ('E', 'C')]}

# COMMAND ----------

# part a - simple visualization of our toy graph using nx (RUN THIS CELL AS IS)
G = nx.DiGraph()
G.add_nodes_from(TOY_GRAPH['nodes'])
G.add_edges_from(TOY_GRAPH['edges'])
display(nx.draw(G, pos=nx.circular_layout(G), with_labels=True, alpha = 0.5))

# COMMAND ----------

# part c - adjacency matrix function
def get_adj_matr(graph):
    """
    Function to create an adjacency matrix representation of a graph.
    arg:
        graph - (dict) of 'nodes' : [], 'edges' : []
    returns:
        pd.DataFrame with entry i,j representing an edge from node i to node j
    """
    n = len(graph['nodes'])
    adj_matr = pd.DataFrame(0, columns = graph['nodes'], index = graph['nodes'])
    ############### YOUR CODE HERE ##################
    for node1, node2 in graph['edges']:
      adj_matr[node2][node1] = 1.0
    ############### (END) YOUR CODE #################
    return adj_matr

# COMMAND ----------

# part c - take a look (RUN THIS CELL AS IS)
TOY_ADJ_MATR = get_adj_matr(TOY_GRAPH)
print(TOY_ADJ_MATR)

# COMMAND ----------

# part d - adjacency list function
def get_adj_list(graph):
    """
    Function to create an adjacency list representation of a graph.
    arg:
        graph - (dict) of 'nodes' : [], 'edges' : []
    returns:
        dictionary of the form {node : [list of edges]}
    """
    adj_list = {node: [] for node in graph['nodes']}
    ############### YOUR CODE HERE ##################
    for node1, node2 in graph['edges']:
      adj_list[node1].append(node2)
    ############### (END) YOUR CODE #################
    return adj_list

# COMMAND ----------

# part d - take a look (RUN THIS CELL AS IS)
TOY_ADJ_LIST = get_adj_list(TOY_GRAPH)
print(TOY_ADJ_LIST)

# COMMAND ----------

# MAGIC %md
# MAGIC # Question 3: Markov Chains and Random Walks
# MAGIC 
# MAGIC As you know from your readings and in class discussions, the PageRank algorithm takes advantage of the machinery of Markov Chains to compute the relative importance of a webpage using the hyperlink structure of the web (we'll refer to this as the 'web-graph'). A Markov Chain is a discrete-time stochastic process. The stochastic matrix has a principal left eigen vector corresponding to its largest eigen value which is one. A Markov chain's probability distribution over its states may be viewed as a probability vector. This steady state probability for a state is the PageRank of the corresponding webpage. In this question we'll briefly discuss a few concepts that are key to understanding the math behind PageRank. 
# MAGIC 
# MAGIC ### Q3 Tasks:
# MAGIC 
# MAGIC * __a) short response:__ It is common to explain PageRank using the analogy of a web surfer who clicks on links at random ad infinitum. In the context of this hypothetical infinite random walk, what does the PageRank metric measure/represent?
# MAGIC 
# MAGIC * __b) short response:__ What is the "Markov Property" and what does it mean in the context of PageRank?
# MAGIC 
# MAGIC * __c) short response:__ A Markov chain consists of \\(n\\) states plus an \\(n\times n \\) transition probability matrix. In the context of PageRank & a random walk over the WebGraph what are the $n$ states? what implications does this have about the size of the transition matrix?
# MAGIC 
# MAGIC * __d) code + short response:__ What is a "right stochastic matrix"? Fill in the code below to compute the transition matrix for the toy graph from question 2. [__`HINT:`__ _It should be right stochastic. Using numpy this calculation can be done in one line of code._]
# MAGIC 
# MAGIC * __e) code + short response:__ To compute the stable state distribution (i.e. PageRank) of a "nice" graph we can apply the power iteration method - repeatedly multiplying the transition matrix by itself, until the values no longer change. Apply this strategy to your transition matrix from `part d` to find the PageRank for each of the pages in your toy graph. Your code should print the results of each iteration. How many iterations does it take to converge? Which node is most 'central' (i.e. highest ranked)? Does this match your intuition? 
# MAGIC     * __`NOTE 1:`__ _this is a naive approach, we'll unpack what it means to be "nice" in the next question_.
# MAGIC     * __`NOTE 2:`__ _no need to implement a stopping criteria, visual inspection should suffice_.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q3 Student Answers:
# MAGIC > __a)__ It is common to explain PageRank using the analogy of a web surfer who clicks on links at random ad infinitum. 
# MAGIC Q: In the context of this hypothetical infinite random walk, what does the PageRank metric measure/represent?
# MAGIC 
# MAGIC > * The PageRank link analysis algorithm "measures" the relative number of visits the "infinite" websurfer will spend on each page in the WebGraph.
# MAGIC 
# MAGIC > __b)__ What is the "Markov Property" and what does it mean in the context of PageRank?
# MAGIC > * The Markov property is memorylessness, meaning the evolution of the Markov process in the future depends only on the present state and does not depend on past history. Markov processes provide a principled approach to calculating each page's PageRank. 
# MAGIC > * PageRank is the steady-state probability distribution of the Markov process underlying the random-surfer navigation model.
# MAGIC 
# MAGIC > __c)__ A Markov chain consists of \\(n\\) states plus an \\( n\times n\\) transition probability matrix. In the context of PageRank & a random walk over the WebGraph what are the \\(n\\) states? what implications does this have about the size of the transition matrix?
# MAGIC > * This implies that the size of the transition matrix will be n x n matrix, where n is the number of nodes, or total number of pages in the whole graph.
# MAGIC 
# MAGIC > __d)__ What is a "right stochastic matrix"?
# MAGIC > * Right stochastic matrix is a non-negative real number square matrix, with each row summing to 1. In the context of Pagerank, it represents transition probabilities in a Markov Chain.
# MAGIC 
# MAGIC > __e)__ To compute the stable state distribution (i.e. PageRank) of a "nice" graph we can apply the power iteration method - repeatedly multiplying the transition matrix by itself, 
# MAGIC until the values no longer change. Apply this strategy to your transition matrix from `part d` to find the PageRank for each of the pages in your toy graph. 
# MAGIC 
# MAGIC > Q: How many iterations does it take to converge? 
# MAGIC > * It takes 50-60 iterations to converge, depending on the stopping criteria.
# MAGIC 
# MAGIC > Q: Which node is most 'central' (i.e. highest ranked)? Does this match your intuition? 
# MAGIC > * Node E is the most central node (or highest ranking node). It is close to my intuition.

# COMMAND ----------

# part d - recall what the adjacency matrix looked like (RUN THIS CELL AS IS)
TOY_ADJ_MATR

# COMMAND ----------

# part d - use TOY_ADJ_MATR to create a right stochastic transition matrix for this graph
################ YOUR CODE HERE #################
def create_transition_matrix(adj_matrix):
    return adj_matrix.mul((1.0 / adj_matrix.sum(axis=1)), axis=0).fillna(0)
    
transition_matrix = create_transition_matrix(TOY_ADJ_MATR) # replace with your code
################ (END) YOUR CODE #################
print(transition_matrix)

# COMMAND ----------

# part e - compute the steady state using the transition matrix 
def power_iteration(xInit, tMatrix, nIter, verbose = True):
    """
    Function to perform the specified number of power iteration steps to 
    compute the steady state probability distribution for the given
    transition matrix.
    
    Args:
        xInit     - (n x 1 array) representing inial state
        tMatrix  - (n x n array) transition probabilities
        nIter     - (int) number of iterations
    Returns:
        state_vector - (n x 1 array) representing probability 
                        distribution over states after nSteps.
    
    NOTE: if the 'verbose' flag is on, your function should print the step
    number and the current matrix at each iteration.
    """
    state_vector = None
    ################ YOUR CODE HERE #################
    state_vector = xInit/np.sum(xInit)

    for i in range(nIter):
        print(i)
        new_state_vector = np.dot(np.transpose(tMatrix.to_numpy()),state_vector)
        state_vector = new_state_vector
        if verbose:
            for i in range(tMatrix.shape[0]):
                print('Node {}: {}'.format(tMatrix.index[i], new_state_vector[i]))
            print(np.sum(new_state_vector))
    for i in range(tMatrix.shape[0]):
        print('Node {}: {}'.format(tMatrix.index[i], new_state_vector[i]))
    ################ (END) YOUR CODE #################
    return state_vector

# COMMAND ----------

# part e - run 10 steps of the power_iteration (RUN THIS CELL AS IS)
xInit = np.array([1.0, 0.0, 0, 0, 0]) # note that this initial state will not affect the convergence states
states = power_iteration(xInit, transition_matrix, 100, verbose = True)

# COMMAND ----------

# MAGIC %md
# MAGIC __`Expected Output for part e:`__  
# MAGIC >Steady State Probabilities:
# MAGIC ```
# MAGIC Node A: 0.10526316  
# MAGIC Node B: 0.15789474  
# MAGIC Node C: 0.18421053  
# MAGIC Node D: 0.23684211  
# MAGIC Node E: 0.31578947  
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC # Question 4: Page Rank Theory
# MAGIC 
# MAGIC Seems easy right? Unfortunately applying this power iteration method directly to the web-graph actually runs into a few problems. In this question we'll tease apart what we meant by a 'nice graph' in Question 3 and highlight key modifications we'll have to make to the web-graph when performing PageRank. To start, we'll look at what goes wrong when we try to repeat our strategy from question 3 on a 'not nice' graph.
# MAGIC 
# MAGIC __`Additional References:`__ http://pi.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture3/lecture3.html
# MAGIC 
# MAGIC ### Q4 Tasks:
# MAGIC 
# MAGIC * __a) code + short response:__ Run the provided code to create and plot our 'not nice' graph. Fill in the missing code to compute its transition matrix & run the power iteration method from question 3. What is wrong with what you see? [__`HINT:`__ _there is a visible underlying reason that it isn't converging... try adding up the probabilities in the state vector after each iteration._]
# MAGIC 
# MAGIC * __b) short response:__  Identify the dangling node in this 'not nice' graph and explain how this node causes the problem you described in 'a'. How could we modify the transition matrix after each iteration to prevent this problem?
# MAGIC 
# MAGIC * __c) short response:__ What does it mean for a graph to be irreducible? Is the webgraph naturally irreducible? Explain your reasoning briefly.
# MAGIC 
# MAGIC * __d) short response:__ What does it mean for a graph to be aperiodic? Is the webgraph naturally aperiodic? Explain your reasoning briefly.
# MAGIC 
# MAGIC * __e) short response:__ What modification to the webgraph does PageRank make in order to guarantee aperiodicity and irreducibility? Interpret this modification in terms of our random surfer analogy.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q4 Student Answers:
# MAGIC 
# MAGIC > __a)__ Run the provided code to create and plot our 'not nice' graph. Fill in the missing code to compute its transition matrix & run the power iteration method from question 3. What is wrong with what you see?
# MAGIC > * The states of this graph do not converge to a stable state after power iteration. This is because the transition matrix does not qualify as a stochastic matrix.
# MAGIC 
# MAGIC > __b.1)__ Identify the dangling node in this 'not nice' graph 
# MAGIC and explain how this node causes the problem you described in 'a'. 
# MAGIC > * The dangling node here is Node-E. 
# MAGIC > * Node-E has two incoming routes, but NO outward route. 
# MAGIC 
# MAGIC > __b2)__ How could we modify the transition matrix after each iteration to prevent this problem?
# MAGIC > * We would modify the transition matrix to incorporate the idea of teleporting to solve this problem. 
# MAGIC > * $$NewTransitionMatrix=(1-\alpha) \* \text{transition matrix} + \alpha \* \text{teleporting matrix}$$
# MAGIC > * Thus with this new stochastic transition matrix, it will solve both the dangling node and periodic problem. 
# MAGIC > * This is important because only a well-behaved graph (irreducible and aperiodic) can converge. 
# MAGIC 
# MAGIC > __c)__ What does it mean for a graph to be irreducible? Is the webgraph naturally irreducible? Explain your reasoning briefly.
# MAGIC > * A graph is irreducible if there is a route from every node to every other node. 
# MAGIC > * The webgraph (Toy2_Graph) is NOT irreducible because you cannot travel to another node from Node E.
# MAGIC 
# MAGIC > __d)__ What does it mean for a graph to be aperiodic? Is the webgraph naturally aperiodic? Explain your reasoning briefly.
# MAGIC > * Aperiodic means that the period, or greatest common divisor (GCD), of all cycle lengths is 1.
# MAGIC > * This Toy2 webgraph is not naturally aperiodic because not even one node can arrive at itself. All the nodes in the  Toy2 webgraph has to at least wait more than 1 iteration to arrive at it self. 
# MAGIC 
# MAGIC > __e)__ What modification to the webgraph does PageRank make in order to guarantee aperiodicity and irreducibility? Interpret this modification in terms of our random surfer analogy.
# MAGIC > * We would modify the transition matrix to incorporate the idea of teleporting to solve this problem. In the context of the random surfer, it means that there is a chance that the surfer could reach any other page in our webgraph from the current page.
# MAGIC > * $$NewTransitionMatrix=(1-\alpha) \* \text{transition matrix} + \alpha \* \text{teleporting matrix}$$
# MAGIC > * Thus with this new stochastic transition matrix, it will solve both the dangling node and periodic problem. 
# MAGIC > * This is important because only a well-behaved graph (irreducible and aperiodic) can converge.

# COMMAND ----------

# part a - run this code to create a second toy graph (RUN THIS CELL AS IS)
TOY2_GRAPH = {'nodes':['A', 'B', 'C', 'D', 'E'],
              'edges':[('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'D'), 
                       ('B', 'E'), ('C', 'A'), ('C', 'E'), ('D', 'B')]}

# COMMAND ----------

# part a - simple visualization of our test graph using nx (RUN THIS CELL AS IS)
G = nx.DiGraph()
G.add_nodes_from(TOY2_GRAPH['nodes'])
G.add_edges_from(TOY2_GRAPH['edges'])
display(nx.draw(G, pos=nx.circular_layout(G), with_labels=True, alpha = 0.5))

# COMMAND ----------

# part a - run 10 steps of the power iteration method here
# HINT: feel free to use the functions get_adj_matr() and power_iteration() you wrote above
################ YOUR CODE HERE #################
TOY_ADJ_MATR2 = get_adj_matr(TOY2_GRAPH)
print(TOY_ADJ_MATR2)
transition_matrix2 = create_transition_matrix(TOY_ADJ_MATR2) # replace with your code
print(transition_matrix2)
xInit = np.array([1.0, 0.0, 0.0, 0.0, 0.0]) # note that this initial state will not affect the convergence states
states = power_iteration(xInit, transition_matrix2, 10, verbose = True)

################ (END) YOUR CODE #################

# COMMAND ----------

# MAGIC %md
# MAGIC # About the Data
# MAGIC The main dataset for this data consists of a subset of a 500GB dataset released by AWS in 2009. The data includes the source and metadata for all of the Wikimedia wikis. You can read more here: 
# MAGIC > https://aws.amazon.com/blogs/aws/new-public-data-set-wikipedia-xml-data. 
# MAGIC 
# MAGIC As in previous homeworks we'll be using a 2GB subset of this data, which is available to you in this dropbox folder: 
# MAGIC > https://www.dropbox.com/sh/2c0k5adwz36lkcw/AAAAKsjQfF9uHfv-X9mCqr9wa?dl=0. 
# MAGIC 
# MAGIC Use the cells below to download the wikipedia data and a test file for use in developing your PageRank implementation(note that we'll use the 'indexed out' version of the graph) and to take a look at the files.

# COMMAND ----------

dbutils.fs.ls(DATA_PATH)

# COMMAND ----------

# open test_graph.txt file to see format (RUN THIS CELL AS IS)
with open('/dbfs/mnt/mids-w261/HW5/test_graph.txt', "r") as f_read:
  for line in f_read:
    print(line)

# COMMAND ----------

# load the data into Spark RDDs for convenience of use later (RUN THIS CELL AS IS)
DATA_PATH = 'dbfs:/mnt/mids-w261/HW5/'
testRDD = sc.textFile(DATA_PATH +'test_graph.txt')
indexRDD = sc.textFile(DATA_PATH + '/indices.txt')
wikiRDD = sc.textFile(DATA_PATH + '/all-pages-indexed-out.txt')

# COMMAND ----------

# display testRDD (RUN THIS CELL AS IS)
testRDD.take(10)

# COMMAND ----------

# display indexRDD (RUN THIS CELL AS IS)
indexRDD.take(10)

# COMMAND ----------

# display wikiRDD (RUN THIS CELL AS IS)
wikiRDD.take(10)

# COMMAND ----------

# MAGIC %md
# MAGIC # Question 5: EDA part 1 (number of nodes)
# MAGIC 
# MAGIC As usual, before we dive in to the main analysis, we'll peform some exploratory data anlysis to understand our dataset. Please use the test graph that you downloaded to test all your code before running the full dataset.
# MAGIC 
# MAGIC ### Q5 Tasks:
# MAGIC * __a) short response:__ In what format is the raw data? What does the first value represent? What does the second part of each line represent? [__`HINT:`__ _no need to go digging here, just visually inspect the outputs of the head commands that we ran after loading the data above._]
# MAGIC 
# MAGIC * __b) code + short response:__ Run the provided bash command to count the number of records in the raw dataset. Explain why this is _not_ the same as the number of total nodes in the graph.
# MAGIC 
# MAGIC * __c) code:__ In the space provided below write a Spark job to count the _total number_ of nodes in this graph. 
# MAGIC 
# MAGIC * __d) short response:__ How many dangling nodes are there in this wikipedia graph? [__`HINT:`__ _you should not need any code to answer this question._]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q5 Student Answers:
# MAGIC * __a)__ In what format is the raw data? What does the first value represent? What does the second part of each line represent? [__`HINT:`__ _no need to go digging here, just visually inspect the outputs of the head commands that we ran after loading the data above._]
# MAGIC > * Each line looks like a key value pair with a tab separated  value, where the key is the node id, and the value is an adjacency list, where each value is outgoing node and the number of edges to the outgoing node.
# MAGIC > * The raw data is a list of string, where inside each string embedded "node" and its a dictionary, which contains the corresponding outward neighbors from the node (the first value of the string). 
# MAGIC > * The first value corresponding to a particular node. 
# MAGIC > * The second part of each line represents the outgoing nodes from the corresponding node (which is the key of each line). Simple answer is the the second part of each line is the outgoing neighbors of the node (which is the key of each line). 
# MAGIC 
# MAGIC * __b)__ Run the provided bash command to count the number of records in the raw dataset. Explain why this is _not_ the same as the number of total nodes in the graph.
# MAGIC > * The number of records is not the same as the number of nodes in the graph. The reason is that each record in the list represents a node and its out-ward neighbors. If say a node has no outgoing edges, then this node will not be represented in a record in the list. Yet, this node might have the incoming edges. 
# MAGIC 
# MAGIC * __d)__ How many dangling nodes are there in this wikipedia graph? [__`HINT:`__ _you should not need any code to answer this question._]
# MAGIC > * Dangling node (nodes that has not out-going edges)
# MAGIC > * num_node_nodes - num_of_records = 15192277 - 5781290 = 9410987 dangling node

# COMMAND ----------

# part b - count the number of records in the raw data (RUN THIS CELL AS IS)
# 5781290
print(wikiRDD.count())

# COMMAND ----------

# part c - write your Spark job here (compute total number of nodes)
def count_nodes(dataRDD):
    """
    Spark job to count the total number of nodes.
    Returns: integer count 
    """    
    ############## YOUR CODE HERE ###############
    
#     temp = dataRDD.flatMap(lambda x: [(k,1) for k,v in ast.literal_eval(x.split('\t')[1]).items()]+ [(x.split('\t')[0],1)]) \
#                   .reduceByKey(lambda x,y : x + y) \
#                   .cache()         
    temp = dataRDD.flatMap(lambda x: [k for k,v in ast.literal_eval(x.split('\t')[1]).items()]+ [x.split('\t')[0]]).distinct().cache()
    ############## (END) YOUR CODE ###############   
    return temp.count()

# COMMAND ----------

# part c - run your counting job on the test file (RUN THIS CELL AS IS)
start = time.time()
tot = count_nodes(testRDD)
print(f'... completed job in {time.time() - start} seconds.')
print(f'Total Nodes: {tot}')

# COMMAND ----------

# part c - run your counting job on the full file (RUN THIS CELL AS IS)
start = time.time()
tot = count_nodes(wikiRDD)
print(f'... completed job in {time.time() - start} seconds.')
print(f'Total Nodes: {tot}')

# COMMAND ----------

# MAGIC %md
# MAGIC # Question 6 - EDA part 2 (out-degree distribution)
# MAGIC 
# MAGIC As you've seen in previous homeworks the computational complexity of an implementation depends not only on the number of records in the original dataset but also on the number of records we create and shuffle in our intermediate representation of the data. The number of intermediate records required to update PageRank is related to the number of edges in the graph. In this question you'll compute the average number of hyperlinks on each page in this data and visualize a distribution for these counts (the out-degree of the nodes). 
# MAGIC 
# MAGIC ### Q6 Tasks:
# MAGIC * __a) code:__ In the space provided below write a Spark job to stream over the data and compute all of the following information:
# MAGIC  * count the out-degree of each non-dangling node and return the names of the top 10 pages with the most hyperlinks
# MAGIC  * find the average out-degree for all non-dangling nodes in the graph
# MAGIC  * take a 1000 point sample of these out-degree counts and plot a histogram of the result. 
# MAGIC  
# MAGIC  
# MAGIC * __b) short response:__ In the context of the PageRank algorithm, how is information about a node's out degree used?
# MAGIC 
# MAGIC * __c) short response:__ What does it mean if a node's out-degree is 0? In PageRank how will we handle these nodes differently than others?
# MAGIC  
# MAGIC __`NOTE:`__ Please observe scalability best practices in the design of your code & comment your work clearly. You will be graded on both the clarity and the design.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q6 Student Answers:
# MAGIC 
# MAGIC > __b)__ In the context of the PageRank algorithm, how is information about a node's out degree used?
# MAGIC > *  It's used to calculate the probabability of reaching a particular page relative to all of the other outgoing pages.
# MAGIC 
# MAGIC > __c1)__ What does it mean if a node's out-degree is 0? 
# MAGIC > * If a node's out-degree is 0, it means that this node is a dangling node. You cannot go to any other node from this node.
# MAGIC 
# MAGIC > __c2)__ In PageRank how will we handle these nodes differently than others?
# MAGIC > * In PageRank, we redistribute the weight of these dangling nodes evenly across all the nodes. 
# MAGIC > * The formula is $$ \left( \frac{m}{|G|} \right) $$ , where m is the weight of the dangling nodes, and $$( |G| )$$ is the number of all of the nodes.
# MAGIC > * Whereas non-dangling nodes (just like what they should do previously), uniformly distribute its weight arcoss all of the other nodes due to the teleportation factor.

# COMMAND ----------

# part a - write your Spark job here (compute average in-degree, etc)
def count_degree(dataRDD, n):
    """
    Function to analyze out-degree of nodes in a a graph.
    Returns: 
        top  - (list of 10 tuples) nodes with most edges
        avgDegree - (float) average out-degree for non-dangling nodes
        sampledCounts - (list of integers) out-degree for n randomly sampled non-dangling nodes
    """
    # helper func
    def parse(line):
        node, edges = line.split('\t')
        return (node, ast.literal_eval(edges))
    
    ############## YOUR CODE HERE ###############
    
    top, avgDegree, sampledCounts = None, None, None
    tempRDD = dataRDD.map(lambda x: parse(x)).map(lambda x: (x[0], len(x[1].keys()))).cache()
    top = tempRDD.takeOrdered(n, key=lambda x: -x[1])
    tempRDD2 = tempRDD.map(lambda x: x[1]).cache()
    sampledCounts = tempRDD2.takeSample(False, n , 0)
    avgDegree = tempRDD2.mean()    
    
    ############## (END) YOUR CODE ###############
    
    return top, avgDegree, sampledCounts

# COMMAND ----------

# part a - run your job on the test file (RUN THIS CELL AS IS)
start = time.time()
test_results = count_degree(testRDD,10)
print(f"... completed job in {time.time() - start} seconds")
print("Average out-degree: ", test_results[1])
print("Top 10 nodes (by out-degree:)\n", test_results[0])
print("Top 10 nodes (by out-degree:)\n", test_results[2])

# COMMAND ----------

# part a - plot results from test file (RUN THIS CELL AS IS)
plt.hist(test_results[2], bins=10)
plt.title("Distribution of Out-Degree")
display(plt.show())

# COMMAND ----------

# part a - run your job on the full file (RUN THIS CELL AS IS)
start = time.time()
full_results = count_degree(wikiRDD,1000)

print(f"... completed job in {time.time() - start} seconds")
print("Average out-degree: ", full_results[1])
print("Top 10 nodes (by out-degree:)\n", full_results[0])

# COMMAND ----------

# part a - plot results from full file (RUN THIS CELL AS IS)
plt.hist(full_results[2], bins=50)
plt.title("Distribution of Out-Degree")
display(plt.show())

# COMMAND ----------

# MAGIC %md
# MAGIC # Question 7 - PageRank part 1 (Initialize the Graph)
# MAGIC 
# MAGIC One of the challenges of performing distributed graph computation is that you must pass the entire graph structure through each iteration of your algorithm. As usual, we seek to design our computation so that as much work as possible can be done using the contents of a single record. In the case of PageRank, we'll need each record to include a node, its list of neighbors and its (current) rank. In this question you'll initialize the graph by creating a record for each dangling node and by setting the initial rank to 1/N for all nodes. 
# MAGIC 
# MAGIC __`NOTE:`__ Your solution should _not_ hard code \\(N\\).
# MAGIC 
# MAGIC ### Q7 Tasks:
# MAGIC * __a) short response:__ What is \\(N\\)? Use the analogy of the infinite random web-surfer to explain why we'll initialize each node's rank to \\(\frac{1}{N}\\). (i.e. what is the probabilistic interpretation of this choice?)
# MAGIC 
# MAGIC * __b) short response:__ Will it be more efficient to compute \\(N\\) before initializing records for each dangling node or after? Explain your reasoning.
# MAGIC 
# MAGIC * __c) code:__ Fill in the missing code below to create a Spark job that:
# MAGIC   * parses each input record
# MAGIC   * creates a new record for any dangling nodes and sets it list of neighbors to be an empty set
# MAGIC   * initializes a rank of 1/N for each node
# MAGIC   * returns a pair RDD with records in the format specified by the docstring
# MAGIC 
# MAGIC 
# MAGIC * __d) code:__ Run the provided code to confirm that your job in `part a` has a record for each node and that your should records match the format specified in the docstring and the count should match what you computed in question 5. [__`TIP:`__ _you might want to take a moment to write out what the expected output should be fore the test graph, this will help you know your code works as expected_]
# MAGIC  
# MAGIC __`NOTE:`__ Please observe scalability best practices in the design of your code & comment your work clearly. You will be graded on both the clarity and the design.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q7 Student Answers:
# MAGIC 
# MAGIC > __a)__ What is \\(N\\)? Use the analogy of the infinite random web-surfer to explain. why we'll initialize each node's rank to \\(\frac{1}{N}\\). (i.e. what is the probabilistic interpretation of this choice?)
# MAGIC > * In the example of the web-server, N is the number of websites.
# MAGIC > * The probabilistic interpretation of this choice is that we have an equal chance of start surfing at any of the webpage. 
# MAGIC > * In fact, no matter which website we start to web-server, say if we do infinite random web-server, eventually the probability that we land at each website will converge to a steady state if the graph is a "well-behaved" graph (an irreducible and aperiodic).
# MAGIC > * Converge to a steady state means that $$ x_{i+1} = x_{i} * P  $$, which means that the result of your i+1 iteration will be the same as your \\( i_{th} \\) iteration.  
# MAGIC 
# MAGIC > __b)__ Will it be more efficient to compute \\(N\\) before initializing records for each dangling node or after? Explain your reasoning.
# MAGIC > * It will be more efficient to compute \\(N\\) before initializing records for each dangling node.
# MAGIC > * The reason is we will need to know the total number of nodes because we need to redistribute the weights of the dangling nodes uniformly across all the nodes at the 2nd Map Job of each iteration.

# COMMAND ----------

# part c - job to initialize the graph (RUN THIS CELL AS IS)
def initGraph(dataRDD):
    """
    Spark job to read in the raw data and initialize an 
    adjacency list representation with a record for each
    node (including dangling nodes).
    
    Returns: 
        graphRDD -  a pair RDD of (node_id , (score, edges))
        
    NOTE: The score should be a float, but you may want to be 
    strategic about how format the edges... there are a few 
    options that can work. Make sure that whatever you choose
    is sufficient for Question 8 where you'll run PageRank.
    """
    ############## YOUR CODE HERE ###############

    # write any helper functions here
    def get_adj_list(line):
        node, edges = line.split('\t')
        edge_list = [(k,v) for k,v in ast.literal_eval(edges).items()]
        yield (node, edge_list)
        for edge_node in edge_list:
          yield(edge_node[0], [])
    
    # write your main Spark code here
    graphRDD = dataRDD.flatMap(lambda x: get_adj_list(x))\
                       .reduceByKey(lambda x,y: x + y).cache()
    N = graphRDD.count()
    graphRDD = graphRDD.map(lambda x: (x[0], (1.0/N, list(set(x[1])))))
    ############## (END) YOUR CODE ##############
    
    return graphRDD

# COMMAND ----------

# part c - run your Spark job on the test graph (RUN THIS CELL AS IS)
start = time.time()
testGraph = initGraph(testRDD).collect()
print(f'... test graph initialized in {time.time() - start} seconds.')
testGraph

# COMMAND ----------

# part c - run your code on the main graph (RUN THIS CELL AS IS)
start = time.time()
wikiGraphRDD = initGraph(wikiRDD)
print(f'... full graph initialized in {time.time() - start} seconds')

# COMMAND ----------

# part c - confirm record format and count (RUN THIS CELL AS IS)
start = time.time()
print(f'Total number of records: {wikiGraphRDD.count()}')
print(f'First record: {wikiGraphRDD.take(1)}')
print(f'... initialization continued: {time.time() - start} seconds')

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Question 8 - PageRank part 2 (Iterate until convergence)
# MAGIC 
# MAGIC Finally we're ready to compute the page rank. In this last question you'll write a Spark job that iterates over the initialized graph updating each nodes score until it reaches a convergence threshold. The diagram below gives a visual overview of the process using a 5 node toy graph. Pay particular attention to what happens to the dangling mass at each iteration.
# MAGIC 
# MAGIC <img src='https://github.com/kyleiwaniec/w261_assets/blob/master/images/HW5/PR-illustrated.png?raw=true' width=50%>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC __`A Note about Notation:`__ The formula above describes how to compute the updated page rank for a node in the graph. The $P$ on the left hand side of the equation is the new score, and the $P$ on the right hand side of the equation represents the accumulated mass that was re-distributed from all of that node's in-links. Finally, $|G|$ is the number of nodes in the graph (which we've elsewhere refered to as $N$).
# MAGIC 
# MAGIC ### Q8 Tasks:
# MAGIC * __a) short response:__ In terms of the infinite random walk analogy, interpret the meaning of the first term in the PageRank calculation: \\( \alpha * \frac{1}{|G|} \\)
# MAGIC 
# MAGIC * __b) short response:__ In the equation for the PageRank calculation above what does \\(m\\) represent and why do we divide it by \\(|G|\\)?
# MAGIC 
# MAGIC * __c) short response:__ Keeping track of the total probability mass after each update is a good way to confirm that your algorithm is on track. How much should the total mass be after each iteration?
# MAGIC 
# MAGIC * __d) code:__ Fill in the missing code below to create a Spark job that take the initialized graph as its input then iterates over the graph and for each pass:
# MAGIC   * reads in each record and redistributes the node's current score to each of its neighbors
# MAGIC   * uses an accumulator to add up the dangling node mass and redistribute it among all the nodes. (_Don't forget to reset this accumulator after each iteration!_)
# MAGIC   * uses an accumulator to keep track of the total mass being redistributed.( _This is just for your own check, its not part of the PageRank calculation. Don't forget to reset this accumulator after each iteration._)
# MAGIC   * aggregates these partial scores for each node
# MAGIC   * applies telportation and damping factors as described in the formula above.
# MAGIC   * combine all of the above to compute the PageRank as described by the formula above.
# MAGIC   * 
# MAGIC   
# MAGIC    __WARNING:__ Some pages contain multiple hyperlinks to the same destination, please take this into account when redistributing the mass.
# MAGIC 
# MAGIC  
# MAGIC __`NOTE:`__ Please observe scalability best practices in the design of your code & comment your work clearly. You will be graded on both the clarity and the design.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q8 Student Answers:
# MAGIC 
# MAGIC > __a)__ In terms of the infinite random walk analogy, interpret the meaning of the first term in the PageRank calculation: \\(\alpha * \frac{1}{|G|}\\)
# MAGIC > * \\( \alpha\ \\) means the teleportation weight.
# MAGIC > * \\( |G|\\) means the number of webpages in the world.
# MAGIC > * Together \\( \alpha * \frac{1}{|G|} \\) means the probability of randomly going from the current webpage to any random webpage (include the current webpage) in the world when surfer choose to just go randomly instead of clicking a hyperlink of the current webpage.
# MAGIC 
# MAGIC > __b)__ In the equation for the PageRank calculation above what does \\( m \\) represent and why do we divide it by \\(|G|\\)?
# MAGIC > * \\(m\\) represent missing PageRank mass of the webpage that do NOT have a hyperlink.
# MAGIC > * We divide \\(m\\) by \\(|G|\\) because we want to evenly distribute that mass of all webpages in the graph.
# MAGIC > * Intuitively it means that when we encounter a dangling webpage, instead of stuck at that webpage, user can type in a random internet address and go to any website (include the current website) from the current place.
# MAGIC 
# MAGIC > __c)__ How much should the total mass be after each iteration?
# MAGIC > * The total mass after each iteration should be 1.

# COMMAND ----------

# part d - provided FloatAccumulator class (RUN THIS CELL AS IS)

from pyspark.accumulators import AccumulatorParam

class FloatAccumulatorParam(AccumulatorParam):
    """
    Custom accumulator for use in page rank to keep track of various masses.
    
    IMPORTANT: accumulators should only be called inside actions to avoid duplication.
    We stringly recommend you use the 'foreach' action in your implementation below.
    """
    def zero(self, value):
        return value
    def addInPlace(self, val1, val2):
        return val1 + val2

# COMMAND ----------

# part d - job to run PageRank (RUN THIS CELL AS IS)
def runPageRank(graphInitRDD, alpha = 0.15, maxIter = 10, verbose = True):
    """
    Spark job to implement page rank
    Args: 
        graphInitRDD  - pair RDD of (node_id , (score, edges))
        alpha         - (float) teleportation factor
        maxIter       - (int) stopping criteria (number of iterations)
        verbose       - (bool) option to print logging info after each iteration
    Returns:
        steadyStateRDD - pair RDD of (node_id, pageRank)
    """
    # teleportation:
    a = sc.broadcast(alpha)
    
    # damping factor:
    d = sc.broadcast(1-a.value)
    
    # initialize accumulators for dangling mass & total mass
    mmAccum = sc.accumulator(0.0, FloatAccumulatorParam())
    totAccum = sc.accumulator(0.0, FloatAccumulatorParam())
    
    ############## YOUR CODE HERE ###############
    
    # write your helper functions here, 
    # please document the purpose of each clearly 
    # for reference, the master solution has 5 helper functions.

      
    def distribute_mass(line):
      """
      Function to emit a key-value pair for each record
      
      ###Input
      line is (node_id, (page_rank, adj_list))
      adj_list is comprised of a list of tuples, where each tuple is (outgoing node id, number of times it is linked in the node id page)
      
      ###Output
      For each line, emit 2 records
      1) Emit (node, (0.0, adj_list))
      
      2) Emit based on whether or not it is an dangling node
      If it is a dangling node, emit
        ('Dangling' (page_rank, []))
      
      Otherwise, emit 
        (outgoing node_id, (page_rank * weighted distribution of mass, []))
      
      """
      
      # Get node id
      node = line[0]
      
      #Get page rank
      page_rank = line[1][0]

      # Get adjacency list
      adj_list = line[1][1]

      # Get number of adjacent nodes
      num_adj = len(adj_list)

      # Emit default 0 page rank score and adjacency list for node 
      yield(node, (0.0, adj_list))
      
      total_weight = 0

      #Check if node is a dangling node
      if num_adj > 0:

        # Get total number of outgoing page links
        for item in adj_list:
          total_weight += item[1]

        for item in adj_list:
          yield(item[0], (page_rank*item[1]/total_weight,[]))
          
      else:
        
        # Emit mass of dangling node
        yield('##Dangling', (page_rank, []))
                
    # write your main Spark Job here (including the for loop to iterate)
    # for reference, the master solution is 21 lines including comments & whitespace

    if verbose:
      print('-------- FINISHED INITIALIZATION------')
    N_bc = sc.broadcast(graphInitRDD.count())
    
    # loop through each iteration
    for i in range(maxIter):
      mmAccum = sc.accumulator(0.0, FloatAccumulatorParam())
      totAccum = sc.accumulator(0.0, FloatAccumulatorParam())
      
      graphInitRDD.foreach(lambda x: totAccum.add(x[1][0]))
  
      if verbose:
        print('Initial Dangling Mass for iter {}: {}'.format(i,mmAccum.value))

      # NORMAL MAP REDUCE FOR PAGE_RANK CALCULATION
      steadyStateRDD = graphInitRDD.flatMap(lambda x: distribute_mass(x)) \
                                   .reduceByKey(lambda x,y: (x[0] +  y[0], x[1] + y[1])) \
                                   .cache()
      # Add all dangling masses
      #print(steadyStateRDD.filter(lambda x: x[0]=='##Dangling').collect())
      mmAccum.add(steadyStateRDD.filter(lambda x: x[0]=='##Dangling').collect()[0][1][0])
      
      # Remove dangling nodes from RDD 
      steadyStateRDD = steadyStateRDD.filter(lambda x: x[0]!='##Dangling')

      mmAccum_bc = sc.broadcast(mmAccum.value)
      if verbose:
#         print('{}: {}'.format(i, mmAccum_bc.value))
        print('Dangling Mass for iter {}: {}'.format(i, mmAccum_bc.value))
        print('Total Mass for iter {}: {}'.format(i, totAccum.value))
        data = steadyStateRDD.collect()
        for item in data:
          print(item)
  
      # SECOND MAP REDUCE JOB
      steadyStateRDD = steadyStateRDD.mapValues(lambda x: (a.value/N_bc.value + d.value * (mmAccum_bc.value/N_bc.value + x[0]), x[1])).cache()
      
      # Reset graph initialization to result of last iteration
      graphInitRDD = steadyStateRDD
      
      if verbose:
        data = steadyStateRDD.take(10)
        print("Iter: {}".format(i))
        for item in data:
          print(item)
    
    # Reformat RDD
    steadyStateRDD = steadyStateRDD.map(lambda x: (x[0], x[1][0]))
    
    ############## (END) YOUR CODE ###############

    return steadyStateRDD

# COMMAND ----------

# part d - run PageRank on the test graph (RUN THIS CELL AS IS)
# NOTE: while developing your code you may want turn on the verbose option
nIter = 20
testGraphRDD = initGraph(testRDD)
start = time.time()
test_results = runPageRank(testGraphRDD, alpha = 0.15, maxIter = nIter, verbose = True)
print(f'...trained {nIter} iterations in {time.time() - start} seconds.')
print(f'Top 20 ranked nodes:')
test_results.takeOrdered(20, key=lambda x: - x[1])

# COMMAND ----------

# MAGIC %md
# MAGIC __`expected results for the test graph:`__
# MAGIC ```
# MAGIC [(2, 0.3620640495978871),
# MAGIC  (3, 0.333992700474142),
# MAGIC  (5, 0.08506399429624555),
# MAGIC  (4, 0.06030963508473455),
# MAGIC  (1, 0.04255740809817991),
# MAGIC  (6, 0.03138662354831139),
# MAGIC  (8, 0.01692511778009981),
# MAGIC  (10, 0.01692511778009981),
# MAGIC  (7, 0.01692511778009981),
# MAGIC  (9, 0.01692511778009981),
# MAGIC  (11, 0.01692511778009981)]
# MAGIC ```

# COMMAND ----------

# part d - run PageRank on the full graph (RUN THIS CELL AS IS)
# NOTE: wikiGraphRDD should have been computed & cached above!
nIter = 10
start = time.time()
full_results = runPageRank(wikiGraphRDD, alpha = 0.15, maxIter = nIter, verbose = True)
print(f'...trained {nIter} iterations in {time.time() - start} seconds.')
print(f'Top 20 ranked nodes:')
full_results.takeOrdered(20, key=lambda x: - x[1])

# COMMAND ----------

top_20 = full_results.takeOrdered(20, key=lambda x: - x[1])

# COMMAND ----------

# Save the top_20 results to disc for use later. So you don't have to rerun everything if you restart the cluster.

import json
dbutils.fs.put(hw5_path+'top20.json',json.dumps(top_20), True)
display(dbutils.fs.ls(hw5_path))

# COMMAND ----------

import json
with open('/dbfs/user/jeffli930@berkeley.edu/HW5/top20.json') as json_data:
    d = json.load(json_data)
    json_data.close()

df = pd.DataFrame(d)
df.columns = ['Page ID', 'Page Rank']
df

# COMMAND ----------

# view record from indexRDD (RUN THIS CELL AS IS)
# title\t indx\t inDeg\t outDeg
indexRDD.take(1)

# COMMAND ----------

# map indexRDD to new format (index, name) (RUN THIS CELL AS IS)
namesKV_RDD = indexRDD.map(lambda x: (int(x.split('\t')[1]), x.split('\t')[0]))

# COMMAND ----------

# see new format (RUN THIS CELL AS IS)
namesKV_RDD.take(2)

# COMMAND ----------

# MAGIC %md
# MAGIC # OPTIONAL
# MAGIC ### The rest of this notebook is optional and doesn't count toward your grade.
# MAGIC The indexRDD we created earlier from the indices.txt file contains the titles of the pages and thier IDs.
# MAGIC 
# MAGIC * __a) code:__ Join this dataset with your top 20 results.
# MAGIC * __b) code:__ Print the results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Join with indexRDD and print pretty

# COMMAND ----------

# part a
joinedWithNames = None
############## YOUR CODE HERE ###############

############## END YOUR CODE ###############

# COMMAND ----------

# part b
# Feel free to modify this cell to suit your implementation, but please keep the formatting and sort order.
print("{:10s}\t| {:10s}\t| {}".format("PageRank","Page id","Title"))
print("="*100)
for r in joinedWithNames:
    print ("{:6f}\t| {:10d}\t| {}".format(r[1][1],r[0],r[1][0]))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## OPTIONAL - GraphFrames
# MAGIC GraphFrames is a graph library which is built on top of the Spark DataFrames API.
# MAGIC 
# MAGIC * __a) code:__ Using the same dataset, run the graphframes implementation of pagerank.
# MAGIC * __b) code:__ Join the top 20 results with indices.txt and display in the same format as above.
# MAGIC * __c) short answer:__ Compare your results with the results from graphframes.
# MAGIC 
# MAGIC __NOTE:__ Feel free to create as many code cells as you need. Code should be clear and concise - do not include your scratch work. Comment your code if it's not self annotating.

# COMMAND ----------

# imports
import re
import ast
import time
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from graphframes import *
from pyspark.sql import functions as F

# COMMAND ----------

# load the data into Spark RDDs for convenience of use later (RUN THIS CELL AS IS)
DATA_PATH = 'dbfs:/mnt/mids-w261/HW5/'
testRDD = sc.textFile(DATA_PATH +'test_graph.txt')
indexRDD = sc.textFile(DATA_PATH + '/indices.txt')
wikiRDD = sc.textFile(DATA_PATH + '/all-pages-indexed-out.txt')

# COMMAND ----------

# MAGIC %md
# MAGIC ### You will need to generate vertices (v) and edges (e) to feed into the graph below. 
# MAGIC Use as many cells as you need for this task.

# COMMAND ----------

# Create a GraphFrame
from graphframes import *
g = GraphFrame(v, e)


# COMMAND ----------

# Run PageRank algorithm, and show results.
results = g.pageRank(resetProbability=0.15, maxIter=10)

# COMMAND ----------

start = time.time()
top_20 = results.vertices.orderBy(F.desc("pagerank")).limit(20)
print(f'... completed job in {time.time() - start} seconds.')

# COMMAND ----------

# MAGIC %%time
# MAGIC top_20.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run the cells below to join the results of the graphframes pagerank algorithm with the names of the nodes.

# COMMAND ----------

namesKV_RDD = indexRDD.map(lambda x: (int(x.split('\t')[1]), x.split('\t')[0]))

# COMMAND ----------

namesKV_DF = namesKV_RDD.toDF()

# COMMAND ----------

namesKV_DF = namesKV_DF.withColumnRenamed('_1','id')
namesKV_DF = namesKV_DF.withColumnRenamed('_2','title')
namesKV_DF.take(1)

# COMMAND ----------

resultsWithNames = namesKV_DF.join(top_20, namesKV_DF.id==top_20.id).orderBy(F.desc("pagerank")).collect()

# COMMAND ----------

# TODO: use f' for string formatting
print("{:10s}\t| {:10s}\t| {}".format("PageRank","Page id","Title"))
print("="*100)
for r in resultsWithNames:
    print ("{:6f}\t| {:10s}\t| {}".format(r[3],r[2],r[1]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Congratulations, you have completed HW5! Please refer to the readme for submission instructions.
# MAGIC 
# MAGIC If you would like to provide feedback regarding this homework, please use the survey at: https://docs.google.com/forms/d/e/1FAIpQLSce9feiQeSkdP43A0ZYui1tMGIBfLfzb0rmgToQeZD9bXXX8Q/viewform

# COMMAND ----------

