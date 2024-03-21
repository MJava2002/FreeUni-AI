# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def getNewValue(self, state):
        qValue = self.computeQValueFromValues(state, self.computeActionFromValues(state))

    def runValueIteration(self):
        # Write value iteration code here
        for i in range(self.iterations):
            val = util.Counter()
            for state in self.mdp.getStates():
                isTerminal = self.mdp.isTerminal(state)
                noActions = not self.mdp.getPossibleActions(state)
                if not (noActions and isTerminal):
                    val[state] = self.computeQValueFromValues(state, self.computeActionFromValues(state))
            self.values = val


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def calculateRawValue(self, state, action, nextState):
        return self.mdp.getReward(state, action, nextState) + self.discount * self.getValue(nextState)

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # if self.mdp.isTerminal(state):
        #     return 0
        transitionInfo = self.mdp.getTransitionStatesAndProbs(state, action)
        qVal = 0
        for st, prob in transitionInfo:
            qVal += prob * self.calculateRawValue(state, action, st)
        return qVal

    def computeBestAction(self, state):
        possibleActions = self.mdp.getPossibleActions(state)
        bestAction = None
        maxVal = float('-inf')
        for action in possibleActions:
            value = self.computeQValueFromValues(state, action)
            if value > maxVal:
                bestAction = action
                maxVal = value
        return bestAction


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        isTerminal = self.mdp.isTerminal(state)
        noActions = not self.mdp.getPossibleActions(state)
        if noActions and isTerminal:
            return None
        return self.computeBestAction(state)

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        nStates = len(states)
        for i in range(self.iterations):
            state = states[i % nStates]
            isTerminal = self.mdp.isTerminal(state)
            if not isTerminal:
                self.values[state] = self.computeQValueFromValues(state, self.computeActionFromValues(state))


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def getPredecessors(self, states):
        predecessors = {state: set() for state in states}
        for state in states:
            isTerminal = self.mdp.isTerminal(state)
            if not isTerminal:
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    for s, _ in self.mdp.getTransitionStatesAndProbs(state, action):
                        predecessors[s].add(state)
        return predecessors

    def runValueIteration(self):
        priorityQueue = util.PriorityQueue()
        states = self.mdp.getStates()
        predecessors = self.getPredecessors(states)
        for state in states:
            isTerminal = self.mdp.isTerminal(state)
            if not isTerminal:
                diff = self.calculateDiff(state)
                priorityQueue.push(state, -1 * diff)
        for iteration in range(self.iterations):
            if priorityQueue.isEmpty():
                break
            state = priorityQueue.pop()
            if self.mdp.isTerminal(state):
                continue
            self.values[state] = self.computeQValueFromValues(state, self.computeActionFromValues(state))
            for predecessor in predecessors[state]:
                diff = self.calculateDiff(predecessor)
                if diff > self.theta:
                    priorityQueue.update(predecessor, -diff)

    def calculateDiff(self, state):
        return abs(self.values[state] - self.computeQValueFromValues(state, self.computeActionFromValues(state)))
