# multiAgents.py
# --------------
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
import sys

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        # print(bestScore)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # print("YOU BETTER WORK B")
        score = 0
        # print(newGhostStates)
        # print(newPos[0])
        food_h = []
        for food in newFood.asList():
            manh = abs(newPos[0] - food[0]) + abs(newPos[1] - food[1])
            food_h.append(manh)
        if food_h:
            score += (1.0 / (min(food_h) + 1))
            # score += 50

        ghost_h = []
        for ghost in newGhostStates:
            ghost_pos = ghost.getPosition()
            manh = abs(newPos[0] - ghost_pos[0]) + abs(newPos[1] - ghost_pos[1])
            ghost_h.append(manh)
        if ghost_h:
            ghost_is_close = min(ghost_h) < 2
            # print(ghost_is_close)
            should_run_away_from_ghost = newScaredTimes[0] == 0 and ghost_is_close
            if should_run_away_from_ghost:
                score -= 500
        #
        got_close_to_ghost = newScaredTimes[0] > 0
        if got_close_to_ghost:
            score += 500

        if successorGameState.isWin():
            score += 1000
        if successorGameState.isLose():
            score -= 1000
        # print(score)
        return successorGameState.getScore() + score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimax(self, id, nextAction, scores, ans, flag):
        res = ans
        if id <= 0:
            if scores[0] > res[0]:
                res = (max(res[0], scores[0]), nextAction)
            if flag:
                res = (scores[0], nextAction)
                flag = False
        else:
            if scores[0] < res[0]:
                res = (min(res[0], scores[0]), nextAction)
        return res, flag

    def recGetAction(self, state, depth, id):
        numAgents = state.getNumAgents()
        currDepth = self.depth
        tmp = (numAgents, currDepth)
        if state.isLose() or state.isWin() or tmp[0] * currDepth == depth:
            return self.evaluationFunction(state), 0

        if id == numAgents:
            id = 0
        ans = (sys.maxsize, 0)
        flag = True
        legalActions = state.getLegalActions(id)
        for nextAction in legalActions:
            successorRes = state.generateSuccessor(id, nextAction)
            scores = self.recGetAction(successorRes, depth + 1, id + 1)
            ans, flag = self.minimax(id, nextAction, scores, ans, flag)
        return ans

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        action = self.recGetAction(gameState, 0, 0)[1]
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        a = float('-inf')
        b = float('inf')
        id = 0
        depth = self.depth
        _, action = self.alpha_betta_getAction(gameState, id, depth, a, b)
        return action

    def is_terminal(self, gameState, depth):
        return gameState.isLose() or gameState.isWin() or depth == 0

    def alpha_betta_getAction(self, gameState, id, depth, a, b):
        # print("HERE1__________________________________________1")
        if self.is_terminal(gameState, depth):
            # print("HERE_________________________________________2")
            return self.evaluationFunction(gameState), 0
        return self.max_value(gameState, id, depth, a, b) if id == 0 else self.min_value(gameState, id, depth, a, b)

    def max_value(self, gameState, id, depth, a, b):
        # print("HERE_________________________________________3")
        score = float('-inf')
        action = 0
        legalActions = gameState.getLegalActions(id)
        for nextAction in legalActions:
            successor = gameState.generateSuccessor(id, nextAction)
            currScore = self.alpha_betta_getAction(successor, 1, depth, a, b)[0]
            if currScore > b:
                return currScore, nextAction
            if currScore > score and currScore > a:
                score = currScore
                action = nextAction
                a = currScore
            elif currScore > score:
                score = currScore
                action = nextAction
        return score, action

    def min_value(self, gameState, id, depth, a, b):
        # print("HERE_________________________________________4")
        score = float('inf')
        action = 0
        legalActions = gameState.getLegalActions(id)
        newId = (id + 1) % gameState.getNumAgents()
        if newId <= 0:
            depth -= 1
        for nextAction in legalActions:
            successor = gameState.generateSuccessor(id, nextAction)
            currScore = self.alpha_betta_getAction(successor, newId, depth, a, b)[0]
            if currScore < a:
                return currScore, nextAction
            if currScore < score:
                score = currScore
                action = nextAction
                if currScore < b:
                    b = currScore
        return score, action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def minimax(self, id, nextAction, scores, ans, flag):
        res = ans
        if id <= 0:
            if scores[0] > res[0]:
                res = (max(res[0], scores[0]), nextAction)
            if flag:
                res = (scores[0], nextAction)
                flag = False
        else:
            res = (res[0] + scores[0], nextAction)
        return res, flag

    def is_terminal(self, gameState):
        return gameState.isLose() or gameState.isWin()

    def recgetAction(self, state, depth, id):
        numAgents = state.getNumAgents()
        currDepth = self.depth
        if self.is_terminal(state):
            return self.evaluationFunction(state), 0
        tmp = (numAgents, currDepth)
        if tmp[0] * currDepth == depth:
            return self.evaluationFunction(state), 0
        if id == numAgents:
            id = 0
        legalActions = state.getLegalActions(id)
        ans = (0, 0)
        flag = True

        for nextAction in legalActions:
            successorRes = state.generateSuccessor(id, nextAction)
            scores = self.recgetAction(successorRes, depth + 1, id + 1)
            ans, flag = self.minimax(id, nextAction, scores, ans, flag)
        if id:
            ans = (ans[0] / len(legalActions), ans[1])
        return ans

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        action = self.recgetAction(gameState, 0, 0)[1]
        return action


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: score should be changed depending on these characteristics:
    +(increase) closet food,
    -(decrease) closest ghost,
    if ghost is scared get closer to ghost, else go far,
    foods remaining,
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
    foodList = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    closestGhostDistance = float('inf')
    closestFoodDistance = float('inf')
    ans = []
    curr = 0
    for state in ghosts:
        if state.scaredTimer > 0:
            continue
        ghostPos = state.getPosition()
        manh = abs(pos[0] - ghostPos[0]) + abs(pos[1] - ghostPos[1])
        closestGhostDistance = getClosetFoodDistance(manh, closestGhostDistance)
        curr += 1
    for food in foodList:
        manh = abs(pos[0] - food[0]) + abs(pos[1] - food[1])
        closestFoodDistance = getClosetFoodDistance(manh, closestFoodDistance)
    if closestGhostDistance == 0:
        closestGhostDistance = 1
    if closestFoodDistance == 0:
        closestFoodDistance = 1
    return score + 1.0 / closestFoodDistance - 1.0 / closestGhostDistance - len(foodList)


def getClosetFoodDistance(x1, x2):
    if x2 is None:
        return x1
    if x1 < x2:
        return x1
    return x2


class GhostCharacteristics:
    def __init__(self):
        self.distance = None
        self.position = 0

    def get_distance(self):
        return self.distance

    def get_position(self):
        return self.position

    def set_distance(self, dist):
        self.distance = dist

    def set_position(self, pos):
        self.position = pos


# Abbreviation
better = betterEvaluationFunction
