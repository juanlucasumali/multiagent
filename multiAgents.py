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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        
        # get closer to food
        foodList = newFood.asList()
        if foodList:
            # find dist to nearest food pellet
            minFoodDistance = min([manhattanDistance(newPos, food) for food in foodList])
            # closer food = better
            score += 15.0 / (minFoodDistance + 1)
        
        # stay away from dangerous ghosts, chase scared ones
        minGhostDistance = float('inf')
        
        for i, ghostState in enumerate(newGhostStates):
            ghostPos = ghostState.getPosition()
            ghostDistance = manhattanDistance(newPos, ghostPos)
            minGhostDistance = min(minGhostDistance, ghostDistance)
            
            if newScaredTimes[i] > 0:
                if ghostDistance <= 1:
                    score += 250  # bonus for eating scared ghost
                else:
                    score += 30.0 / (ghostDistance + 1)  # closer is better
            else:
                # ghost is dangerous, avoid
                if ghostDistance <= 1:
                    score -= 1200  # penalty for being too close
                elif ghostDistance <= 2:
                    score -= 300   # penalty for being close
                else:
                    score += ghostDistance * 2  # farther is better
        
        # reward eating food pellets
        foodEaten = currentGameState.getNumFood() - successorGameState.getNumFood()
        score += foodEaten * 25
        
        # reward eating power pellets
        if newPos in currentGameState.getCapsules():
            score += 150
        
        # penalty for leaving too much food (encourage completion)
        remainingFood = successorGameState.getNumFood()
        score -= remainingFood * 0.15
        
        # reward being far from all ghosts
        if minGhostDistance > 3:
            score += 75
        
        # big reward for winning
        if successorGameState.isWin():
            score += 500
        
        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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
        "*** YOUR CODE HERE ***"
        def minimax(state, depth, agentIndex):
            # base case: terminal state or max depth reached
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            
            # get possible actions for current agent
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)
            
            # determine next agent and depth
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            if nextAgent == 0:  # back to pacman, decrease depth
                depth -= 1
            
            # evaluate all possible actions
            values = []
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value = minimax(successor, depth, nextAgent)
                values.append(value)
            
            # return max for pacman, min for ghosts
            if agentIndex == 0:
                return max(values)
            else:
                return min(values)
        
        # get pacman's possible actions
        legalActions = gameState.getLegalActions(0)
        if not legalActions:
            return Directions.STOP
        
        # find the best action for pacman
        bestAction = None
        bestValue = float('-inf')
        
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = minimax(successor, self.depth, 1)  # start with first ghost
            if value > bestValue:
                bestValue = value
                bestAction = action
        
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphabeta(state, depth, agentIndex, alpha, beta):
            """
            Alpha-beta pruning algorithm:
            - alpha: best value for MAX (pacman) so far
            - beta: best value for MIN (ghosts) so far
            - prune branches that won't affect the final decision
            """
            # base case: terminal state/max depth reached
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            
            # get possible actions for current agent
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)
            
            # determine next agent and depth
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            if nextAgent == 0:  # back to pacman, decrease depth
                depth -= 1
            
            # pacman's turn
            if agentIndex == 0:
                v = float('-inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    v = max(v, alphabeta(successor, depth, nextAgent, alpha, beta))
                    if v > beta:  # prune: min won't choose this branch
                        return v
                    alpha = max(alpha, v)
                return v
            
            # ghosts turn
            else:
                v = float('inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    v = min(v, alphabeta(successor, depth, nextAgent, alpha, beta))
                    if v < alpha:  # prune
                        return v
                    beta = min(beta, v)
                return v
        
        # get pacman's possible actions
        legalActions = gameState.getLegalActions(0)
        if not legalActions:
            return Directions.STOP
        
        # find the best action for pacman using alpha-beta
        bestAction = None
        bestValue = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = alphabeta(successor, self.depth, 1, alpha, beta)  # start with first ghost
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, value)  # update alpha for next iteration
        
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
