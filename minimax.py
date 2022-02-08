from pacman_module.game import Agent
from pacman_module.pacman import Directions


def key(state):
    """
    Returns a key that uniquely identifies a Pacman game state.

    Arguments:
    ----------
    - 'state': the current game state. See FAQ and class `pacman.GameState`.

    Return:
    -------
    - A hashable key object that uniquely identifies a Pacman game state containing
    pacman's position, the current food grid representation, the ghost's position and direction.
    """

    return state.getPacmanPosition(), state.getFood(), state.getGhostPosition(1), state.getGhostDirection(1)


def terminal(state):
    """
    Returns whether the state of the game is an end state or not

    Arguments:
    ----------
    - 'state': the current game state. See FAQ and class `pacman.GameState`.

    Return:
    -------
    - a boolean that is True if the game state is an end state and False otherwise
    """

    if state.isLose() | state.isWin():
        return True
    return False


def utility_function(state):
    """
    Returns the final numeric value for an end state, i.e. a higher value if pacman wins than if it loses

    Arguments:
    ----------
    - 'state': the current game state. See FAQ and class `pacman.GameState`.

    Return:
    -------
    - the final numeric value for an end state, i.e. 1 if pacman wins, 0 if it loses
    """

    return state.isWin()


def minimax_aux(state, alpha, beta, is_pacman, visited):
    """
    Recursive auxiliary minimax function (see minimax algorithm)

    Arguments:
    ----------
    - 'state': the current game state. See FAQ and class `pacman.GameState`.
    - 'alpha': alpha argument for pruning the minimax tree (see alpha-beta pruning algorithm)
    - 'beta': beta argument for pruning the minimax tree (see alpha-beta pruning algorithm)
    - 'is_pacman': boolean that is true if it is pacman turn (the maximising player) in the minimax tree,
                  false otherwise
    - 'visited': the dictionary of visited state; key: the state key; value: the score of the state

    Return:
    -------
    - a list containing the utility value of a possible series of moves from the current state to a final state and
    the best action leading to the best next state
    """

    if terminal(state):
        return [utility_function(state), None]

    # Maximising Player (Pacman)
    if is_pacman:

        max_value = float('-inf')
        best_action = None
        for next_state, action in state.generatePacmanSuccessors():
            next_key = key(next_state)
            score = next_state.getScore()
            better_state = False
            if next_key in visited.keys() and score > visited[next_key]:
                better_state = True  # An already visited state has a better score now, so it should be updated

            if next_key not in visited.keys() or better_state:
                visited[next_key] = score
                value = minimax_aux(next_state, alpha, beta, False, visited)[0]
                if value > max_value and value != float('inf'):
                    max_value = value
                    best_action = action

                # alpha-beta pruning
                if max_value >= beta:
                    return [max_value, best_action]

                alpha = max(alpha, max_value)
                #

        return [max_value, best_action]

    # Minimising Player (Ghost)
    else:

        min_value = float('inf')
        best_action = None
        for next_state, action in state.generateGhostSuccessors(1):
            next_key = key(next_state)
            score = next_state.getScore()
            better_state = False
            if next_key in visited.keys():
                if score > visited[next_key]:
                    better_state = True  # An already visited state has a better score now, so it should be updated

            if next_key not in visited.keys() or better_state:
                visited[next_key] = score
                value = minimax_aux(next_state, alpha, beta, True, visited)[0]
                if value < min_value and value != float('-inf'):
                    min_value = value

                if min_value <= alpha:
                    return [min_value, best_action]

                beta = min(beta, min_value)

        return [min_value, best_action]


def minimax(state):
    """
    Function of the minimax algorithm

    Arguments:
    ----------
    - 'state': the current game state. See FAQ and class `pacman.GameState`.

    Return:
    -------
    - the best action leading to the best next state
    """

    return minimax_aux(state, float('-inf'), float('inf'), True, {})[1]


class PacmanAgent(Agent):
    """
    PacmanAgent responsible for deciding the next moves of pacman
    """

    def __init__(self, args):
        """
        Arguments:
        ----------
        - 'args': Namespace of arguments from command-line prompt.
        """

        self.args = args
        self.moves = []

    def get_action(self, state):
        """
        Given a pacman game state, returns a legal move.

        Arguments:
        ----------
        - 'state': the current game state. See FAQ and class `pacman.GameState`.

        Return:[]
        -------
        - A legal move as defined in `game.Directions`.
        """

        if not self.moves:
            self.moves.append(minimax(state))

        try:
            return self.moves.pop()

        except IndexError:
            return Directions.STOP