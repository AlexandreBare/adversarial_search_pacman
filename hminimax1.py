from pacman_module.game import Agent
from pacman_module.pacman import Directions
from pacman_module.util import manhattanDistance


def key(state):
    """
    Returns a key that uniquely identifies a Pacman game state.

    Arguments:
    ----------
    - `state`: the current game state. See FAQ and class
               `pacman.GameState`.

    Return:
    -------
    - A hashable key object that uniquely identifies a Pacman game state containing
    pacman's position, the current food grid representation, the ghost's position and direction.
    """

    return state.getPacmanPosition(), state.getFood(), state.getGhostPosition(1), state.getGhostDirection(1)


def eval_function(state):
    """
    Returns a numeric value to evaluate a state, i.e. a higher value if pacman has more chances to win than lose
    This function penalises a state evaluation with the average manhattan distance between pacman and the food dots,
    and the number of food left to eat.
    It rewards a state evaluation with the score of the current state.

    Arguments:
    ----------
    - 'state': the current game state. See FAQ and class `pacman.GameState`.

    Return:
    -------
    - the numeric value to evaluate the state
    """

    pac_pos = state.getPacmanPosition()
    num_food = 0
    min_pac_food_dist = float('inf')
    i = 0
    j = 0

    for row in state.getFood():
        for element in row:
            if element:
                num_food += 1
                dist = manhattanDistance((i, j), pac_pos)
                if dist < min_pac_food_dist:
                    min_pac_food_dist = dist

            j += 1
        i += 1
        j = 0

    if num_food == 0:
        min_pac_food_dist = 0

    return state.getScore() - 4 * min_pac_food_dist - 10 * num_food


def cutoff(state, depth, is_pacman):
    """
    Determines when the hminimax algorithm stops to expand a state, typically at a quiescent or final state

    Arguments:
    ----------
    - 'state': the current game state. See FAQ and class `pacman.GameState`.
    - 'depth': the depth of the recursive call
    - 'is_pacman': a boolean that is true it it is pacman's turn, false otherwise

    Return:
    -------
    - a boolean that is true if the hminimax algorithm stops the expansion of a state, false otherwise
    """

    if depth > 0:
        # If the state is a final state
        if state.isWin() or state.isLose():
            return True

        pac_pos = state.getPacmanPosition()
        ghost_pos = state.getGhostPosition(1)

        # If pacman is closer to the nearest food dot than the ghost
        if is_pacman:
            if manhattanDistance(pac_pos, ghost_pos) > 2:
                return True
        else:
            if manhattanDistance(pac_pos, ghost_pos) >= 2:
                return True

        pac_food_dist = 0
        ghost_food_dist = 0

        i = 0
        j = 0

        for row in state.getFood():
            for element in row:
                if element:
                    pac_food_dist += manhattanDistance((i, j), pac_pos)
                    ghost_food_dist += manhattanDistance((i, j), ghost_pos)
                j += 1
            i += 1
            j = 0

        pac_food_dist = pac_food_dist
        ghost_food_dist = ghost_food_dist

        # If the sum of manhattan distances between pacman and the food dots is smaller than the sum of manhattan
        # distances between the ghost and the food dots and it is pacman's turn
        if pac_food_dist <= ghost_food_dist and is_pacman:
            return True

    return False


def hminimax_aux(state, alpha, beta, is_pacman, visited, depth):
    """
    Recursive auxiliary hminimax function (see hminimax algorithm)

    Arguments:
    ----------
    - 'state': the current game state. See FAQ and class `pacman.GameState`.
    - 'alpha': alpha argument for pruning the minimax tree (see alpha-beta pruning algorithm)
    - 'beta': beta argument for pruning the minimax tree (see alpha-beta pruning algorithm)
    - 'is_pacman': boolean that is true if it is pacman's turn (the maximising player) in the minimax tree,
                  false otherwise
    - 'visited': the dictionary of visited states; key: the state key; value: the score of the state
    - 'depth': the depth of the recursive call

    Return:
    -------
    - a list containing the evaluation of a possible series of moves from the current state to a quiescent or
    final state and the estimated best action leading to the estimated best next state
    """

    if cutoff(state, depth, is_pacman):
        return [eval_function(state), None]

    # Maximising Player (Pacman)
    if is_pacman:

        max_value = float('-inf')
        best_action = None
        for next_state, action in state.generatePacmanSuccessors():
            next_key = key(next_state)
            score = next_state.getScore()
            better_state = False
            if next_key in visited.keys():
                if score > visited[next_key]:
                    better_state = True  # An already visited state has a better score now, so it should be updated

            if next_key not in visited.keys() or better_state:
                visited[next_key] = score
                value = hminimax_aux(next_state, alpha, beta, False, visited, depth + 1)[0]

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
                value = hminimax_aux(next_state, alpha, beta, True, visited, depth + 1)[0]

                if value < min_value and value != float('-inf'):
                    min_value = value

                # alpha-beta pruning
                if min_value <= alpha:
                    return [min_value, best_action]

                beta = min(beta, min_value)
                #

        return [min_value, best_action]


def hminimax(state):
    """
    Method of the hminimax algorithm

    Arguments:
    ----------
    - 'state': the current game state. See FAQ and class `pacman.GameState`.

    Return:
    -------
    - the estimated best action leading to the estimated best next state
    """
    return hminimax_aux(state, float('-inf'), float('inf'), True, {}, 0)[1]


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
            self.moves.append(hminimax(state))

        try:
            action = self.moves.pop()
            return action

        except IndexError:
            return Directions.STOP