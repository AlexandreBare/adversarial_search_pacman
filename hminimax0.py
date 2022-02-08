from pacman_module.game import Agent
from pacman_module.pacman import Directions
from pacman_module.util import manhattanDistance


class PacmanAgent(Agent):
    """
    PacmanAgent responsible for deciding the next moves of pacman
    """

    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """

        self.args = args
        self.moves = []
        self.previousKeyStates = set()
        self.current_score = 0

    def get_action(self, state):
        """
        Given a pacman game state, returns a legal move.

        Arguments:
        ----------
        - `state`: the current game state. See FAQ and class `pacman.GameState`.

        Return:[]
        -------
        - A legal move as defined in `game.Directions`.
        """

        self.previousKeyStates.add((state.getPacmanPosition(), state.getFood()))

        if not self.moves:
            self.current_score = state.getScore()
            self.moves.append(self.hminimax(state))

        try:
            action = self.moves.pop()
            return action

        except IndexError:
            return Directions.STOP

    def eval_function(self, state):
        """
        Returns a numeric value to evaluate a state, i.e. a higher value if pacman has more chances to win than lose
        This function penalises a state evaluation with the distance between pacman and the nearest food dot, and with
        the number of food dots left to eat. It also penalises pacman when revisiting
        a previously taken (pacman position, food grid) state.
        It rewards a state evaluation with the the distance between pacman and the ghost, with the difference of score
        between the current state and the evaluated state, with the maximum distance between a food dot and pacman,
        and with the number of directions pacman can take

        Arguments:
        ----------
        - 'state': the current game state. See FAQ and class `pacman.GameState`.

        Return:
        -------
        - the numeric value to evaluate the state
        """

        pac_pos = state.getPacmanPosition()
        ghost_pos = state.getGhostPosition(1)
        pac_ghost_dist = manhattanDistance(pac_pos, ghost_pos)

        num_food = 0
        min_pac_food_dist = float('inf')
        max_pac_food_dist = float('-inf')
        i = 0
        j = 0
        for row in state.getFood():
            for element in row:
                if element:
                    num_food += 1
                    dist = manhattanDistance((i, j), pac_pos)
                    if dist < min_pac_food_dist:
                        min_pac_food_dist = dist
                    if dist > max_pac_food_dist:
                        max_pac_food_dist = dist

                j += 1

            i += 1
            j = 0

        if num_food == 0:
            min_pac_food_dist = 0
            max_pac_food_dist = 0

        already_visited_state_penalty = 0

        if (state.getPacmanPosition(), state.getFood()) in self.previousKeyStates:
            already_visited_state_penalty = 400

        return state.getScore() - self.current_score - already_visited_state_penalty + 1 / 2 * pac_ghost_dist \
               - min_pac_food_dist - 2 * num_food + 1 / 2 * max_pac_food_dist + 1 / 2 * len(state.getLegalActions(0))

    def cutoff(self, state, depth, is_pacman):
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
            pac_ghost_dist = manhattanDistance(pac_pos, ghost_pos)

            # If pacman is too far from the ghost to be caught
            if is_pacman:
                if pac_ghost_dist > 2:
                    return True
            else:
                if pac_ghost_dist >= 2:
                    return True

            x_diff = ghost_pos[0] - pac_pos[0]
            y_diff = ghost_pos[1] - pac_pos[1]
            ghost_dir = state.getGhostDirection(1)
            horizontal_dir = ""
            vertical_dir = ""

            if x_diff > 0:
                horizontal_dir = "East"
            elif x_diff < 0:
                horizontal_dir = "West"

            if y_diff > 0:
                vertical_dir = "North"
            elif y_diff < 0:
                vertical_dir = "South"

            # If the ghost is oriented in the opposite direction
            # of pacman's position, the ghost can not catch
            # pacman next turn as it can not make a half turn
            # (unless it is obliged to do so)
            if vertical_dir == ghost_dir or horizontal_dir == ghost_dir:
                return True

        return False

    def hminimax_aux(self, state, alpha, beta, is_pacman, depth):
        """
        Recursive auxiliary hminimax method (see hminimax algorithm)

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

        if self.cutoff(state, depth, is_pacman):
            return [self.eval_function(state), None]

        # Maximising Player (Pacman)
        if is_pacman:

            max_value = float('-inf')
            best_action = None
            for next_state, action in state.generatePacmanSuccessors():
                value = self.hminimax_aux(next_state, alpha, beta, False, depth + 1)[0]

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
                value = self.hminimax_aux(next_state, alpha, beta, True, depth + 1)[0]

                if value < min_value and value != float('-inf'):
                    min_value = value

                # alpha-beta pruning
                if min_value <= alpha:
                    return [min_value, best_action]

                beta = min(beta, min_value)
                #

            return [min_value, best_action]

    def hminimax(self, state):
        """
        Method of the hminimax algorithm

        Arguments:
        ----------
        - 'state': the current game state. See FAQ and class `pacman.GameState`.

        Return:
        -------
        - the estimated best action leading to the estimated best next state
        """

        return self.hminimax_aux(state, float('-inf'), float('inf'), True, 0)[1]