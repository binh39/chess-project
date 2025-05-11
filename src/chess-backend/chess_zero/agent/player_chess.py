"""
This encapsulates all of the functionality related to actually playing the game itself, not just
making / training predictions.
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from threading import Lock

import chess
import numpy as np

from chess_zero.config import Config
from chess_zero.env.chess_env import ChessEnv, Winner

logger = getLogger(__name__)


# these are from AGZ nature paper
class VisitStats:
    """
    Holds information for use by the AGZ MCTS algorithm on all moves from a given game state (this is generally used inside
    of a defaultdict where a game state in FEN format maps to a VisitStats object).
    Attributes:
        :ivar defaultdict(ActionStats) a: known stats for all actions to take from the the state represented by
            this visitstats.
        :ivar int sum_n: sum of the n value for each of the actions in self.a, representing total
            visits over all actions in self.a.
    """
    def __init__(self):
        self.a = defaultdict(ActionStats)
        self.sum_n = 0


class ActionStats:
    """
    Holds the stats needed for the AGZ MCTS algorithm for a specific action taken from a specific state.

    Attributes:
        :ivar int n: number of visits to this action by the algorithm
        :ivar float w: every time a child of this action is visited by the algorithm,
            this accumulates the value (calculated from the value network) of that child. This is modified
            by a virtual loss which encourages threads to explore different nodes.
        :ivar float q: mean action value (total value from all visits to actions
            AFTER this action, divided by the total number of visits to this action)
            i.e. it's just w / n.
        :ivar float p: prior probability of taking this action, given
            by the policy network.

    """
    def __init__(self):
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = 0


class ChessPlayer:
    """
    Plays the actual game of chess, choosing moves based on policy and value network predictions coming
    from a learned model on the other side of a pipe.

    Attributes:
        :ivar list: stores info on the moves that have been performed during the game
        :ivar Config config: stores the whole config for how to run
        :ivar PlayConfig play_config: just stores the PlayConfig to use to play the game. Taken from the config
            if not specifically specified.
        :ivar int labels_n: length of self.labels.
        :ivar list(str) labels: all of the possible move labels (like a1b1, a1c1, etc...)
        :ivar dict(str,int) move_lookup: dict from move label to its index in self.labels
        :ivar list(Connection) pipe_pool: the pipes to send the observations of the game to to get back
            value and policy predictions from
        :ivar dict(str,Lock) node_lock: dict from FEN game state to a Lock, indicating
            whether that state is currently being explored by another thread.
        :ivar VisitStats tree: holds all of the visited game states and actions
            during the running of the AGZ algorithm
    """
    # dot = False
    def __init__(self, config: Config, pipes=None, play_config=None, dummy=False):
        self.moves = []

        self.tree = defaultdict(VisitStats)
        self.config = config
        self.play_config = play_config or self.config.play
        self.labels_n = config.n_labels
        self.labels = config.labels
        self.move_lookup = {chess.Move.from_uci(move): i for move, i in zip(self.labels, range(self.labels_n))}
        if dummy:
            return

        self.pipe_pool = pipes
        self.node_lock = defaultdict(Lock)

    def reset(self):
        """
        reset the tree to begin a new exploration of states
        """
        self.tree = defaultdict(VisitStats)

    def deboog(self, env):
        print(env.testeval())

        state = state_key(env)
        my_visit_stats = self.tree[state]
        stats = []
        for action, a_s in my_visit_stats.a.items():
            moi = self.move_lookup[action]
            stats.append(np.asarray([a_s.n, a_s.w, a_s.q, a_s.p, moi]))
        stats = np.asarray(stats)
        a = stats[stats[:,0].argsort()[::-1]]

        for s in a:
            print(f'{self.labels[int(s[4])]:5}: '
                  f'n: {s[0]:3.0f} '
                  f'w: {s[1]:7.3f} '
                  f'q: {s[2]:7.3f} '
                  f'p: {s[3]:7.5f}')

    def action(self, env, can_stop = True) -> str:
        """
        Figures out the next best move
        within the specified environment and returns a string describing the action to take.

        :param ChessEnv env: environment in which to figure out the action
        :param boolean can_stop: whether we are allowed to take no action (return None)
        :return: None if no action should be taken (indicating a resign). Otherwise, returns a string
            indicating the action to take in uci format
        """
        self.reset()

        # for tl in range(self.play_config.thinking_loop):
        root_value, naked_value = self.search_moves(env)
        policy = self.calc_policy(env)
        my_action = int(np.random.choice(range(self.labels_n), p = self.apply_temperature(policy, env.num_halfmoves)))

        if can_stop and self.play_config.resign_threshold is not None and \
                        root_value <= self.play_config.resign_threshold \
                        and env.num_halfmoves > self.play_config.min_resign_turn:
            # noinspection PyTypeChecker
            return None
        else:
            self.moves.append([env.observation, list(policy)])
            return self.config.labels[my_action]

    def search_moves(self, env) -> (float, float):
        """
        Looks at all the possible moves using the AGZ MCTS algorithm
         and finds the highest value possible move. Does so using multiple threads to get multiple
         estimates from the AGZ MCTS algorithm so we can pick the best.

        :param ChessEnv env: env to search for moves within
        :return (float,float): the maximum value of all values predicted by each thread,
            and the first value that was predicted.
        """
        # Lấy số lượng nước đi đã thực hiện
        moves_played = env.num_halfmoves
        
        # Tính toán số lượng mô phỏng dựa trên tiến trình trận đấu
        base_simulations = self.play_config.simulation_num_per_move
        
        # Phân loại giai đoạn trận đấu
        if moves_played < 10:  # Khai cuộc
            # Giảm 50% số lượng mô phỏng ở đầu game
            simulation_factor = 0.5
        elif moves_played < 30:  # Đầu trung cuộc
            # Tăng dần từ 50% đến 100%
            simulation_factor = 0.5 + (moves_played - 10) * 0.025  # Tăng 2.5% mỗi nước đi
        elif moves_played < 60:  # Trung cuộc
            # Giữ nguyên 100%
            simulation_factor = 1.0
        else:  # Tàn cuộc
            # Tăng lên đến 200% trong tàn cuộc
            simulation_factor = 1.0 + min(1.0, (moves_played - 60) * 0.02)  # Tăng 2% mỗi nước, tối đa 200%
        
        # Kiểm tra thêm nếu là tàn cuộc (dựa vào số quân)
        if self.is_endgame(env):
            simulation_factor = max(simulation_factor, 2.0)  # Đảm bảo ít nhất 150% nếu là tàn cuộc
        
        # Tính số lượng mô phỏng thực tế
        simulation_num = int(base_simulations * simulation_factor)
        
        # Log để theo dõi (tuỳ chọn)
        # print(f"Nước {moves_played}: {simulation_num} mô phỏng (hệ số: {simulation_factor:.2f})")
        
        futures = []
        with ThreadPoolExecutor(max_workers=self.play_config.search_threads) as executor:
            for _ in range(simulation_num):  # Sử dụng simulation_num đã điều chỉnh
                futures.append(executor.submit(self.search_my_move,env=env.copy(),is_root_node=True))

        vals = [f.result() for f in futures]

        return np.max(vals), vals[0] # vals[0] is kind of racy

    def search_my_move(self, env: ChessEnv, is_root_node=False) -> float:
        """
        Q, V is value for this Player(always white).
        P is value for the player of next_player (black or white)

        This method searches for possible moves, adds them to a search tree, and eventually returns the
        best move that was found during the search.

        :param ChessEnv env: environment in which to search for the move
        :param boolean is_root_node: whether this is the root node of the search.
        :return float: value of the move. This is calculated by getting a prediction
            from the value network.
        """
        if env.done:
            if env.winner == Winner.draw:
                return 0
            # assert env.whitewon != env.white_to_move # side to move can't be winner!
            return -1

        state = state_key(env)

        with self.node_lock[state]:
            if state not in self.tree:
                leaf_p, leaf_v = self.expand_and_evaluate(env)
                self.tree[state].p = leaf_p
                return leaf_v # I'm returning everything from the POV of side to move

            # SELECT STEP
            action_t = self.select_action_q_and_u(env, is_root_node)

            if self.is_endgame(env):
            # Giảm hoặc loại bỏ virtual loss trong tàn cuộc
                virtual_loss = 0.1  # Hoặc giá trị nhỏ hơn như 1 thay vì 3
            else:
                virtual_loss = self.play_config.virtual_loss

            my_visit_stats = self.tree[state]
            my_stats = my_visit_stats.a[action_t]

            my_visit_stats.sum_n += virtual_loss
            my_stats.n += virtual_loss
            my_stats.w += -virtual_loss
            my_stats.q = my_stats.w / my_stats.n

        env.step(action_t.uci())
        leaf_v = self.search_my_move(env)  # next move from enemy POV
        leaf_v = -leaf_v

        # BACKUP STEP
        # on returning search path
        # update: N, W, Q
        with self.node_lock[state]:
            my_visit_stats.sum_n += -virtual_loss + 1
            my_stats.n += -virtual_loss + 1
            my_stats.w += virtual_loss + leaf_v
            my_stats.q = my_stats.w / my_stats.n

        return leaf_v

    def expand_and_evaluate(self, env) -> (np.ndarray, float):
        """ expand new leaf, this is called only once per state
        this is called with state locked
        insert P(a|s), return leaf_v

        This gets a prediction for the policy and value of the state within the given env
        :return (float, float): the policy and value predictions for this state
        """
        state_planes = env.canonical_input_planes()

        leaf_p, leaf_v = self.predict(state_planes)
        # these are canonical policy and value (i.e. side to move is "white")

        if not env.white_to_move:
            leaf_p = Config.flip_policy(leaf_p) # get it back to python-chess form

        return leaf_p, leaf_v

    def predict(self, state_planes):
        """
        Gets a prediction from the policy and value network
        :param state_planes: the observation state represented as planes
        :return (float,float): policy (prior probability of taking the action leading to this state)
            and value network (value of the state) prediction for this state.
        """
        pipe = self.pipe_pool.pop()
        pipe.send(state_planes)
        ret = pipe.recv()
        self.pipe_pool.append(pipe)
        return ret

    #@profile
    def select_action_q_and_u(self, env, is_root_node) -> chess.Move:
        """
        Picks the next action to explore using the AGZ MCTS algorithm.

        Picks based on the action which maximizes the maximum action value
        (ActionStats.q) + an upper confidence bound on that action.

        :param Environment env: env to look for the next moves within
        :param is_root_node: whether this is for the root node of the MCTS search.
        :return chess.Move: the move to explore
        """
        # this method is called with state locked
        state = state_key(env)

        my_visitstats = self.tree[state]

        if my_visitstats.p is not None: #push p to edges
            tot_p = 1e-8
            for mov in env.board.legal_moves:
                mov_p = my_visitstats.p[self.move_lookup[mov]]
                my_visitstats.a[mov].p = mov_p
                tot_p += mov_p
            for a_s in my_visitstats.a.values():
                a_s.p /= tot_p
            my_visitstats.p = None

        xx_ = np.sqrt(my_visitstats.sum_n + 1)  # sqrt of sum(N(s, b); for all b)

        e = self.play_config.noise_eps
        c_puct = self.play_config.c_puct
        dir_alpha = self.play_config.dirichlet_alpha

        best_s = -999
        best_a = None
        if is_root_node:
            if self.is_endgame(env):
                e = self.play_config.noise_eps * 0.2  # Giảm nhiễu trong tàn cuộc
            noise = np.random.dirichlet([dir_alpha] * len(my_visitstats.a))
        
        i = 0
        for action, a_s in my_visitstats.a.items():
            p_ = a_s.p
            if is_root_node:
                p_ = (1-e) * p_ + e * noise[i]
                i += 1
            b = a_s.q + c_puct * p_ * xx_ / (1 + a_s.n)
            if b > best_s:
                best_s = b
                best_a = action

        return best_a

    def apply_temperature(self, policy, turn):
        """
        Applies a random fluctuation to probability of choosing various actions
        """
        # Kiểm tra nếu đang ở tàn cuộc
        is_endgame = False
        for state, visit_stats in self.tree.items():
            if len(self.tree) > 0:  # Có ít nhất một trạng thái trong cây
                # Lấy trạng thái đầu tiên để kiểm tra
                board = chess.Board(state)
                if len(board.piece_map()) <= 10:  # Đơn giản hóa kiểm tra tàn cuộc
                    is_endgame = True
                    break
        
        # Trong tàn cuộc, đặt tau=0 để luôn chọn nước đi tốt nhất
        if is_endgame:
            tau = 0
        else:
            tau = np.power(self.play_config.tau_decay_rate, turn + 1)
            if tau < 0.1:
                tau = 0
        
        if tau == 0:
            action = np.argmax(policy)
            ret = np.zeros(self.labels_n)
            ret[action] = 1.0
            return ret
        else:
            ret = np.power(policy, 1/tau)
            ret /= np.sum(ret)
            return ret

    def calc_policy(self, env):
        """calc π(a|s0)
        :return list(float): a list of probabilities of taking each action, calculated based on visit counts.
        """
        state = state_key(env)
        my_visitstats = self.tree[state]
        policy = np.zeros(self.labels_n)
        for action, a_s in my_visitstats.a.items():
            policy[self.move_lookup[action]] = a_s.n

        policy /= np.sum(policy)
        return policy

    def sl_action(self, observation, my_action, weight=1):
        """
        Logs the action in self.moves. Useful for generating a game using game data.

        :param str observation: FEN format observation indicating the game state
        :param str my_action: uci format action to take
        :param float weight: weight to assign to the taken action when logging it in self.moves
        :return str: the action, unmodified.
        """
        policy = np.zeros(self.labels_n)

        k = self.move_lookup[chess.Move.from_uci(my_action)]
        policy[k] = weight

        self.moves.append([observation, list(policy)])
        return my_action

    def finish_game(self, z):
        """
        When game is done, updates the value of all past moves based on the result.

        :param self:
        :param z: win=1, lose=-1, draw=0
        :return:
        """
        for move in self.moves:  # add this game winner result to all past moves.
            move += [z]

    def is_endgame(self, env):
        """Xác định xem vị thế hiện tại có phải là tàn cuộc không.
        
        Tàn cuộc thường được định nghĩa là:
        1. Tổng số quân ít hơn một ngưỡng cụ thể
        2. Không có hậu
        3. Hoặc ít tốt
        """
        board = env.board
        
        # Đếm tổng số quân
        total_pieces = len(board.piece_map())
        
        # Kiểm tra xem có hậu không
        has_queens = board.pieces(chess.QUEEN, chess.WHITE) or board.pieces(chess.QUEEN, chess.BLACK)
        
        # Đếm số tốt
        white_pawns = len(board.pieces(chess.PAWN, chess.WHITE))
        black_pawns = len(board.pieces(chess.PAWN, chess.BLACK))
        total_pawns = white_pawns + black_pawns
        
        # Điều kiện tàn cuộc
        if total_pieces <= 10:  # Ít quân
            return True
        if not has_queens and total_pieces <= 14:  # Không có hậu và ít quân
            return True
        if total_pawns <= 4 and total_pieces <= 12:  # Ít tốt và ít quân
            return True
        
        return False


def state_key(env: ChessEnv) -> str:
    """
    :param ChessEnv env: env to encode
    :return str: a str representation of the game state
    """
    fen = env.board.fen().rsplit(' ', 1) # drop the move clock
    return fen[0]