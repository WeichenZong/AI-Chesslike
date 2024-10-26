# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from time import sleep
import random

import heapq
from collections import deque


@register_agent("student_agent17")
class StudentAgent17(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent17, self).__init__()
        self.name = "StudentAgent17"
        self.autoplay = True
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        x_max, _, _ = chess_board.shape
        self.board_size = x_max

        self.start_time = time.time()
        
        # Generate a random move in case first iteration of alpha-beta doesnt finish (shouldn't happen but just in case)
        best_move = self.possibleMoves2(chess_board, my_pos, adv_pos, max_step)[-1] 
        
        # Continously increment depth by 1 and repeat alpha-beta algorithm
        for i in range(1, 60):
            try:
                # Best move from current iteration
                best_value, new_best_move = self.max_value(chess_board, my_pos, adv_pos, max_step, depth=i, alpha=float("-inf"), beta=float("inf"))
                if best_value > -100000:
                    # Not a guaranteed loss, proceed as normal
                    best_move = new_best_move
                else:
                    # Guaranteed loss with optimal play, play to stay alive as long as possible in case opponent makes a mistake
                    return best_move[0], best_move[1]
            except RuntimeError:
                # Upon timeout, return best move from previous iteration
                return best_move[0], best_move[1]
        
        # If we finish all iterations somehow (shouldn't happen but if it does the game is already decided anyways)
        return best_move[0], best_move[1]
        
    def max_value(self, chess_board, my_pos, adv_pos, max_step, depth, alpha, beta):
        """ 
        Max function of Alpha-Beta algorithm
        """
        if time.time() - self.start_time > 1.95:
            raise RuntimeError("2 second time limit reached")

        value = self.evaluation(chess_board, my_pos, adv_pos, max_step, depth)
        if value != None:
            return value, None
        
        # Filtered subset of available moves
        p = self.possibleMoves2(chess_board, my_pos, adv_pos, max_step)
        
        best_move = None
        
        for i in reversed(p):
            r, c, dir = i[0][0], i[0][1], i[1]
            
            # Update board
            chess_board[r, c, dir] = True
            move = self.moves[dir]
            chess_board[r + move[0], c + move[1], self.opposites[dir]] = True
            
            # Get value from each min node in the next level
            min_value, min_move = self.min_value(chess_board, adv_pos, (r, c), max_step, depth - 1, alpha, beta)
            
            # Undo board update
            chess_board[r, c, dir] = False
            chess_board[r + move[0], c + move[1], self.opposites[dir]] = False
            
            # Update alpha value if needed
            if min_value > alpha:
                alpha = min_value
                best_move = i
            if alpha >= beta:
                return beta, i
            
        return alpha, best_move
    
    def min_value(self, chess_board, my_pos, adv_pos, max_step, depth, alpha, beta):
        """
        Min function of Alpha-Beta algorithm
        """
        if time.time() - self.start_time > 1.95:
            raise RuntimeError("2 second time limit reached")
        
        value = self.evaluation(chess_board, adv_pos, my_pos, max_step, depth)
        if value != None:
            return value, None
        
        # Filtered subset of available moves
        p = self.possibleMoves2(chess_board, my_pos, adv_pos, max_step)
        
        best_move = None
        
        for i in p:
            r, c, dir = i[0][0], i[0][1], i[1]
            
            # Update board
            chess_board[r, c, dir] = True
            move = self.moves[dir]
            chess_board[r + move[0], c + move[1], self.opposites[dir]] = True
            
            # Get value from each max node in the next level
            max_value, max_move = self.max_value(chess_board, adv_pos, (r, c), max_step, depth - 1, alpha, beta)
            
            # Undo board update
            chess_board[r, c, dir] = False
            chess_board[r + move[0], c + move[1], self.opposites[dir]] = False
            
            # Update beta value if needed
            if max_value < beta:
                beta = max_value
                best_move = max_move
            if alpha >= beta:
                return alpha, i
        
        return beta, best_move
    
    def evaluation(self, chess_board, my_pos, adv_pos, max_step, depth):
        """ 
        Evaluation function.
        """
        # If game over, check what player won and assign value: win = 1M, loss = -1M, tie = -20
        res, p0, p1 = self.check_endgame(chess_board, my_pos, adv_pos)
        if res:
            if p0 > p1:
                return 1000000
            elif p0 < p1:
                return -1000000
            else:
                return -20
        
        # Reached max depth of the tree, estimate value of the game state
        if depth == 0:
            moves_val = len(self.possibleMoves(chess_board, my_pos, adv_pos, max_step)) - len(self.possibleMoves(chess_board, adv_pos, my_pos, max_step))
            control_val = self.check_player_board_control(chess_board, my_pos, adv_pos)
        
            return moves_val + control_val
            
        # Go to the next level of the tree
        return None
    
    def check_player_board_control(self, chess_board, my_pos, adv_pos):
        """ 
        Calculate how many squares of the board are closer to each player than the other and return the difference.
        """
        # Store the distance to each visited node
        distances = np.zeros((self.board_size, self.board_size))
        
        s = ((-1, 0), (0, 1), (1, 0), (0, -1))
        
        # An element in the queue contains the position of the square, the distance from the closest player to it and the number of the player
        state_queue = deque([(my_pos, 0, 0)]) # (x, y), distance, player | 0 if p0, 1 if p1, 2 if both
        state_queue.append((adv_pos, 0, 1))
        
        # Keep track of squares visited by each player
        visited_p0 = set()
        visited_p0.add((my_pos[0], my_pos[1]))
        visited_p1 = set()
        visited_p1.add((adv_pos[0], adv_pos[1]))

        # Custom BFS with 2 starting nodes
        while state_queue:
            cur_pos, cur_step, player = state_queue.popleft()
            r = cur_pos[0]
            c = cur_pos[1]
            for dir, move in enumerate(s):
                if chess_board[r, c, dir]:
                    continue
                next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
                
                if player == 0:
                    # Already visited or not reachable
                    if next_pos in visited_p0 or (next_pos[0] == adv_pos[0] and next_pos[1] == adv_pos[1]):
                        continue
                    # Visited by other player
                    if next_pos in visited_p1:
                        if distances[next_pos[0]][next_pos[1]] == cur_step + 1:
                            # Same distance to each player so add it to both, update entry in queue to player number 2
                            visited_p0.add(next_pos)
                            for i in range(144):
                                if state_queue[i] == (next_pos, cur_step + 1, 1):
                                    state_queue[i]=(next_pos, cur_step + 1, 2)
                                    break
                    else:
                        # Add to visited and to queue
                        visited_p0.add(next_pos)
                        distances[next_pos[0]][next_pos[1]] = cur_step + 1
                        state_queue.append((next_pos, cur_step + 1, 0))
                
                elif player == 1:
                    # Already visited or not reachable
                    if next_pos in visited_p1 or (next_pos[0] == my_pos[0] and next_pos[1] == my_pos[1]):
                        continue
                    # Visited by other player
                    if next_pos in visited_p0:
                        if distances[next_pos[0]][next_pos[1]] == cur_step + 1:
                            # Same distance to each player so add it to both, update entry in queue to player number 2
                            visited_p1.add(next_pos)
                            for i in range(144):
                                if state_queue[i] == (next_pos, cur_step + 1, 0):
                                    state_queue[i]=(next_pos, cur_step + 1, 2)
                                    break
                    else:
                        # Add to visited and to queue
                        visited_p1.add(next_pos)
                        distances[next_pos[0]][next_pos[1]] = cur_step + 1
                        state_queue.append((next_pos, cur_step + 1, 1))
                        
                else:
                    # Branch shared by both players with same distance, add to both
                    if next_pos in visited_p1 or next_pos in visited_p0 or (next_pos[0] == my_pos[0] and next_pos[1] == my_pos[1]) or (next_pos[0] == adv_pos[0] and next_pos[1] == adv_pos[1]):
                        continue
                    visited_p1.add(next_pos)
                    visited_p0.add(next_pos)
                    distances[next_pos[0]][next_pos[1]] = cur_step + 1
                    state_queue.append((next_pos, cur_step + 1, 2))
        
        return len(visited_p0) - len(visited_p1)
                            
    
    def possibleMoves(self, chess_board, my_pos, adv_pos, max_step):
        """ 
        Generate the list of all possible moves and return them. Uses BFS to find the available moves.
        """
        s = ((-1, 0), (0, 1), (1, 0), (0, -1))
        moves = []
        state_queue = deque([(my_pos, 0)])
        visited = set()
        visited.add((my_pos[0], my_pos[1]))
        while state_queue:
                cur_pos, cur_step = state_queue.popleft()
                r = cur_pos[0]
                c = cur_pos[1]
                for dir, move in enumerate(s):
                    if chess_board[r, c, dir]:
                        continue
                    moves.append((cur_pos, dir))
                    next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
                    if next_pos in visited or cur_step == max_step or (next_pos[0] == adv_pos[0] and next_pos[1] == adv_pos[1]):
                        continue
                    visited.add(next_pos)
                    state_queue.append((next_pos, cur_step + 1))
                            
        return moves
    
    def possibleMoves2(self, chess_board, my_pos, adv_pos, max_step):
        """ 
        Generate the list of possible moves and return a subset of them which includes moves that are close to an existing wall 
        or that move the player the longest distance that it can. Uses BFS to find the available moves.
        """
        s = ((-1, 0), (0, 1), (1, 0), (0, -1))
        moves = []
        state_queue = deque([(my_pos, 0)])
        visited = set()
        visited.add((my_pos[0], my_pos[1]))
        while state_queue:
            cur_pos, cur_step = state_queue.popleft()
            r = cur_pos[0]
            c = cur_pos[1]
            
            edges_0 = sum(chess_board[r][c]) # Walls directly around player (including border of the board)
            edges_1 = 0 # Walls directly around player (excluding border of the board)
            edges_2 = 0 # Walls adjacent to walls around player
            if r > 0:
                edges_2 += chess_board[r-1][c][1] + chess_board[r-1][c][3]
                edges_1 += chess_board[r][c][0]
            if r < self.board_size - 1:
                edges_2 += chess_board[r+1][c][1] + chess_board[r+1][c][3]
                edges_1 += chess_board[r][c][2]
            if c > 0:
                edges_2 += chess_board[r][c-1][0] + chess_board[r][c-1][2]
                edges_1 += chess_board[r][c][3]
            if c < self.board_size - 1:
                edges_2 += chess_board[r][c+1][0] + chess_board[r][c+1][2]
                edges_1 += chess_board[r][c][1]
            
            for dir, move in enumerate(s):
                if chess_board[r, c, dir]:
                    continue
                
                if edges_0 < 2 and edges_1 + edges_2 > 0:
                    moves.append((cur_pos, dir))
                elif cur_step == max_step:
                    moves.append((cur_pos, dir))
                
                next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
                if next_pos in visited or cur_step == max_step or (next_pos[0] == adv_pos[0] and next_pos[1] == adv_pos[1]):
                    continue
                visited.add(next_pos)
                state_queue.append((next_pos, cur_step + 1))
                            
        if len(moves) == 0:
            # If no moves found with this method return all possible moves (Don't think its possible to happen but just in case)
            return self.possibleMoves(chess_board, my_pos, adv_pos, max_step)
        return moves
    
    
    def check_endgame(self, board, p0, p1):
        """ 
        Checks whether the game is over in the given input state. 
        If it is return true, as well as each player's score, otherwise return false.
        """
        # Check if there is a path connecting the two players
        if self.players_connected(board, p0, p1):
            return False, 0, 0
        
        def calculate_score(board, p0):
            """ 
            Calculates the score of a player in the given board assuming that the game has ended.
            """
            s = ((-1, 0), (0, 1), (1, 0), (0, -1))
            queue = deque([p0])
            visited = set()
            visited.add((p0[0], p0[1]))
            while queue:
                cur_pos = queue.popleft()
                r = cur_pos[0]
                c = cur_pos[1]
                for dir, move in enumerate(s):
                    if board[r, c, dir]:
                        continue
                    next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
                    if next_pos in visited:
                        continue
                    visited.add(next_pos)
                    queue.append(next_pos)
                    
            return len(visited)
        
        # Calculate each player's score and return them
        p0_score = calculate_score(board, p0)
        p1_score = calculate_score(board, p1)

        return True, p0_score, p1_score
    
    def players_connected(self, board, p0, p1):
        """
        Checks whether there is a path connecting the two players in the given board.
        Returns true if there is one, or false otherwise.
        """
        moves = self.moves
        queue = []
        heapq.heappush(queue, (0, p0))
        visited = set()
        visited.add((p0[0], p0[1]))
        while queue:
            _, cur_pos = heapq.heappop(queue)
            if cur_pos[0] == p1[0] and cur_pos[1] == p1[1]:
                return True
            r, c = cur_pos
            for dir, move in enumerate(moves):
                if board[r, c, dir]:
                    continue
                next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
                if next_pos in visited:
                    continue
                visited.add(next_pos)
                priority = abs(next_pos[0] - p1[0]) + abs(next_pos[1] - p1[1])
                heapq.heappush(queue, (priority, next_pos))
        return False
        
    
    
    
    
    
    
    
    
