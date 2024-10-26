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


@register_agent("student_agent13")
class StudentAgent13(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent13, self).__init__()
        self.name = "StudentAgent13"
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

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        self.start_time = time.time()
        
        self.time0 = 0
        self.time1 = 0
        self.time2 = 0
        self.time3 = 0
        self.time4 = 0
        
        best_move = self.possibleMoves2(chess_board, my_pos, adv_pos, max_step)[-1]
        
        for i in range(1, 20):
            try:
                best_value, new_best_move = self.max_value(chess_board, my_pos, adv_pos, max_step, depth=i, alpha=float("-inf"), beta=float("inf"))
                best_move = new_best_move
            except RuntimeError:
                print("agent 11", i, time.time() - self.start_time) #, self.time0, self.time1, self.time2, self.time3, self.time4)
                return best_move[0], best_move[1]
        
        return best_move[0], best_move[1]
        
    def max_value(self, chess_board, my_pos, adv_pos, max_step, depth, alpha, beta):
        if time.time() - self.start_time > 1.95:
            raise RuntimeError("2 second time limit reached")

        value = self.evaluation(chess_board, my_pos, adv_pos, max_step, depth)
        if value != None:
            return value, None
        
        p = self.possibleMoves2(chess_board, my_pos, adv_pos, max_step)
        
        best_move = None
        
        for i in reversed(p):
            r, c, dir = i[0][0], i[0][1], i[1]
            board = deepcopy(chess_board)
            board[r, c, dir] = True
            move = self.moves[dir]
            board[r + move[0], c + move[1], self.opposites[dir]] = True
            
            min_value, min_move = self.min_value(board, adv_pos, (r, c), max_step, depth - 1, alpha, beta)
            
            if min_value > alpha:
                alpha = min_value
                best_move = i
            if alpha >= beta:
                return beta, i
            
        return alpha, best_move
    
    def min_value(self, chess_board, my_pos, adv_pos, max_step, depth, alpha, beta):
        if time.time() - self.start_time > 1.95:
            raise RuntimeError("2 second time limit reached")
        
        value = self.evaluation(chess_board, adv_pos, my_pos, max_step, depth)
        if value != None:
            return value, None
        p = self.possibleMoves2(chess_board, my_pos, adv_pos, max_step)
        
        best_move = None
        
        for i in p:
            r, c, dir = i[0][0], i[0][1], i[1]
            board = deepcopy(chess_board)
            board[r, c, dir] = True
            move = self.moves[dir]
            board[r + move[0], c + move[1], self.opposites[dir]] = True
            
            max_value, max_move = self.max_value(board, adv_pos, (r, c), max_step, depth - 1, alpha, beta)
            if max_value < beta:
                beta = max_value
                best_move = max_move
            if alpha >= beta:
                return alpha, i
        
        return beta, best_move
    
    def evaluation(self, chess_board, my_pos, adv_pos, max_step, depth):
        # Game over, check what player won
        #t = time.time()
        
        connected, dist = self.players_connected_with_distance(chess_board, my_pos, adv_pos)
        
        if not connected:
                res, p0, p1 = self.check_endgame(chess_board, my_pos, adv_pos)
                if p0 > p1:
                    return 1000000
                elif p0 < p1:
                    return -1000000
                else:
                    return -20


        
        # Reached max depth of the tree
        if depth == 0:
            moves_val = len(self.possibleMoves(chess_board, my_pos, adv_pos, max_step)) - len(self.possibleMoves(chess_board, adv_pos, my_pos, max_step))
            #t = time.time()
            #control_val = self.check_player_board_control(chess_board, my_pos, adv_pos)
            #self.time1 += time.time() - t
            dist_val = max(0, max_step * 2 - dist)
            return moves_val + dist_val
            
        # Go to the next level of the tree
        return None
    
    def check_player_board_control(self, chess_board, my_pos, adv_pos):
        #t = time.time()
        distances = np.zeros((self.board_size, self.board_size))
        #self.time2 += time.time() - t
        #t = time.time()
        
        # Add distance from self to each square on the board
        s = ((-1, 0), (0, 1), (1, 0), (0, -1))
        state_queue = deque([(my_pos, 0)])
        visited = set()
        visited.add((my_pos[0], my_pos[1]))
        while state_queue:
                cur_pos, cur_step = state_queue.popleft()
                r = cur_pos[0]
                c = cur_pos[1]
                distances[r][c] += cur_step
                for dir, move in enumerate(s):
                    if chess_board[r, c, dir]:
                        continue
                    next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
                    if next_pos in visited or (next_pos[0] == adv_pos[0] and next_pos[1] == adv_pos[1]):
                        continue
                    visited.add(next_pos)
                    state_queue.append((next_pos, cur_step + 1))
                    
        # Subtract distance from opponent to each square on the board
        state_queue = deque([(adv_pos, 0)])
        visited = set()
        visited.add((adv_pos[0], adv_pos[1]))
        while state_queue:
                cur_pos, cur_step = state_queue.popleft()
                r = cur_pos[0]
                c = cur_pos[1]
                distances[r][c] -= cur_step
                for dir, move in enumerate(s):
                    if chess_board[r, c, dir]:
                        continue
                    next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
                    if next_pos in visited or (next_pos[0] == my_pos[0] and next_pos[1] == my_pos[1]):
                        continue
                    visited.add(next_pos)
                    state_queue.append((next_pos, cur_step + 1))
                    
        #self.time3 += time.time() - t
        #t = time.time()
        
        res = np.sum(distances < 0) - np.sum(distances > 0)
        
        #self.time4 += time.time() - t
        
        return res
    
                            
    
    def possibleMoves(self, chess_board, my_pos, adv_pos, max_step):
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
        s = ((-1, 0), (0, 1), (1, 0), (0, -1))
        moves = []
        state_queue = deque([(my_pos, 0)])
        visited = set()
        visited.add((my_pos[0], my_pos[1]))
        while state_queue:
                cur_pos, cur_step = state_queue.popleft()
                r = cur_pos[0]
                c = cur_pos[1]
                
                edges_0 = sum(chess_board[r][c])
                edges_1 = 0
                #edges_1 = sum(chess_board[r][c])
                edges_2 = 0
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
            return self.possibleMoves(chess_board, my_pos, adv_pos, max_step)
        return moves
    
    
    def check_endgame(self, board, p0, p1):
        #if self.players_connected(board, p0, p1):
        #    return False, 0, 0
        

        def calculate_score(board, p0):
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
        
        p0_score = calculate_score(board, p0)
        p1_score = calculate_score(board, p1)

        return True, p0_score, p1_score
    
    def players_connected(self, board, p0, p1):
        # t = time.time()
        moves = self.moves
        queue = []
        heapq.heappush(queue, (0, p0))
        visited = set()
        visited.add((p0[0], p0[1]))
        while queue:
            _, cur_pos = heapq.heappop(queue)
            if cur_pos[0] == p1[0] and cur_pos[1] == p1[1]:
                # self.time4 += time.time() - t
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
        # self.time4 += time.time() - t
        return False
    

    def players_connected_with_distance(self, board, p0, p1):
        moves = self.moves
        queue = []
        heapq.heappush(queue, (0, p0))
        visited = set()
        visited.add((p0[0], p0[1]))
        distances = {p0: 0}  # Store distances from p0 to other positions
    
        while queue:
            cur_dist, cur_pos = heapq.heappop(queue)
            if cur_pos[0] == p1[0] and cur_pos[1] == p1[1]:
                return True, distances[p1]  # Return True and distance if connected
            r, c = cur_pos
    
            for dir, move in enumerate(moves):
                if board[r, c, dir]:
                    continue
                next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
                if next_pos in visited:
                    continue
                visited.add(next_pos)
                distances[next_pos] = distances[cur_pos] + 1
                priority = distances[next_pos] + abs(next_pos[0] - p1[0]) + abs(next_pos[1] - p1[1])
                heapq.heappush(queue, (priority, next_pos))
    
        return False, None  # Return False and None if not connected

        
    
    
    
    
    
    
    
    
