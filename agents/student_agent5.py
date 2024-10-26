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

@register_agent("student_agent5")
class StudentAgent5(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent5, self).__init__()
        self.name = "StudentAgent5"
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
        start_time = time.time()
        
        p = self.possibleMoves2(chess_board, my_pos, adv_pos, max_step)
        n = len(p)
        wins = np.zeros(n)
        losses = np.zeros(n)
        
        total = 0
        
        self.time0 = 0; # Add barrier
        self.time1 = 0; # Check endgame
        self.time2 = 0; # Possible moves
        self.time3 = 0; # Random choice
        self.time4 = 0; # Players connected
        
        
        while (time.time() - start_time < 1.9):
            i = random.randint(0, n-1)
            pos, dir = p[i]
            result = self.run_simulation(pos, dir, deepcopy(chess_board), tuple(adv_pos), max_step)
            
            if result:
                wins[i] += 1
            else:
                losses[i] += 1
                
            total += 1
        
        print("AGENT 4")
        print(total)
        print(self.time0, "    |    ", self.time1, "    |    ", self.time2, "    |    ", self.time3, "    |    ", self.time4)
            
        max_i = -1;
        max_win_rate = -1;
        for i in range(n):
            if wins[i] == 0:
                continue
            win_rate = wins[i] / (wins[i] + losses[i])
            if win_rate > max_win_rate:
                max_win_rate = win_rate
                max_i = i          
        
        pos, dir = p[max_i]

        return pos, dir
    
    
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
                edges_2 = np.zeros(4)
                if r > 0:
                    edges_2[0] += chess_board[r-1][c][1] + chess_board[r-1][c][3]
                    edges_2[1] += chess_board[r-1][c][1]
                    edges_2[3] += chess_board[r-1][c][3]
                    edges_1 += chess_board[r][c][0]
                if r < self.board_size - 1:
                    edges_2[2] += chess_board[r+1][c][1] + chess_board[r+1][c][3]
                    edges_2[1] += chess_board[r+1][c][1]
                    edges_2[3] += chess_board[r+1][c][3]
                    edges_1 += chess_board[r][c][2]
                if c > 0:
                    edges_2[3] += chess_board[r][c-1][0] + chess_board[r][c-1][2]
                    edges_2[0] += chess_board[r][c-1][0]
                    edges_2[2] += chess_board[r][c-1][2]
                    edges_1 += chess_board[r][c][3]
                if c < self.board_size - 1:
                    edges_2[1] += chess_board[r][c+1][0] + chess_board[r][c+1][2]
                    edges_2[0] += chess_board[r][c+1][0]
                    edges_2[2] += chess_board[r][c+1][2]
                    edges_1 += chess_board[r][c][1]
                
                for dir, move in enumerate(s):
                    if chess_board[r, c, dir]:
                        continue
                    
                    if edges_0 < 2 and edges_1 + edges_2[dir] > 0:
                        moves.append((cur_pos, dir))
                                
                            
                    next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
                    if next_pos in visited or cur_step == max_step or (next_pos[0] == adv_pos[0] and next_pos[1] == adv_pos[1]):
                        continue
                    visited.add(next_pos)
                    state_queue.append((next_pos, cur_step + 1))
                            
        if len(moves) == 0:
            return self.possibleMoves(chess_board, my_pos, adv_pos, max_step)
        return moves
    
    
    def run_simulation(self, my_pos, dir, board, adv_pos, max_step):
        
        t = time.time()
        
        # add barrier
        r, c = my_pos
        board[r, c, dir] = True
        move = self.moves[dir]
        board[r + move[0], c + move[1], self.opposites[dir]] = True
        
        self.time0 += time.time() - t
        t = time.time()
        
        res, p0, p1 = self.check_endgame(board, my_pos, adv_pos)
        if res:
            self.time1 += time.time() - t
            return p0 > p1
        
        self.time1 += time.time() - t
        t = time.time()
        
        my_turn = False
    
        while True:
            if my_turn:
                    p = self.possibleMoves(board, my_pos, adv_pos, max_step)
                    self.time2 += time.time() - t
                    t = time.time()
                    pos_, dir_ = random.choice(p)
                    
                    self.time3 += time.time() - t
                    t = time.time()
                    
                    r, c = pos_
                    board[r, c, dir_] = True
                    move = self.moves[dir_]
                    board[r + move[0], c + move[1], self.opposites[dir_]] = True
                    
                    self.time0 += time.time() - t
                    t = time.time()
                    
                    my_pos = pos_
                    res, p0, p1 = self.check_endgame(board, my_pos, adv_pos)
                    if res:
                        self.time1 += time.time() - t
                        return p0 > p1
                    
                    self.time1 += time.time() - t
                    t = time.time()
                    
                    my_turn = False

            else:
                    p = self.possibleMoves(board, adv_pos, my_pos, max_step)
                    self.time2 += time.time() - t
                    t = time.time()
                    pos_, dir_ = random.choice(p)
                    
                    self.time3 += time.time() - t
                    t = time.time()
                    
                    r, c = pos_
                    board[r, c, dir_] = True
                    move = self.moves[dir_]
                    board[r + move[0], c + move[1], self.opposites[dir_]] = True
                    
                    self.time0 += time.time() - t
                    t = time.time()
                    
                    adv_pos = pos_
                    res, p0, p1 = self.check_endgame(board, my_pos, adv_pos)
                    if res:
                        self.time1 += time.time() - t
                        return p0 > p1
                    
                    self.time1 += time.time() - t
                    t = time.time()
                    
                    my_turn = True

                    
    
    
    def check_endgame(self, board, p0, p1):
        if self.players_connected(board, p0, p1):
            return False, 0, 0
        
        # Union-Find
        father = dict()
        for r in range(self.board_size):
            for c in range(self.board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(self.board_size):
            for c in range(self.board_size):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(self.board_size):
            for c in range(self.board_size):
                find((r, c))
        p0_r = find(tuple(p0))
        p1_r = find(tuple(p1))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            win_blocks = p1_score
        else:
            player_win = -1  # Tie
        return True, p0_score, p1_score
    
    def players_connected(self, board, p0, p1):
        t = time.time()
        moves = self.moves
        queue = []
        heapq.heappush(queue, (0, p0))
        visited = set()
        visited.add((p0[0], p0[1]))
        while queue:
            _, cur_pos = heapq.heappop(queue)
            if cur_pos[0] == p1[0] and cur_pos[1] == p1[1]:
                self.time4 += time.time() - t
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
        self.time4 += time.time() - t
        return False
        
    
    
    
    
    
    
    
    
