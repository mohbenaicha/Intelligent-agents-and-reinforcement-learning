import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import namedtuple
from IPython.display import display
from enum import Enum
from pomegranate import *
from pit_wumpus_networks import *

Agent, Pit, Wumpus, Gold, Breeze, Stench = "A", "P", "W", "G", "B", "S"


class Environment:
    def __init__(self, pitprob, addwumpus=True):
        self.pit_prob = pitprob
        self.allowclimbwithoutgold = True
        self.addwumpus = addwumpus
        self.EnvSize = 4
        self.getIndexes()
        self.getEnv()
        self.startloc = self.setAgent(getloc=True)
        self.pits = []
        self.get_pit_loc()

    def getMatrix(self, item):
        matrix = []
        try:
            copy = getattr(item, 'copy')
        except AttributeError:
            copy = None
        for i in range(self.EnvSize):
            matrix.append([])
            for j in range(self.EnvSize):
                if copy:
                    matrix[i].append(copy())
                else:
                    matrix[i].append(item)
        return matrix

    def getIndexes(self):
        Indexes = []
        for x in range(self.EnvSize):
            for y in range(self.EnvSize):
                Indexes.append((x, y))
        return Indexes

    def setElement(self, index, value):
        x, y = index
        env[x][y] = value
        # Removes location from Indexes after it is used.
        Indexes.remove(index)

    def randomize_pits(self):
        for index in set(Indexes):
            setPit = np.random.binomial(1, self.pit_prob, 1)
            if setPit == 1:
                self.setElement(index, Pit)

    def get_pit_loc(self):
        pit_loc = np.where(self.environment == "P")
        [self.pits.append((pit_loc[0][i], pit_loc[1][i])) for i in range(len(pit_loc[0]))]

    def setWumpus(self):
        index = random.choice(Indexes)
        self.setElement(index, Wumpus)
        self.wumpusloc = index

    def setGold(self):
        index = random.choice(Indexes)
        self.setElement(index, Gold)
        self.goldloc = index

    def setAgent(self, getloc=False):
        index = (3, 0)
        if getloc:
            return index
        else:
            self.setElement(index, Agent)
            self.startloc = index

    def refreshGlobals(self):
        global Indexes
        global env
        Indexes = self.getIndexes()
        env = self.getMatrix(0)
        return Indexes, env

    def getEnv(self):
        """ Returns a new Wumpus World environment """
        self.refreshGlobals()
        self.setAgent()
        self.setGold()
        if self.addwumpus:
            self.setWumpus()
        else:
            self.wumpusloc = None
            print("👾 Wumpus not spawned 👾")
        self.randomize_pits()
        self.environment = np.array(env)

    def printEnv(self, target, agentsteps, spath=None, final_state=False, returnpath=False):
        state = pd.DataFrame(index=["0", "1", "2", "3"], columns=["0", "1", "2", "3"])
        if final_state:
            for step in agentsteps:
                state.iloc[step] = "👣"
        else:
            state.iloc[agentsteps[-1]] = "👣"
        for pit in self.pits:
            state.iloc[pit] = "🕳️"
        if self.goldloc: state.iloc[self.goldloc] = "💰"
        if self.wumpusloc: state.iloc[self.wumpusloc] = "👾"
        if returnpath:
            try:
                for step in spath:
                    state.iloc[step] = "🦶"
            except:
                pass
        state = state.fillna(".")
        print(
            "\nAgent bee line: 👣 |  Shortest route to {}: 🦶\nGold: 💰 | Wumpus: 👾 | Pits: 🕳️ | Unattended: .\n".format(
                target))

        return state


percept_mapping = {Pit: "B", Wumpus: "S"}
items = {1: "Agent", 2: "Pit", 3: "Wumpus", 4: "Gold", 5: "Breeze", 6: "Stench"}


class Percepts(Environment):
    def __init__(self, pitprob, addwumpus):
        super().__init__(pitprob, addwumpus)
        self.wumpuscry = False

        self.percepts = self.getMatrix(set())  # empty sets containing all state information of each room

        Indexes = self.getIndexes()
        for x, y in Indexes:
            if self.environment[x, y] and self.environment[x, y] != "A":
                self.addPercept((x, y), self.environment[x, y])

            if (x + 1, y) in Indexes:
                num = percept_mapping.get(self.environment[x + 1, y], None)
                if num:
                    self.addPercept((x, y), num)

            if (x - 1, y) in Indexes:
                num = percept_mapping.get(self.environment[x - 1, y], None)
                if num:
                    self.addPercept((x, y), num)

            if (x, y + 1) in Indexes:
                num = percept_mapping.get(self.environment[x, y + 1], None)
                if num:
                    self.addPercept((x, y), num)

            if (x, y - 1) in Indexes:
                num = percept_mapping.get(self.environment[x, y - 1], None)
                if num:
                    self.addPercept((x, y), num)

        self.percept_hist = [list(self.getPercept(self.startloc[0], self.startloc[1])), ]

    def addPercept(self, index, num):
        x, y = index
        self.percepts[x][y].add(num)

    def getPercepts(self):
        return self.percepts

    def getPercept(self, x, y):
        return (self.percepts[x][y])


class KB(Percepts):
    def __init__(self, pitprob, addwumpus):
        super().__init__(pitprob, addwumpus)

        self.loc_path = [(self.startloc), ]
        self.curr_dir = [1, ]  # 0, 1, 2, 3 - up, right, down, left
        self.action = ["Forward", "Tright", "Tleft", "Shoot", "Grab", "Climb"]
        self.haveGold = False
        self.dead = False
        self.haveArrow = True
        self.arrow_path = []
        self.score = 0
        self.moves = 0

    def get_arrow_path(self):
        if self.haveArrow:
            if self.curr_dir[-1] == 0:  # Up
                for i in range(self.loc_path[-1][0]):
                    self.arrow_path.append((self.loc_path[-1][0] - i - 1, self.loc_path[-1][1]))
            if self.curr_dir[-1] == 1:  # Right
                for i in range(len(env) - (self.loc_path[-1][1]) - 1):
                    self.arrow_path.append((self.loc_path[-1][0], self.loc_path[-1][1] + i + 1))
            if self.curr_dir[-1] == 2:  # Down
                for i in range(len(env) - self.loc_path[-1][0] - 1):
                    self.arrow_path.append((i, self.loc_path[-1][1]))
            if self.curr_dir[-1] == 3:  # Left
                for i in range(self.loc_path[-1][1]):
                    self.arrow_path.append((self.loc_path[-1][0], i))

    def get_shortest_route(self, source, target, intermediate):
        '''
        Note: this includes turns.
        '''
        G = nx.Graph()
        # Adding nodes
        if intermediate:
            nodes = []
            [nodes.append(i) for i in self.loc_path]
            nodes.append(target)

        else:
            nodes = list(set(self.loc_path))

        for a, (i, j) in enumerate(nodes):
            G.add_node(a, coordinate=(i, j))

        # mapping nodes to coordinates
        dictionary = {}
        for e in range(len(G.nodes)):
            x = G.nodes[e]['coordinate']
            dictionary[x] = e

        # adding edges
        edges = []

        # Connecting nodes for edges
        for i in range(10 ** (self.EnvSize - 1)):
            a = random.choice(nodes)
            b = random.choice(nodes)
            if a[0] == b[0] and (np.absolute(a[1] - b[1]) == 1):
                edges.append((a, b))
            elif a[1] == b[1] and (np.absolute(a[0] - b[0]) == 1):
                edges.append((a, b))
        edges = set(edges)

        # generating edges
        edge_list = []
        for i, j in edges:
            x, y = dictionary.get(i), dictionary.get(j)
            edge_list.append((x, y))

        for i in edge_list:
            G.add_edge(*i)

        # remapping nodes to coordinates
        all_paths = nx.all_simple_paths(G, source=dictionary.get(source), target=dictionary.get(target))
        paths = []
        for i in all_paths:
            paths.append(i)

        dictionaries = []
        [dictionaries.append(dict()) for i in range(len(paths))]

        for i in range(len(dictionaries)):
            for j in paths[i]:
                x = G.nodes[j]['coordinate']
                dictionaries[i][j] = x

        paths_back = []
        [paths_back.append(list()) for i in range(len(paths))]

        for i in range(len(dictionaries)):
            for j in paths[i]:
                x, y = dictionaries[i].get(j)
                paths_back[i].append((x, y))

        lengths = []
        for i in range(len(dictionaries)):
            lengths.append(len(paths_back[i]))
            for j in range(len(paths_back[i])):
                try:
                    if (paths_back[i][j + 1][0] == paths_back[i][j][0]) and (
                            paths_back[i][j + 1][1] != paths_back[i][j][1]):
                        if (paths_back[i][j + 2][0] != paths_back[i][j + 1][0]):
                            lengths[i] += 1
                    elif (paths_back[i][j + 1][1] == paths_back[i][j][1]) and (
                            paths_back[i][j + 1][0] != paths_back[i][j][0]):
                        if (paths_back[i][j + 2][1] != paths_back[i][j + 1][1]):
                            lengths[i] += 1
                    else:
                        pass
                except:
                    pass

        # Original paths without turns
        lengths2 = []
        for i in range(len(dictionaries)):
            lengths2.append(len(paths_back[i]))
        #         shortest path back considering turns
        try:
            shortest_path_back = paths_back[lengths.index(min(lengths))]
        except:
            shortest_path_back = [self.loc_path[-1]]
        # number of turns
        try:
            no_turns = lengths[lengths.index(min(lengths))] - lengths2[lengths.index(min(lengths))]
        except:
            no_turns = 0
        # execute path
        try:
            [self.loc_path.append(shortest_path_back[i + 1]) for i in range(len(shortest_path_back) - 1)]
        except:
            pass
        return shortest_path_back, no_turns


class Actuators(KB):
    def __init__(self, pitprob, addwumpus, verbose):
        super().__init__(pitprob, addwumpus)
        self.verbose = verbose

    def forward(self, direction):
        if direction == 0:  # facing up
            self.loc_path.append((self.loc_path[-1][0] - 1, self.loc_path[-1][1]))
        elif direction == 1:  # facing right
            self.loc_path.append((self.loc_path[-1][0], self.loc_path[-1][1] + 1))
        elif direction == 2:  # facing down
            self.loc_path.append((self.loc_path[-1][0] + 1, self.loc_path[-1][1]))
        else:  # facing left
            self.loc_path.append((self.loc_path[-1][0], self.loc_path[-1][1] - 1))
        self.score -= 1
        if self.verbose:
            print("Current score: {}".format(self.score))
        try:
            self.percept_hist.append(list(self.getPercept(self.loc_path[-1][0], self.loc_path[-1][1])))
        except:
            pass

    def bump(self):
        if len(self.loc_path) > 1:
            self.loc_path.pop(-1)
        if self.verbose:
            print("Agent attempts to move, bumps into a wall.")
            print("Current score: {}".format(self.score))

    def turnright(self, direction):
        if direction < 3:
            self.curr_dir.append(self.curr_dir[-1] + 1)
        else:
            self.curr_dir.append(0)
        self.score -= 1
        if self.verbose:
            print("Agent turns right.")
            print("Current score: {}".format(self.score))

    def turnleft(self, direction):
        if direction > 0:
            self.curr_dir.append(self.curr_dir[-1] - 1)
        else:
            self.curr_dir.append(3)
        self.score -= 1
        if self.verbose:
            print("Agent turns left.")
            print("Current score: {}".format(self.score))

    def shoot(self, probagent):
        arrow_path = self.arrow_path
        print("🏹🏹 Agent fires arrow 🏹🏹")
        if not probagent: # for non-probablistic agent
            if self.wumpusloc and (self.wumpusloc in arrow_path):
                self.wumpuscry = True
                self.wumpusloc = None
                print("Wumpus killed!")
            else:
                print("Arrow misses its mark")
        else: # for probabilistic agent
            self.wumpuscry = True
            self.wumpusloc = None
            print("Wumpus killed!")
        self.haveArrow = False
        self.score -= 10

    def grab(self, probagent):
        if self.verbose: print("Agent attempts to grab something.")
        if not probagent:
            if (self.goldloc == self.loc_path[-1]) and not self.haveGold:
                self.haveGold = True
                print("💰💰Agent picks up the gold💰💰")
                self.goldloc = None
            else:
                if self.verbose:
                    print("Agent grabs nothing.")
                    print("Current score: {}".format(self.score))
        else: # for probabilistic agent
            self.haveGold = True
            print("💰💰Agent picks up the gold💰💰")
            self.goldloc = None

        self.score -= 1

    def climb(self, allowclimbwithoutgold=True):
        if not allowclimbwithoutgold:
            if self.havegold:
                print("Agent climbs out with the gold")
                self.score += (1000 - 1)
            else:
                if self.verbose:
                    print("Agent attempts to climb out but does not have the gold.")

        else:
            if self.haveGold:
                self.score += 1000
                print("🎉🎉Agent climbs out of the cave with the gold.🎉🎉")
            else:
                print("Agent climbs out without the gold.")
        self.score -= 1
        print("Final score: ", self.score)

    def make_shortest_turn(self, pathback):  # note the use of a numpy array means indexes for rows are inversed

        current = pathback[0]
        try:
            nextone = pathback[1]
        except:
            nextone = current

        if nextone != current:
            if current[0] != nextone[0] and current[1] == nextone[1]:
                if current[0] > nextone[0]:  # if shortest way back starts above agent

                    if self.curr_dir[-1] == 1:
                        self.turnleft(direction=1)
                    elif self.curr_dir[-1] == 3:
                        self.turnright(direction=3)
                    elif self.curr_dir[-1] == 2:  # if agent facing down, turn right twice  (or left, it doesn't matter)
                        self.turnright(direction=2), self.turnright(direction=3)
                    else:
                        pass

                elif current[0] < nextone[0]:  # if shortest way back starts below agent

                    if self.curr_dir[-1] == 1:
                        self.turnright(direction=1)
                    elif self.curr_dir[-1] == 3:
                        self.turnleft(direction=3)
                    elif self.curr_dir[-1] == 0:  # if agent facing up, turn right twice (or left, it doesn't matter)
                        self.turnright(direction=0), self.turnright(direction=1)
                    else:
                        pass
                else:
                    pass

            elif current[0] == nextone[0] and current[1] != nextone[1]:

                if current[1] > nextone[1]:  # if shortest way back starts left of agent

                    if self.curr_dir[-1] == 0:
                        self.turnleft(direction=0)
                    elif self.curr_dir[-1] == 2:
                        self.turnright(direction=2)
                    elif self.curr_dir[-1] == 1:  # if agent facing right, turn right twice (or left, it doesn't matter)
                        self.turnright(direction=1), self.turnright(direction=2)
                    else:
                        pass

                elif current[1] < nextone[1]:  # if shortest way back starts right of agent

                    if self.curr_dir[-1] == 0:
                        self.turnright(direction=0)
                    elif self.curr_dir[-1] == 2:
                        self.turnleft(direction=2)
                    elif self.curr_dir[-1] == 3:  # if agent facing left, turn right twice (or left, it doesn't matter)
                        self.turnright(direction=3), self.turnright(direction=0)
                    else:
                        pass
                else:
                    pass
            else:
                pass