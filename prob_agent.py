from pit_wumpus_networks import *
from wumpusworld import *

class ProbAgent(Actuators):
    def __init__(self, pitprob, addwumpus, verbose):
        super().__init__(pitprob, addwumpus, verbose)
        self.pitprob = pitprob
        self.beliefs = self.populate_beliefs()
        self.confirmedpits = []
        self.confirmedwumpus = None
        self.get_beliefs()
        self.getPercepts()

    def populate_beliefs(self):
        beliefs = np.array(self.getMatrix(dict()))
        items = ["Ok", "Known", "Frontier", "Stench", "Wumpus", "Breeze", "Pit"]
        for item in items:
            for i in range(self.EnvSize):
                for j in range(self.EnvSize):
                    (beliefs[i, j])["{}".format(item)] = 0
        return beliefs

    def get_beliefs(self):

        # updating confirmed pits/no pits, ok/not ok locations, non-frontier locations,
        Indexes = self.getIndexes()
        if len(self.confirmedpits) > 0:  # and any(loc in adjacents for loc in self.confirmedpits):
            for loc in self.confirmedpits:
                (self.beliefs[loc])["Pit"] = 1
        for loc in self.loc_path:
            self.beliefs[loc]["Ok"] = 1
            self.beliefs[loc]["Frontier"] = 0
            self.beliefs[loc]["Pit"] = 0

        # Updating frontier, confirmed stench and breeze
        for loc in self.loc_path:
            i, j = loc[0], loc[1]

            percept = self.getPercept(i, j)
            if "S" in percept: self.beliefs[(i, j)]["Stench"] = 1
            if "B" in percept: self.beliefs[(i, j)]["Breeze"] = 1

            adjacents = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
            for a in adjacents:
                if (a not in self.loc_path) and (a in Indexes) and (self.beliefs[a]["Pit"] != 1):
                    (self.beliefs[a])["Frontier"] = 1

    def get_wumpus_proba(self, confirmedpits, stenchsensed, wumpustarget, wumpusalive):
        model = bake_wumpus_network(confirmedpits=confirmedpits, wumpusalive=wumpusalive, wumpusspawned=self.addwumpus)
        query, wumpusprobs = dict(), []
        for i in range(len(stenchsensed)):
            query["Stench{}{}".format(stenchsensed[i][0][0], stenchsensed[i][0][1])] = "{}".format(stenchsensed[i][1])
        dist = model.predict_proba([query])[0][0].parameters[0]

        for loc in (wumpustarget):
            prob = [(loc[0], loc[1]), dist["{},{}".format(loc[0], loc[1])]]
            wumpusprobs.append(prob)
        return wumpusprobs

    def get_pit_proba(self, wumpusloc, breezelocs, pitlocs, targetpit):
        if wumpusloc:
            model = bake_pit_network(wumpusloc=wumpusloc[0], pitprob=self.pitprob)
        else:
            model = bake_pit_network(wumpusloc=None, pitprob=self.pitprob)
        #         pits = self.getIndexes()
        pits = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3),
                (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]

        target, pitprobs, query = [], [], {}

        target.append([pits.index(i) for i in targetpit])
        for i in range(30):
            try:
                query["Breeze{}{}".format((breezelocs[i][0][0]),
                                          (breezelocs[i][0][1]))] = "{}".format(breezelocs[i][1])
                query["Pit{}{}".format((pitlocs[i][0][0]),
                                       (pitlocs[i][0][1]))] = "{}".format(pitlocs[i][1])
            except:
                pass
        for i, j in enumerate(target[0]):
            pitprobs.append(
                [((targetpit[i][0]), (targetpit[i][1])), model.predict_proba([query])[0][j].parameters[0].get("1")])
        return pitprobs

    def confirmed_pit_wumpus(self, pitprob, wumpusprob):
        # update beliefs if wumpus location is confirmed

        if (not self.wumpuscry) and (not self.confirmedwumpus):
            for loc in wumpusprob:
                if loc[1] == 1:
                    self.beliefs[loc[0]]["Wumpus"] = 1
                    self.confirmedwumpus = 1
                else:
                    pass
        # update beliefs if pit location is confirmed
        for loc in pitprob:
            if loc[1] == 1:
                self.beliefs[loc[0]]["Pit"] = 1
                self.confirmedpits.append(loc[0])

    def shoot_wumpus(self, wumpus):
        candidates, distances = [], []
        # get possible points form which wumpus may be shot at
        for loc in set(self.loc_path):
            try:
                if loc[0] == self.confirmedwumpus[0][0] or loc[1] == self.confirmedwumpus[0][1]:
                    candidates.append(loc)
            except:
                pass
        # get closes of above points to current agent location
        for loc in candidates:
            euclidist = ((self.loc_path[-1][0] - loc[0]) ** 2 + (self.loc_path[-1][1] - loc[1]) ** 2) ** 0.5
            distances.append(euclidist)
        elected = candidates[np.argmin(distances)]
        if self.verbose:
            print("Agent confirms existence of wumpus. Agent is currently at {} and moved to {} to shoot the wumpus."
                  .format(self.loc_path[-1], elected))
        path = self.get_shortest_route(source=self.loc_path[-1], target=elected, intermediate=True)  # get shortest path
        self.make_shortest_turn(pathback=path[0])  # turn to shortest path
        self.make_shortest_turn(pathback=[self.loc_path[-1], wumpus[0]])  # turn to face wumpus
        if self.verbose:
            display(self.printEnv(agentsteps=self.loc_path, spath=path[0], final_state=True, returnpath=True,
                                  target=elected))
            print("Agent facing:", self.curr_dir[-1], "\n [0: Up | 1: Right | 2: Down | 3: Left]")
        self.shoot(probagent=True)

        # updating sense map and beliefs to remove wumpus and stench and mark wumpus loc as Ok
        Indexes = self.getIndexes()
        for loc in Indexes:
            self.beliefs[loc]["Wumpus"] = 0
            self.beliefs[loc]["Stench"] = 0
            self.percepts[loc[0]][loc[1]].discard("W")
            self.percepts[loc[0]][loc[1]].discard("S")
        self.beliefs[self.confirmedwumpus[0]]["Ok"] = 1

    def decide_on_action(self, risk):
#         if self.verbose:
#             print("-" * 10, "Current environment", "-" * 10)
#             display(
#                 self.printEnv(agentsteps=self.loc_path, spath=None, final_state=False, returnpath=False, target=None))
#             display("Percepts: \n", pd.DataFrame(self.percepts))

        targets, pits, breezes, stenches, candidates, elected = [], [], [], [], [], []
        self.get_beliefs()

        Indexes = self.getIndexes()
        for loc in Indexes:
            if self.beliefs[loc]["Frontier"] == 1 and loc not in self.confirmedpits: targets.append(loc)
            if self.beliefs[loc]["Pit"] == 1: pits.append([loc, 1])
        for loc in set(self.loc_path):
            if self.beliefs[loc]["Breeze"] == 1:
                breezes.append([loc, 1])
            else:
                breezes.append([loc, 0])
            if self.beliefs[loc]["Stench"] == 1:
                stenches.append([loc, 1])
            else:
                stenches.append([loc, 0])
            if self.beliefs[loc]["Pit"] == 0: pits.append([loc, 0])

        pitproba = self.get_pit_proba(wumpusloc=self.confirmedwumpus, breezelocs=breezes, pitlocs=pits,
                                      targetpit=targets)
        wumpusproba = self.get_wumpus_proba(confirmedpits=self.confirmedpits, stenchsensed=stenches,
                                            wumpustarget=targets, wumpusalive=(False == self.wumpuscry))

        if self.verbose: display("Wumpus probabilities:", wumpusproba)

        if not self.wumpuscry:
            for loc in wumpusproba:
                if loc[1] == 1:
                    self.confirmedwumpus = [loc[0]]
                    self.shoot_wumpus(self.confirmedwumpus)
                    self.wumpuscry = True
                    if self.verbose: display(np.array(self.percepts))
            # Update probability if wumpus killed:
            wumpusproba = self.get_wumpus_proba(confirmedpits=self.confirmedpits, stenchsensed=stenches,
                                                wumpustarget=targets, wumpusalive=(False == self.wumpuscry))
        self.confirmed_pit_wumpus(pitprob=pitproba, wumpusprob=wumpusproba)
        if self.verbose:
            display("Pit probabilities:", pitproba)
            display("Agent's current beliefs\n", self.beliefs)
            print("Agent target locations:", targets)

        for i, j in zip(pitproba, wumpusproba):
            try:
                x = np.argmax([i[1], j[1]])
                if x == 1:
                    candidates.append(j)
                elif x == 0:
                    candidates.append(i)
            except:
                candidates.append(pitproba[i], wumpusproba[j])
        if self.verbose:
            print("Candidates: ", candidates)

        [elected.append(candidates[i][1]) for i in range(len(candidates))]
        if self.verbose: print("Probabilities of candiadte pits: ", elected)
        try:
            elected = candidates[np.argmin(elected)]
        except:
            return "no next move"
        if self.verbose: print("Chosen location: ", elected)

        # move to another location if nothing in this location and if it is safe to move
        # else, quit
        if elected[1] < risk:  # move
            path = self.get_shortest_route(source=self.loc_path[-1], target=elected[0], intermediate=True)
            self.make_shortest_turn(pathback=path[0])
            self.score -= (len(path[0]) + path[1] - 1)  # excludes current room
            if self.verbose:
                print("Agent attempts to move to location: ", elected[0], "Agent path: ", self.loc_path)
                

            if elected[0] in self.pits:
                if self.verbose: display(
                    self.printEnv(agentsteps=self.loc_path, spath=path[0], final_state=True, returnpath=True,
                                  target=self.loc_path[-1]))
                self.score -= 1000
                print("-" * 10, "Agent moves to {}. Current environment/state".format(self.loc_path[-1]), "-" * 10)
                print("🕳️ Agent falls into a pit and dies. 🕳️")
                print("Final score: ", self.score)
                return "no next move"

            elif (not self.wumpuscry) and elected[0] == self.wumpusloc:
                if self.verbose: display(
                    self.printEnv(agentsteps=self.loc_path, spath=path[0], final_state=True, returnpath=True,
                                  target=self.wumpusloc))
                self.score -= 1000
                print("-" * 10, "Agent moves to {}. Current environment/state".format(self.loc_path[-1]), "-" * 10)
                print("👾 Agent falls victim to the wumpus. 👾")
                print("Final score: ", self.score)
                return "no next move"
            elif elected[0] == self.goldloc:
                self.loc_path.append(self.goldloc)
                print("-" * 10, "Agent moves to {}. Current environment/state".format(self.goldloc), "-" * 10)
                self.grab(probagent=1)
                path = self.get_shortest_route(source=self.loc_path[-1], target=self.loc_path[0], intermediate=False)
                self.make_shortest_turn(pathback=path[0])
                self.score -= (len(path[0]) + path[1] - 1)  # excludes current room
                self.climb()
                display(self.printEnv(agentsteps=self.loc_path, spath=path[0], final_state=True, returnpath=True,
                                      target=self.loc_path[0]))
                return "no next move"
            else:
                print("-" * 10, "Agent moves to {}. Current environment/state".format(self.loc_path[-1]), "-" * 10)
                display(self.printEnv(agentsteps=self.loc_path, spath=path[0], final_state=True, returnpath=True,
                                      target=elected[0]))
                print("Current score: ", self.score)
                return "move"

        else:  # quit

            path = self.get_shortest_route(source=self.loc_path[-1], target=self.loc_path[0], intermediate=False)
            self.make_shortest_turn(pathback=path[0])
            self.score -= (len(path[0]) + path[1] - 1)  # excludes current room
            display("Too risky! Quitting...")
            print("-" * 10, "Agent returns and exits. Current environment/state.", "-" * 10)
            display(self.printEnv(agentsteps=self.loc_path, spath=path[0], final_state=True, returnpath=True,
                                  target=self.loc_path[0]))
            self.climb()

            return "no next move"


