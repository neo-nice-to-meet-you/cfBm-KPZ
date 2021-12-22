import numpy as np
import random
import fbm
import matplotlib.pyplot as plt


################################################################################
################################### JUST FBM ###################################
################################################################################


class fBM:
    """ A fractional Brownian motion type object
    === Class Attributes ===
    - .hurst : a real number in (0,1), the Hurst index of the fBM
    - .source : the position of the fBM when first initialized
    - .trajectory: a numpy array of the trajectory
    - .num_steps: number of steps """

    def __init__(self, hurst, source):
        self.hurst = hurst
        self.source = source
        self.trajectory = [source]
        self.stepped = False
        self.num_steps = 0

    def steps(self, steps):
        """ Randomly generate steps number of discrete steps of fractional
        Gaussian noise via the Davis-Harte algorithm. Note that the fbm
        implementation of the Davis-Harte method may fail if the self.hurst is
        too close to 1 and if steps is too small. """

        trajectory = self.source + fbm.fbm(steps,
                                           self.hurst,
                                           length=steps,
                                           method='daviesharte')
        self.trajectory = trajectory
        self.stepped = True
        self.num_steps = steps

    def plot(self):
        """ Plot the trajectory of the fBm.
        Preconditions:
        - self.stepped == True """

        H, trajectory, N = self.hurst, self.trajectory, self.num_steps

        # plot results
        plt.title('fBM: H = {H}, N = {N}'.format(H=H, N=N))
        plt.plot([n for n in range(len(trajectory))],
                 trajectory,
                 label='Position')
        plt.xlabel('Time')
        plt.legend()
        plt.show()


################################################################################
########################### USEFUL PRIVATE FUNCTIONS ###########################
################################################################################


def _pairing(lst):
    """ Return a list of all non-ordered pairs of two distinct elements in lst.
    """

    pairs = []
    for i in range(len(lst) - 1):
        pairs += [(lst[i], j) for j in lst[i + 1:]]

    return pairs


def _sign(num):
    """ Return -1 if num is negative, and 1 if num is 0 or positive. """

    return (-1) ** int(num < 0)


def _precedence_score(bef0, bef1, aft0, aft1):
    """ We have two particles labeled 0 and 1. befk and aftk are the positions
    of particle k (k=0,1) before and after, respectively, an incremental motion.
    Since the increment is small, we assume that particles move linearly and
    their trajectories are piecewise line-segments. The particles coalesce if
    the order of the particle's position changes. If the order is unchanged,
    return None. If the order is changed, meaning that the two line segments
    intersect, return the precedence score, which is the time of intersection.
    """

    if _sign(bef1 - bef0) == _sign(aft1 - aft0):
        return None
    else:
        return (bef0 - bef1) / (aft1 - bef1 - (aft0 - bef0))


################################################################################
####################### COIN-FLIP MODEL PRIVATE FUNCTIONS ######################
################################################################################


def _coin(pair):
    """ Choose a leader and a follower out of pair via a fair coin-flip. """
    leader_index = random.choice([0, 1])
    follower_index = 0 ** leader_index

    return pair[leader_index], pair[follower_index]


def _coin_flip_leaderboard(trajectories, steps):
    """ Given a list of independent fractional Gaussian noise trajectories of
    the same length, return a list of dictionaries - a leaderboard.
    For each step, we look at all paths that coalesce in that step, and order
    them based on intersection time, and let them "compete" in that order.
    The competition decides which path to continue, the leader, and the other
    path will follow the leader for all future time.
     For each step in the trajectory, the corresponding dictionary records which
    paths are "defeated" by which paths, if they coalesce in that step. If a
    path is defeated by another path, the former becomes a follower of the
    latter, which comes a leader of the former. When a path's leader is defeated
    by another path, it follows the new leader in lieu of the old leader, which
    is now also a follower or the new leader. The leaderboard records how this
    hierarchy evolves over all steps of the trajectories. Note that the
    trajectories are uniquely identified by their sources, i.e. positions at
    time 0, so the dictionaries maps the source of a trajectory to the source
    of its leader at a given step. The decision of which path wins when they
    coalesce is based on a fair coin-flip. """

    # Follower to leader dictionary at initial time
    leaders = {}
    for source in trajectories:
        leaders[source] = source
    leaderboard = [leaders]

    for i in range(steps):
        old_leaders = leaderboard[-1]
        new_leaders = {}
        competition_results = {}

        # Establish pair precedence, pairs that intersect earlier in time would
        #   compete prior to pairs that intersect later in time.
        pairs = _pairing(list(set(old_leaders.values())))
        pairs_precendence = {}
        matches = []
        for pair in pairs:
            bef0, bef1 = trajectories[pair[0]][i], trajectories[pair[1]][i]
            aft0, aft1 = trajectories[pair[0]][i + 1], trajectories[pair[1]][
                i + 1]
            score = _precedence_score(bef0, bef1, aft0, aft1)
            if score is not None:
                pairs_precendence[score] = pair
                matches.append(score)
        matches.sort()

        # For each pair that competes, in the order given by precedence score,
        #   decides the follower and the leader through a fair coin-flip.
        for match in matches:
            pair = pairs_precendence[match]
            leader, follower = _coin(pair)

            competition_results[follower] = leader
            for another in competition_results:
                if competition_results[another] == follower:
                    competition_results[another] = leader

        # Create a new leader dictionary and append it to the leaderboard.
        for follower in old_leaders:
            old_leader = old_leaders[follower]
            if old_leader in competition_results:
                new_leaders[follower] = competition_results[old_leader]
            else:
                new_leaders[follower] = old_leader
        leaderboard.append(new_leaders)

    return leaderboard


################################################################################
####################### POLYA-URN MODEL PRIVATE FUNCTIONS ######################
################################################################################


def _polya_urn(mass0, mass1, polya):
    """ Randomly choose options 0 or 1 given respective weights mass0 and mass1
    according to the Polya urn scheme with polya index polya. """

    if polya == 'infty':
        if mass0 == mass1:
            return random.choice([0, 1])
        else:
            return [mass0, mass1].index(max(mass0, mass1))
    elif polya == '-infty':
        if mass0 == mass1:
            return random.choice([0, 1])
        else:
            return [mass0, mass1].index(min(mass0, mass1))
    else:
        population = [0, 1]
        denom = mass0 ** polya + mass1 ** polya
        weights = [(mass0 ** polya) / denom, (mass1 ** polya) / denom]
        return random.choices(population, weights=weights, k=1)[0]


def _polya_compete(pair, polya):
    """ Return the selected leader and follower's index in pair based on the
    Polya urn scheme with polya index polya.

    Format: pair = ((mass0, source0), (mass1, source1)) """

    leader_index = _polya_urn(pair[0][0], pair[1][0], polya)
    follower_index = 0 ** leader_index  # Basically sends 0 to 1 and 1 to 0

    return pair[leader_index], pair[follower_index]


def _tuple_counting(lst):
    """ Return a list of counts of all items in lst, organized as 2-tuples,
    of the form (count, item). The purpose of this private functions is to
    calculate the masses of particles, if lst is a list of particles with
    repetition. """

    counted = []
    counting = []
    for item in lst:
        if not (item in counted):
            count = len([i for i in lst if i == item])
            counting.append((count, item))
            counted.append(item)

    return counting


def _polya_urn_leaderboard(trajectories, steps, polya):
    """ Given a list of independent fractional Gaussian noise trajectories of
    the same length, return a list of dictionaries - a leaderboard.
    For each step, we look at all paths that coalesce in that step, and order
    them based on intersection time, and let them "compete" in that order.
    The competition decides which path to continue, the leader, and the other
    path will follow the leader for all future time.
    For each step in the trajectory, the corresponding dictionary records which
    paths are "defeated" by which paths, if they coalesce in that step. If a
    path is defeated by another path, the former becomes a follower of the
    latter, which comes a leader of the former. When a path's leader is defeated
    by another path, it follows the new leader in lieu of the old leader, which
    is now also a follower or the new leader. The leaderboard records how this
    hierarchy evolves over all steps of the trajectories. Note that the
    trajectories are uniquely identified by their sources, i.e. positions at
    time 0, so the dictionaries maps the source of a trajectory to the source
    of its leader at a given step. The decision of which path wins when they
    coalesce is based on the Polya urn scheme with polya index polya, the mass
    of a certain surviving particle is the number of particles following it,
    including itself. All particles have initial mass 1.
    """

    leaders = {}
    for source in trajectories:
        leaders[source] = source
    leaderboard = [leaders]
    for i in range(steps):
        old_leaders = leaderboard[-1]
        # _tuple_counting is meant to calculate masses of surviving particles
        old_leader_mass = _tuple_counting(old_leaders.values())
        new_leaders = {}
        competition_results = {}

        # Establish pair precedence
        # Assuming no repetition of values and mass is already calculated
        pairs = _pairing(old_leader_mass)
        pairs_precendence = {}
        scores = []
        for pair in pairs:
            bef0, bef1 = trajectories[pair[0][1]][i], \
                         trajectories[pair[1][1]][i]

            aft0, aft1 = trajectories[pair[0][1]][i + 1], \
                         trajectories[pair[1][1]][i + 1]

            score = _precedence_score(bef0, bef1, aft0, aft1)
            if score is not None:
                pairs_precendence[score] = pair
                scores.append(score)

        scores.sort()

        # For each pair that competes, in the order given by precedence score,
        #   decides the follower and the leader through the Polya urn.
        for score in scores:
            pair = pairs_precendence[score]
            leader, follower = _polya_compete(pair, polya)

            competition_results[follower[1]] = leader[1]
            for another in competition_results:
                if competition_results[another] == follower[1]:
                    competition_results[another] = leader[1]

        # Create a new leader dictionary and append it to the leaderboard.
        for follower in old_leaders:
            old_leader = old_leaders[follower]
            if old_leader in competition_results:
                new_leaders[follower] = competition_results[old_leader]
            else:
                new_leaders[follower] = old_leader

        leaderboard.append(new_leaders)

    return leaderboard


################################################################################
###################### REGENERATE MODEL PRIVATE FUNCTIONS ######################
################################################################################


def _intersect(bef0, bef1, aft0, aft1):
    """ We have two particles labeled 0 and 1. befk and aftk are the positions
    of particle k (k=0,1) before and after, respectively, an incremental motion.
    Since the increment is small, we assume that particles move linearly and
    their trajectories are piecewise line-segments. The particles intersect if
    the order of the particle's position changes. Return true if the two
    particles intersect. """

    return _sign(bef1 - bef0) != _sign(aft1 - aft0)


def _connected_components(edges):
    """ Return a list of connected components with more than one elements in the
    simple graph given by edge-set edges. The vertices of the graph is
    implicitly given by edges. """

    components = []
    for edge in edges:
        item0, item1 = edge[0], edge[1],
        component0 = _check_membership(components, item0)
        if component0 is not None:
            component0.append(item1)
        else:
            membership1 = _check_membership(components, item1)
            if membership1 is not None:
                membership1.append(item0)
            else:
                components.append(list(edge))

    return [list(set(component)) for component in components]


def _check_membership(components, item):
    """ Return which component in components is such that item is in component.
    Assuming that components is a list of disjoint lists. Return None if item
    is not found in any component in components. """

    for component in components:
        if item in component:
            return component

    return None


################################################################################
############################ LPP PRIVATE FUNCTIONS #############################
################################################################################


def _optimvec(looker, arr, options):
    """ Looker is an index in an array arr of real number. Options are the
    possible movements that the looker can make. Return the move that maximizes
    the number on the entry in arr corresponding to the looker's new position.
    """

    m = -np.inf
    bestlook = None
    for look in options:
        val = arr[looker + look]
        if val > m:
            m = val
            bestlook = look

    return bestlook, m


def _firstgen(arr, options):
    """ Return the optimal directions and the associated weights of going in
    these directions if the possible directions are given by options and arr is
    the generated weights on the last row. Separating firstgen from nextgen is
    to allow us to specific boundary conditions for the LPP geodesics. """

    newdir = []
    newval = []
    newarr = arr

    for looker in range(len(arr) - 1):
        direction, bestvalue = _optimvec(looker+1, newarr, options)
        newdir.append(direction)
        newval.append(bestvalue)
    newvalnp = np.array(newval)

    return newdir, newvalnp


def _nextgen(arr, options):
    """ Return the optimal directions and the associated weights of going in
    these directions if the possible directions are given by options and arr is
    the total future weights of going in the optimal directions. """

    newdir = []
    newval = []
    newarr = arr + np.random.exponential(1, len(arr))
    for looker in range(len(arr) - 1):
        direction, bestvalue = _optimvec(looker+1, newarr, options)
        newdir.append(direction)
        newval.append(bestvalue)
    newvalnp = np.array(newval)

    return newdir, newvalnp


################################################################################
############################ IMPLEMENTATION OF CFBM ############################
################################################################################


class CfBM:
    """ A collection of coalescing fBM based on a given coalescing rule.

    =========================== Class Attributes ===========================

    Warning: All attributes are private, and should only be accessed through
    public methods for plug-in-plug-out consistency.

    - .sources: a list of DISTINCT real numbers that are the starting
        positions of fBMs

    - .members : a dictionary that maps a source to a fBM object that starts
        at the given source.

    - .hurst: all the members share the same Hurst parameter

    - .num_steps : the number of steps of all members in the system

    - .stepped : whether or not the system's steps have already been generated

    - .i_trajectories : trajectories of independent fBM. A dictionary that maps
        sources to the trajectories of the corresponding particle.

    - .c_trajectories : trajectories of coalesced fBM. A dictionary that maps
        sources to the trajectories of the corresponding particle.

    - .rule: any real number corresponding to the Polya index, or the strings
        'infty', '-infty', 'regenerate'. Specifies the coalescing rule.
        Specifically, when self.rule == 0, we are generating the coin-flip
        model. When self.rule == 'regenerate' we are in the regenerate model.
        Default is the coin-flip model.
    """

    def __init__(self, hurst, sources, steps, rule):
        """ Initialize an CfBM type object. hurst must be a number 0< and <1.
        sources must be a list of DISTINCT real numbers that are the starting
        positions of fBMs. rule must be any real number corresponding to the
        Polya index, or the strings 'infty', '-infty', 'regenerate'. Specifies
        the coalescing rule. Specifically, when self.rule == 0, we are
        generating the coin-flip model. When self.rule == 'regenerate' we are
        in the regenerate model. Default is the coin-flip model. steps is the
        number of steps required for the model, notice that scaling is left to
        the user's discretion.

        Warning: the davisharte implementation of fbm might fail when
        hurst is close to 1 and when steps is small. """

        members = {}
        for source in sources:
            member = fBM(hurst, source)
            members[source] = member

        self.sources = sources
        self.hurst, self.members = hurst, members
        self.binpow = None, None
        self.stepped = False
        self.i_trajectories = None
        self.c_trajectories = None
        self.rule = rule
        self.num_steps = steps

    def step_number(self):
        """ Return the number of steps specified in initialization. """

        return self.num_steps

    def hurst_index(self):
        """ Return the hurst index specified in initialization. """

        return self.hurst

    def get_sources(self):
        """ Return the sources, or initial starting points specified in
        initialization. """

        return list(self.sources)

    def coalescing_rule(self):
        """ Return the coalescing rule specified in initialization. """

        if self.rule == 0:
            return "coin-flip"
        elif self.rule in ['infty', '-infty']:
            return "Polya urn with Polya index " + str(self.rule)
        elif self.rule == 'regenerate':
            return 'regenerate'
        else:
            return "Polya urn with Polya index " + str(self.rule)

    def post_coal_trajectory(self, source):
        """ Return the post-coalescence trajectory of the particle originating
        from source, as a numpy array. """

        if source not in self.members:
            print('Invalid source. ')
        else:
            trajs = self.coalesce()
            return trajs[source]

    def pre_coal_trajectory(self, source):
        """ Return the pre-coalescence trajectory of the particle originating
        from source, as a numpy array.
        Warning: This method only applies to non-regenerate models.
        """
        if self.rule == 'regenerate':
            print('Method invalid for non-regenerate models. ')
        elif source not in self.members:
            print('Invalid source. ')
        else:
            trajs = self._trajectories()
            return trajs[source]

    def _steps(self):
        """ Generate independent fBM for each source. """

        if not self.stepped:
            steps = self.step_number()
            for source in self.members:
                member = self.members[source]
                member.steps(steps)

            self.stepped = True

    def _trajectories(self):
        """ Save and/or return the i_trajectories. This is only a necessary step
        for the purpose of coin-flip and Polya coalescing rules, and not needed
        for the regenerate model. """

        if self.i_trajectories is not None:
            return self.i_trajectories
        else:
            self._steps()
            trajectories = {}
            for source in self.members:
                member = self.members[source]
                trajectories[source] = member.trajectory

            self.i_trajectories = trajectories
            return trajectories

    def _coin_flip_coalesce(self):
        """ Coalesce the i_trajectories using the coin-flip coalescing rule and
        generate the c_trajectories. Store the c_trajectories and return it.
        If c_trajectories have already been generated, return the stored
        c_trajectories. """

        if self.c_trajectories is not None:
            return self.c_trajectories
        else:
            trajectories = self._trajectories()
            leaderboard = _coin_flip_leaderboard(trajectories, self.num_steps)
            c_trajectories = {}
            for source in self.members:
                c_trajectories[source] = []
            for i in range(self.num_steps + 1):
                leaders = leaderboard[i]
                for follower in leaders:
                    item = trajectories[leaders[follower]][i]
                    c_trajectories[follower].append(item)

            self.c_trajectories = c_trajectories
            return c_trajectories

    def _polya_urn_coalesce(self):
        """ Coalesce the i_trajectories using the Polya urn coalescing rule and
        generate the c_trajectories. Store the c_trajectories and return it.
        If c_trajectories have already been generated, return the stored
        c_trajectories. """

        polya = self.rule
        if self.c_trajectories is not None:
            return self.c_trajectories
        else:
            trajectories = self._trajectories()
            leaderboard = _polya_urn_leaderboard(trajectories,
                                                 self.num_steps,
                                                 polya)
            c_trajectories = {}
            for source in self.members:
                c_trajectories[source] = []
            for i in range(self.num_steps + 1):
                leaders = leaderboard[i]
                for follower in leaders:
                    item = trajectories[leaders[follower]][i]
                    c_trajectories[follower].append(item)

            self.c_trajectories = c_trajectories
            return c_trajectories

    def _regenerate_coalesce(self):
        """ Coalesce the i_trajectories using the regenerate coalescing rule and
        generate the c_trajectories. Store the c_trajectories and return it.
        If c_trajectories have already been generated, return the stored
        c_trajectories.

        Note: The regenerate coalescence algorithm works very differently from
        the other ones, since new fBM must be generated after every intersection
        of particles. Hence a dynamical approach is more suited.
        """

        steps = self.num_steps
        SBS = []
        current_trajectories = []
        for source in self.sources:
            item = tuple(source + fbm.fbm(steps,
                                          self.hurst,
                                          length=steps,
                                          method='daviesharte'))
            current_trajectories.append(item)

        # The length of an fbm trajectory of n steps is of length n+1
        #   so the iteration is over range(steps) rather than range(steps-1)
        for step in range(steps):
            links = {}
            all_pairs = _pairing(current_trajectories)
            competitions = []

            for pair in all_pairs:
                bef0, bef1, aft0, aft1 = pair[0][step], \
                                         pair[1][step], \
                                         pair[0][step + 1], \
                                         pair[1][step + 1]
                if _intersect(bef0, bef1, aft0, aft1):
                    competitions.append(pair)

            pools = _connected_components(competitions)
            new_trajectories = []

            for traj in current_trajectories:
                if _check_membership(pools, traj) is None:
                    new_trajectories.append(traj)
                    links[traj] = traj

            for pool in pools:
                positions = [traj[step + 1] for traj in pool]
                # The starting point of the regenerated fBM is a uniform random
                #   choice out of the position of all the particles that
                #   coalesced in that step. One can alternatively take the
                #   average of their positions.
                placement = random.choice(positions)
                new_born_fbm = placement + fbm.fbm(steps,
                                                   self.hurst,
                                                   length=steps,
                                                   method='daviesharte')
                # The +1 here is because step is iterated 0-based and we are
                #   talking about length up to that index hence +1.
                new_traj = tuple(np.concatenate((np.full(step + 1,
                                                         placement),
                                                 new_born_fbm)))
                new_trajectories.append(new_traj)
                for traj in pool:
                    links[traj] = new_traj

            current_trajectories = new_trajectories
            SBS.append(links)

        histories = {}

        for traj in SBS[0]:
            chain = [traj]
            for links in SBS:
                chain.append(links[chain[-1]])
            # Notice that things aren't ordered so we need to call traj[0]
            #   to retrieve the source
            histories[traj[0]] = chain

        c_trajectories = {}
        # The 0th trajectory is always there
        length = len(histories[0])
        for source in self.sources:
            c_trajectories[source] = [histories[source][step][step] for step in range(length)]

        self.c_trajectories = c_trajectories
        return c_trajectories

    def coalesce(self):
        """ Coalesce the paths and create c_trajectories according the
        coalescing rule given by self.rule. """

        if self.c_trajectories is not None:
            return self.c_trajectories
        elif self.rule == 0:
            return self._coin_flip_coalesce()
        elif self.rule == 'regenerate':
            return self._regenerate_coalesce()
        else:
            return self._polya_urn_coalesce()

    def inde_plot(self):
        """ Plot the i_trajectories.
        Warning: Only non-regenerate models have plots of independent fBM. """

        if not self.stepped:
            print("This method only applies to non-regenerate models.")
        else:
            N, H = self.num_steps, self.hurst
            trajectories = self._trajectories()

            plt.title('fBMs: H = {H}, steps = {N}'.format(H=H, N=N))
            plt.xlabel('Time')

            for source in trajectories:
                trajectory = trajectories[source]
                plt.plot([n for n in range(N + 1)], trajectory)

            plt.show()

    def coal_plot(self):
        """ Plot the c_trajectories. """

        N, H = self.num_steps, round(self.hurst, 3)
        if self.rule == 'regenerate':
            desc = 'regenerate'
        elif self.rule == 0:
            desc = "coin-flip"
        else:
            desc = "Polya urn with index " + str(self.rule)
        trajectories = self.coalesce()

        plt.title('fBMs: H = {H}, steps = {N}, {D}'.format(H=H,
                                                           N=N,
                                                           D=desc))
        plt.xlabel('Time')

        for source in trajectories:
            trajectory = trajectories[source]
            plt.plot([n for n in range(len(trajectory))], trajectory)

        plt.show()

    def point_fields(self):
        """ Return the upper and lower point_fields of the c_trajectories. """

        trajectories = self.coalesce()
        updown = {}
        for source in trajectories:
            traj = trajectories[source]
            updown[source] = traj[-1]
        upendpts = list(set(updown.values()))
        downendpts = list(set([min([source for source in updown
                                    if updown[source] == val])
                               for val in upendpts]))
        upendpts.sort()
        downendpts.sort()

        return upendpts, downendpts


################################################################################
############################ IMPLEMENTATION OF LPP #############################
################################################################################


class LPP:
    """ An LPP (last passage percolation) type object.
    =========================== Class Attributes ===========================

    Warning: All attributes are private, and should only be accessed through
    public methods for plug-in-plug-out consistency.

    The LPP trajectories are generated dynamically. We dynamically produce a
    discrete percolation "vector field" that tells a particle at a given
    space-time point its next optimal position. We start backward, from the last
    row of weights, which is generated with a specific boundary condition, and
    decide that is the optional direction for all positions in the previous row,
    and record the total weights of going along the optional path. Then we
    generate the second last row of weights and add it to the total weights, and
    look for the optional directions at each points in the third last row, as
    well as total future weights of going along the optimal directions. We keep
    going until the entire vector field is generated.

    - .fills : the initial positions/sources are all integers in [-fills, fills]
    - .VF : the discrete vector field
    - .c_trajectories : the trajectories of particles in the LPP process. It is
        a dictionary that maps the source (position at time 0) of a particle to
        its trajectory
    - .num_steps : the number of steps being generated
    """

    def __init__(self, fills, steps):
        self.fills = fills
        self.VF = None
        self.c_trajectories = None
        self.num_steps = steps

    def _steps(self):
        """ Generate, store and return the vector field. """

        if self.VF is not None:
            return self.VF
        else:
            steps = self.num_steps
            initial = 2*self.fills + steps + 1
            options = [-1, 0]

            arr = []
            phi = 0
            i = 0
            # This is the specification of our boundary condition which is
            #   distributed as the different of two Exp(2) random variables.
            while i < initial:
                arr.append(phi)
                phi += (np.random.exponential(2) - np.random.exponential(2))
                i += 1

            VF = []

            arr = np.array(arr)

            dire, arr = _firstgen(arr, options)
            VF.append((dire, arr))

            for step in range(steps-1):
                dire, arr = _nextgen(arr, options)
                VF.append((dire, arr))

            self.VF = VF
            return VF

    def get_sources(self):
        """ Return the sources, or initial starting points specified in
        initialization. """

        return [i for i in range(-self.fills, self.fills+1)]

    def step_number(self):
        """ Return the number of steps specified in initialization. """

        return self.num_steps

    def coalesce(self):
        """ Coalesce the paths and create c_trajectories. """

        if self.c_trajectories is not None:
            return self.c_trajectories
        else:
            VF = self._steps()
            sources = [i for i in range(-self.fills, self.fills+1)]
            act_sources = [2*i for i in range(-self.fills, self.fills+1)]
            trajectories = {}
            for i in range(2*self.fills+1):
                pos = sources[i]
                act_pos = act_sources[i]
                traj = [act_pos]
                step = 0
                for field in reversed(VF):
                    dirs = field[0]
                    v = dirs[pos+self.fills+step]
                    pos += v
                    if v == 0:
                        act_pos += 1
                    else:
                        act_pos += v
                    traj.append(act_pos)
                    step += 1
                trajectories[sources[i]] = traj

            self.c_trajectories = trajectories
            return trajectories

    def coal_plot(self):
        """ Plot the c_trajectories. """
        trajectories = self.coalesce()
        steps = len(trajectories[0])-1

        # plot results
        plt.title('LPP: fills={fills}, steps={steps}'.format(fills=self.fills,
                                                             steps=steps))
        plt.xlabel('Time')

        for source in trajectories:
            trajectory = trajectories[source]
            plt.plot([n for n in range(steps+1)], trajectory)

        plt.show()

    def point_fields(self):
        """ Return the upper and lower point_fields of the c_trajectories. """

        trajectories = self.coalesce()
        updown = {}
        for source in trajectories:
            traj = trajectories[source]
            updown[source] = traj[-1]
        upendpts = list(set(updown.values()))
        downendpts = list(set([min([source for source in updown
                                    if updown[source] == val])
                               for val in upendpts]))
        upendpts.sort()
        downendpts.sort()

        return upendpts, downendpts
