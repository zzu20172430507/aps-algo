import plotly as py
import copy
import time
import plotly.figure_factory as ff


class Place:
    def __init__(self, holding):
        """
        Place vertex in the petri net
        :holding: Number of token the place is initialized with.
        """
        self.holding = holding
        self.capacity = 999999999999
        self.available = 0

    def set_up(self, holding):
        self.holding = holding


class ArcBase:
    def __init__(self, place, amount=1):
        """
        Arc in the Petri net
        :place : The one place acting as source/target of the arc as arc in the net
        :amount: The amount of token removed /added from/to the place 
        """
        self.place = place
        self.amount = amount


class Out(ArcBase):
    def trigger(self):
        """
        remove token.
        """
        self.place.holding -= self.amount

    def non_blocking(self):
        """
        Validate action of outgoing arc is possible
        """
        return self.place.holding >= self.amount


class In(ArcBase):
    def trigger(self):
        """
        Add tokens
        """
        self.place.holding += self.amount

    def non_blocking(self):
        """
        Validate if enough space to hold 
        """
        return self.place.capacity >= self.place.holding + self.amount


class Transition:
    def __init__(self, out_arcs, in_arcs, pre_machine, next_machine, time, name):
        """
        Transition vertex in the petri net.
        :out_arcs: Collection of ingoing arcs, to the transition vertex.
        :in_arcs: Collection of outgoing arcs, to the transition vertex.
        """
        self.out_arcs = set(out_arcs)
        self.in_arcs = in_arcs
        self.arcs = self.out_arcs.union(in_arcs)
        self.pre_machine = pre_machine
        self.next_machine = next_machine
        self.time = time
        self.name = name

    def fire(self):
        """
        Fire!
        """
        not_blocked = all(arc.non_blocking() for arc in self.arcs)  # 是否之前的库所都有资源
        # Note: This would have to be checked differently for variants of 
        # petri nets that take moure than once from a place ,per transition
        if not_blocked:
            for arc in self.arcs:
                arc.trigger()
        return not_blocked

    def able_fire(self):
        not_blocked = all(arc.non_blocking() for arc in self.arcs)  # 是否之前的库所都有资源
        return not_blocked


class PetriNet:
    def __init__(self, transitions):
        """
        The Petri net runner.
        :transitions: The transition encoding the net
        """
        self.transitions = transitions

    def run(self, firing_sequece, ps, Tm):
        """
        Run the Petri net.
        Details: This is a loop over the transaction firing and then some printing.
        :firing_sequnce: Sequnce of transition names use for run
        :ps : Place holdings to print during the run (debugging).
        """
        print('Using firing sequnce::\n' + "=>".join(firing_sequece))
        print('start {}\n'.format([p.holding for p in ps]))
        gant = [[] for _ in Tm]
        for name in firing_sequece:
            t = self.transitions[name]
            if t.fire():
                print("{} fired!".format(name))
                print("   =>  {}".format([p.holding for p in ps]))
                print("{}".format(t.pre_machine), "{}".format(t.next_machine), t.time)
                begin_time = 0
                for arc in t.out_arcs:  # 所有前驱库所中的可用时间最晚的作为begintime
                    begin_time = max(begin_time, arc.place.available)
                for arc in t.out_arcs:
                    arc.place.available = begin_time + t.time
                gant[t.pre_machine - 1].append([begin_time, begin_time + t.time, t.name])
                for arc in t.in_arcs:
                    arc.place.available = begin_time + t.time
            else:
                # print("{}  ...fizzled".format(name))
                pass
        print("\nfinal {}".format([p.holding for p in ps]))
        return gant


def make_parser():
    """
    : return : A parser reading in some of our simulation parameters
    """
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--firings', type=int)
    parser.add_argument('--marking', type=int, nargs='+')
    parser.add_argument('--capacity', type=int, nargs='+')
    parser.add_argument('--machine_list', type=int, nargs='+')
    parser.add_argument('--aim', type=int, nargs='+')
    return parser


def my_sort(elum):
    return int(elum['Task'][1:])


def create_draw_defination(gant):
    start_time = int(time.time()) * 1000
    millis_seconds_per_minutes = 60 * 1000
    df = []
    for id, machine in enumerate(gant):
        for m in machine:
            operation = {}
            operation['Task'] = 'M' + str(id)
            operation['Start'] = start_time + m[0] * millis_seconds_per_minutes
            operation['Finish'] = start_time + m[1] * millis_seconds_per_minutes
            operation['Resource'] = m[2]
            df.append(copy.deepcopy(operation))
            print(operation)
    # df.sort(key=my_sort, reverse=True)
    print(df)
    return df


def printM(M):
    m = [_.holding for _ in M]
    return m


def htime(M):
    return 0


def gtime(M):
    return 0


if __name__ == "__main__":
    args = make_parser().parse_args()
    ps = [Place(m) for m in args.marking]  # 初始化库所，其中初始状态表示每个库所内的token数量
    M0 = printM(ps)
    Maim = args.aim
    ks = [k for k in args.capacity]
    machine_list = [machine for machine in args.machine_list]
    Tm = [0 for machine in machine_list]
    for i, place in enumerate(ps):
        place.capacity = ks[i]
    print(M0)
    print(Maim)
    if (len(ks) != len(ps)):
        print("input wrong")
        exit()
    ts = dict(  # 初始化变迁
        t1=Transition(
            [Out(ps[0]), Out(ps[1]), Out(ps[17])],
            [In(ps[2])],
            1, 5, 15, "t1"
        ),
        t2=Transition(
            [Out(ps[2]), Out(ps[3]), Out(ps[4])],
            [In(ps[5])],
            2, 5, 3, "t2"
        ),
        t3=Transition(
            [Out(ps[5]), Out(ps[6]), Out(ps[7])],
            [In(ps[8])],
            3, 5, 3, "t3"
        ),
        t4=Transition(
            [Out(ps[8]), Out(ps[9]), Out(ps[10])],
            [In(ps[11])],
            4, 5, 3, "t4"
        ),
        t5=Transition(
            [Out(ps[11])],
            [In(ps[1]), In(ps[12]), In(ps[13])],
            5, 8, 27, "t5"
        ),
        t6=Transition(
            [Out(ps[13])],
            [In(ps[3]), In(ps[14]), In(ps[15])],
            5, 7, 3, "t6"
        ),
        t7=Transition(
            [Out(ps[15])],
            [In(ps[6]), In(ps[14]), In(ps[16])],
            5, 7, 3, "t7"
        ),
        t8=Transition(
            [Out(ps[16])],
            [In(ps[9]), In(ps[14]), In(ps[17])],
            5, 7, 3, "t8"
        ),
        t9=Transition(
            [Out(ps[0]), Out(ps[18]), Out(ps[19])],
            [In(ps[20])],
            1, 6, 15, "t9"
        ),
        t10=Transition(
            [Out(ps[4]), Out(ps[20]), Out(ps[21])],
            [In(ps[22])],
            2, 6, 3, "t10"
        ),
        t11=Transition(
            [Out(ps[7]), Out(ps[22]), Out(ps[23])],
            [In(ps[24])],
            3, 6, 3, "t11"
        ),
        t12=Transition(
            [Out(ps[10]), Out(ps[24]), Out(ps[25])],
            [In(ps[26])],
            4, 6, 3, "t12"
        ),
        t13=Transition(
            [Out(ps[26])],
            [In(ps[12]), In(ps[18]), In(ps[27])],
            6, 8, 27, "t13"
        ),
        t14=Transition(
            [Out(ps[27])],
            [In(ps[14]), In(ps[21]), In(ps[28])],
            6, 7, 3, "t14"
        ),
        t15=Transition(
            [Out(ps[28])],
            [In(ps[14]), In(ps[23]), In(ps[29])],
            6, 7, 3, "t15"
        ),
        t16=Transition(
            [Out(ps[29])],
            [In(ps[14]), In(ps[19]), In(ps[25])],
            6, 7, 3, "t16"
        ),
    )
    # 根据L1算法计算firing sequence
    import heapq as hq

    OPEN_LIST = []
    CLOSED_LIST = []
    hq.heapify(OPEN_LIST)
    hq.heappush(OPEN_LIST, (0, M0))
    flag = 0
    while (len(OPEN_LIST) != 0):
        M_cur = hq.heappop(OPEN_LIST)[1]
        # print(M_cur)
        if (M_cur == Maim):
            flag = 1
            break
        for idx, p in enumerate(ps):
            p.set_up(M_cur[idx])
        # 库所状态为M_cur状态下，看哪些变迁能够发生
        able_fire = []
        for t_name in ts:
            t = ts[t_name]
            if (t.able_fire() == True):  # 如果当前变迁可以发生
                able_fire.append(t_name)
        print(able_fire)
        for t in able_fire:
            ts[t].fire()
            M_changed = [p.holding for p in ps]
            if M_changed in OPEN_LIST:
                print("already in OPEN LIST")
            if M_changed in CLOSED_LIST:
                print("already in CLOSED LIST")
            ftime = htime(M_changed) + gtime(M_changed)
            hq.heappush(OPEN_LIST, (ftime, M_changed))
            for idx, p in enumerate(ps):
                p.set_up(M_cur[idx])

    if flag == 0:
        print("Failed to find optimal solution")
        exit()
    print("find a solution!")
    exit()

    # 有了firing sequence 后进行仿真。
    from random import choice

    firing_sequnce = [choice(list(ts.keys())) for _ in range(args.firings)]  # 随机执行序列
    petri_net = PetriNet(ts)
    gant = petri_net.run(firing_sequnce, ps, Tm)
    print("gant:", gant)
    pyplt = py.offline.plot
    df = create_draw_defination(gant)

    colors = ('rgb(46, 137, 205)',
              'rgb(114, 44, 121)',
              'rgb(198, 47, 105)',
              'rgb(58, 149, 136)',
              'rgb(107, 127, 135)',
              'rgb(46, 180, 50)',
              'rgb(150, 44, 50)',
              'rgb(100, 47, 150)',
              'rgb(58, 100, 180)',
              'rgb(150, 127, 50)',
              'rgb(157, 127, 135)',
              'rgb(146, 180, 50)',
              'rgb(50, 44, 50)',
              'rgb(150, 47, 150)',
              'rgb(158, 100, 180)',
              'rgb(50, 127, 50)')
    fig = ff.create_gantt(df, colors=colors, index_col='Resource', show_colorbar=True,
                          group_tasks=True)
    # pyplt(fig, filename='tmp1.html')
    fig.show()