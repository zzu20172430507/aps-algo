import plotly as py
import copy
import time
import json
import plotly.figure_factory as ff
# from pprint import pprint


class Task:
    """
    Base Task job.
    """
    def __init__(self, id, occupy) -> None:
        self.id = id
        # self.occupy表示占用的设备资源，例如 [1,2,3] 表示从1，2，3中选择一个
        self.occupy = occupy
        self.release = []
        self.occupy_dependency = []
        self.release_dependency = []
        self.dependency2 = -1
        # self.status表示该任务当前状态，0表示未执行，1表示已执行
        self.status = 0
        self.priority = 0
        self.time = 0
        self.pre = None
        self.next = None
        self.machine = -1  # machine 表示当前task在哪个machine上工作
        self.position = -1  # position 表示当前task在哪个位置资源上工作
        self.release_machine = []  # 表示当前task释放的machine
        self.release_position = []  # 表示当前task释放的position
        self.dependency = -1  # dependency 表示当前task和哪个task需要有相同的设备
        self.available = 0
        self.start_time = 0
        self.heuristic = 0
        self.task_name = ""
        self.PlateProcess = ""
      
    def __lt__(self, other):
        """定义<比较操作符"""
        if self.priority == other.priority:
            return self.heuristic > other.heuristic
        return self.priority > other.priority


class MachineBase:
    def __init__(self, id) -> None:
        self.id = id


class Position:
    def __init__(self, id) -> None:
        self.id = id
        self.status = 0  # status为0表示空闲、为1表示忙碌。
        self.machine = None
        self.machineId = None


class TaskGragh:
    """
    A DAG. Representing the relationship between Tasks.
    """
    def __init__(self, tasks) -> None:
        self.tasks = tasks


def make_parser():
    """
    : return : A parser reading in some of our simulation parameters
    """
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--task', type=int)
    parser.add_argument('--machine', type=int)
    parser.add_argument('--position', type=int)
    parser.add_argument('--heuristics', type=int)
    parser.add_argument('--base_num', type=int)
    return parser


def dfs(tasks, task_id, t_idx_dic):
    max_time = 0
    for next in tasks[t_idx_dic[task_id]].next:
        max_time = max(max_time, tasks[t_idx_dic[next]].priority)
    tasks[t_idx_dic[task_id]].priority = tasks[t_idx_dic[task_id]].time + max_time
    for pre in tasks[t_idx_dic[task_id]].pre:
        tasks = dfs(tasks, pre, t_idx_dic)
    return tasks


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


def generate_colors(x):
    import random
    colors = []
    for i in range(x):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        color = f'rgb({r}, {g}, {b})'
        colors.append(color)
    return colors


if __name__ == "__main__":
    args = make_parser().parse_args()
    # 执行命令：python test_case2.py --task 10 --machine 5 --position 7 --heuristic 0 --base_num 14
    with open("./data_low.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    tasks_json = data['assaysmodel'][0]['tasks']
    positions_json = data['assaysmodel'][0]['positions']
    machines_json = data['assaysmodel'][0]['machines']
    base_num = len(tasks_json)
    board_num = 1
    t_idx_dic = {}  # 表示Id在数组中的下标
    p_idx_dic = {}
    m_idx_dic = {}
    tasks = [Task(t, []) for t in range(board_num * base_num)]
    positions = [Position(p) for p in range(len(positions_json))]
    heuristics = 0
    machines = [MachineBase(m+1) for m in range(len(machines_json))]
    for tidx, task in enumerate(tasks):
        # if tidx >= base_num:
        #     break
        task.id = tasks_json[tidx]['id']
        task.idx = tidx
        t_idx_dic[task.id] = tidx
        task.occupy = tasks_json[tidx]['occupy']
        task.release = tasks_json[tidx]['release']
        task.time = tasks_json[tidx]['time']
        task.task_name = tasks_json[tidx]['taskname']
        task.PlateProcess = tasks_json[tidx]['plateprocess']
        task.pre = tasks_json[tidx]['pre']
        task.next = tasks_json[tidx]['next']
    for pidx, pos in enumerate(positions):
        pos.id = positions_json[pidx]['id']
        pos.name = positions_json[pidx]['positionname']
        pos.machine = positions_json[pidx]['machine']
        p_idx_dic[positions_json[pidx]['id']] = pidx
    for midx, mac in enumerate(machines):
        mac.id = machines_json[midx]['id']
        m_idx_dic[machines_json[midx]['id']] = midx

    for task in tasks:  # 通过occupy_dependency和release_dependency标识是否存在占用释放依赖关系
        for occ in task.occupy:
            task.occupy_dependency.append(-1)
        for rel in task.release:
            task.release_dependency.append(-1)
    for t in range(board_num * base_num):  # 复制的任务占用设备和释放设备应该是不变的
        if t >= base_num:
            tasks[t].occupy = tasks[t % base_num].occupy
            tasks[t].release = tasks[t % base_num].release
    for t in range(board_num * base_num):  # 复制的任务的前驱和后继的id对应加上
        if t >= base_num:
            task_base = tasks[t % base_num]
            tasks[t].id = task_base.id + str(t)
            t_idx_dic[tasks[t].id] = t
            tasks[t].pre = [t_idx_dic[_]+int(t//base_num*base_num) for _ in task_base.pre]
            tasks[t].next = [t_idx_dic[_]+int(t//base_num*base_num) for _ in task_base.next]
            tasks[t].time = tasks[t % base_num].time
    # task.dependency 的设置  dependency约束表示占用时要求在同一个设备，dependency2要求同一位置
    for t in range(board_num * base_num):  # 复制的任务的前驱和后继的id对应加上
        if t >= base_num:
            task_base = tasks[t % base_num]
            tasks[t].occupy_dependency = copy.deepcopy(task_base.occupy_dependency)
            tasks[t].release_dependency = copy.deepcopy(task_base.release_dependency)
            for occ_id, occ in enumerate(task_base.occupy_dependency):
                if occ != -1:
                    tasks[t].occupy_dependency[occ_id] = task_base.occupy_dependency[occ_id] + int(t//base_num*base_num)
            for rel_id, rel in enumerate(task_base.release_dependency):
                if rel != -1:
                    tasks[t].release_dependency[rel_id] = task_base.release_dependency[rel_id] + int(t//base_num*base_num)
    # 根据L1列表调度算法计算优先级
    # L1列表调度算法通过计算到终点节点的时间来作为优先级。
    pre_END = []  # pre_END 表示最终节点的前驱节点。
    for t in tasks:
        if not t.next:
            pre_END.append(t.id)
    for node in pre_END:
        tasks = dfs(tasks, node, t_idx_dic)
    print("tasks' priority are shown as follows:")
    for _id, task in enumerate(tasks):
        print(f"task {_id} priority:", task.priority)
    # L1 scheduling to compute priority 优先级计算时考虑启发式规则，优先选择执行后prioqueue长度更长的
    if heuristics == 1:
        print("--------------------------启发式模块开启--------------------------")
    else:
        print("--------------------------未启用启发式模块--------------------------")
    print("\n")
    # 对资源的建模
    start_time = time.time()
    for idx, task in enumerate(tasks):
        print("task Idx:", idx, " ID: ", task.id, " Name: ", task.task_name)
    for idx, pos in enumerate(positions):
        print("pos Idx:", idx, " ID: ", pos.id, " Name: ", pos.name)
    import heapq
    q = []
    pq_tmp = []
    # ---------初始化任务：判断当前哪些任务可以执行
    for task in tasks:
        flag = 1
        if task.pre is not None:
            for pre in task.pre:  # 判断是否所有前驱任务都完成
                if tasks[t_idx_dic[pre]].status == 0:
                    flag = 0
        # 判断所需要的资源是否都满足
        flag2 = 1
        for occ_id, occ in enumerate(task.occupy):
            tmp_f = 0
            for pos in occ:
                if positions[p_idx_dic[pos]].status == 0:
                    tmp_f = 1
            if tmp_f == 0:  # 如果资源里有一项
                flag2 = 0
        if len(task.occupy) == 0:
            flag2 = 1
        # 如果前驱任务完成了并且资源可满足，那么当前任务可执行 PUT INTO PRIORITY QUEUE
        if flag == 1 and flag2 == 1:
            heapq.heappush(q, task)
            pq_tmp.append(t_idx_dic[task.id])

    # 进程回溯，保存当前状态
    i = 0
    Finished = []
    SAVED_INFORMATION = []  # 用于存储每步选择过的task，当前这步选择的task，选择前的状态
    SAVED_CUR_TASK = []  # 存储的是当前步选择的设备
    SAVED_CUR_TASK_STATUS = [[] for _ in range(len(tasks)+1)]  # 存储 step 时的任务的状态
    SAVED_CUR_RESOURCE_POSITIONS = [[] for _ in range(len(tasks)+1)]  # 存储position状态
    SAVED_CUR_RESOURCE_MACHINES = [[] for _ in range(len(tasks)+1)]    # 存储machine状态
    SAVED_PRE_DECISIONS = [[] for _ in range(len(tasks)+1)]  # 记录每一个step曾经做过的决策
    SAVED_PRIOR_QUEUE = [[] for _ in range(len(tasks)+1)]  # 记录在做决策前的优先队列的内容
    SAVED_PRIOR_QUEUE[0] = [t.id for t in q]

    TASK_SELECT = [[] for _ in range(len(tasks)+1)]
    DEAD = 0  # 用于调试
    step = 0  # step 用于表示当前的步骤
    print("step:", step, "Priority_queue:", pq_tmp)
    """
    · 首先从当前优先队列中取出一个优先级最高的任务， 判断当前的任务是否之前选择过，如果选择过就再向前回溯一步。
    · 如果回溯至第0步仍没有可用的任务，那么则输出报错信息。
    
    """
    while len(q) > 0:
        task = heapq.heappop(q)  # EACH TIME GET THE PRIOR TASK
        SIG_CALL_BACK = 0  # SIG_CALL_BACK用于记录是否在当前队列中找到了可执行的任务。
        while task.id in SAVED_PRE_DECISIONS[step]:  # 所有优先队列中的任务如果出现在了之前这一步的记录中，那么则不会选择，如果都没有，那么sigcallback记为1
            if len(q) == 0:
                SIG_CALL_BACK = 1
                break
            task = heapq.heappop(q)

        while SIG_CALL_BACK == 1:  # 如果当前所有的队列中的都尝试过了，就再往前回溯一步
            SAVED_PRE_DECISIONS[step] = []  # 再往前一步要清空现在这一步的内容，否则可能对后续会产生影响
            if step == 0:  # 如果当前没有上一步了，那么需要报错处理
                print("PLEASE CHECK INPUT NO RESULT AVAILABLE")
                exit()
            step -= 1
            # print("remove:", SAVED_CUR_TASK.pop(step))
            tasks = copy.deepcopy(SAVED_CUR_TASK_STATUS[step])  # 恢复状态
            positions = copy.deepcopy(SAVED_CUR_RESOURCE_POSITIONS[step])
            machines = copy.deepcopy(SAVED_CUR_RESOURCE_MACHINES[step])
            q_id = copy.deepcopy(SAVED_PRIOR_QUEUE[step])
            q = [tasks[t_idx_dic[t_id]] for t_id in q_id]
            task = heapq.heappop(q)

            SIG_tmp = 0
            while task.id in SAVED_PRE_DECISIONS[step]:
                if len(q) == 0 and (task.id in SAVED_PRE_DECISIONS[step]):
                    SIG_tmp = 1
                    break
                task = heapq.heappop(q)
            if SIG_tmp == 0:
                break

        SAVED_CUR_TASK_STATUS[step] = copy.deepcopy(tasks)  # 保存当前step的task-status
        SAVED_CUR_RESOURCE_POSITIONS[step] = copy.deepcopy(positions)
        SAVED_CUR_RESOURCE_MACHINES[step] = copy.deepcopy(machines)
        SAVED_PRE_DECISIONS[step].append(copy.deepcopy(task.id))

        step += 1

        # 如果当前task没有设置依赖约束，则从所有可用设备中选择第一个可用设备占用
        tmp_machine = []
        tmp_position = []
        for occ_id, occ in enumerate(task.occupy):
            if task.occupy_dependency[occ_id] == -1:
                for pos in occ:  # 从里面选择第一个可用的
                    if positions[p_idx_dic[pos]].status == 0:
                        positions[p_idx_dic[pos]].status = 1
                        tmp_position.append(positions[p_idx_dic[pos]].id)
                        tmp_machine.append(positions[p_idx_dic[pos]].machine)
                        break
        # 若当前task 选择了依赖约束：和其他task在相同的machine上执行   处理某些任务必须在相同设备上执行的约束。
            else:
                for pos in occ:
                    if positions[p_idx_dic[pos]].status == 0 and positions[p_idx_dic[pos]].machine in tasks[task.occupy_dependency[occ_id]].machine:
                        positions[p_idx_dic[pos]].status = 1
                        tmp_position.append(positions[p_idx_dic[pos]].id)
                        tmp_machine.append(positions[p_idx_dic[pos]].machine)
                        break
                if len(task.occupy) == 0:
                    tmp_machine.append(tasks[task.occupy_dependency[occ_id]].machine)
        task.machine = copy.deepcopy(tmp_machine)
        task.position = copy.deepcopy(tmp_position)

        # 当前任务释放设备
        for rel_id, rel in enumerate(task.release):
            if len(rel) > 1 and task.release_dependency[rel_id] != -1:
                pre_pos = tasks[task.release_dependency[t_idx_dic[rel_id]]].position
                for cur_pos in rel:
                    if cur_pos in pre_pos:
                        positions[cur_pos].status = 0
                        task.release_position.append(cur_pos)
                        task.release_machine.append(positions[cur_pos].machine)
            else:
                for pos in rel:
                    positions[p_idx_dic[pos]].status = 0
                    task.release_position.append(pos)
                    task.release_machine.append(positions[p_idx_dic[pos]].machine)
        task.status = 1
    
        tmpPos = [0 for _ in range(len(positions))]
        for pos_id, pos in enumerate(positions):
            tmpPos[pos_id] = pos.status
        print("pos状态为:", tmpPos)
        # 更新队列
        i = 0
        other_q = []
        while i < len(q):  # 输出一下当前优先队列里的内容，用于调试   先将优先队列内容清空，看所有任务当前是否能执行，后续可以对这里进行一定的优化工作。
            other_q.append(heapq.heappop(q).id)
        task_state = []
        for t in tasks:
            task_state.append(t.status)
        if DEAD == 1:
            print(task.id, "其他优先队列中的task:", other_q)
            print("task_state:", task_state)
        pq_list = []
        for t in tasks:
            flag = 1
            if t.pre is not None:
                for pre in t.pre:  # 判断是否所有前驱任务都完成
                    if tasks[t_idx_dic[pre]].status == 0:
                        flag = 0
                    # 判断所需要的资源是否都满足
            flag2 = 1
            for occ_id, occ in enumerate(t.occupy):
                tmp_f = 0
                for pos in occ:
                    if positions[p_idx_dic[pos]].status == 0:
                        tmp_f = 1
                if tmp_f == 0:  # 如果资源里有一项
                    flag2 = 0
            if len(t.occupy) == 0:
                flag2 = 1
            if flag == 1 and flag2 == 1 and t.status == 0:
                heapq.heappush(q, t)
                pq_list.append(t.id)
                if heuristics == 1:   # 启发式规则计算第二优先级
                    tmp_Task = tasks   # 暂存状态，假设执行一步，计算下一步可执行的任务数量作为优先级
                    tmp_position = positions
                    tmp_machine = machines
                    tmp_cnt = 0
                    # -----------------simulate exe task-------------------------------------------------------------------------------------------------------------
                    # 如果当前task没有设置依赖约束，则从所有可用设备中选择第一个可用设备占用
                    if task.dependency == -1:
                        for pos in task.occupy:
                            if positions[pos].status == 0:
                                positions[pos].status = 1
                                task.position = positions[pos].id
                                task.machine = positions[pos].machine
                                break
                    # 若当前task 选择了依赖约束：和其他task在相同的machine上执行   处理某些任务必须在相同设备上执行的约束。
                    else:
                        for pos in task.occupy:
                            if positions[pos-1].status == 0 and positions[pos].machine == tasks[task.dependency].machine:
                                positions[pos-1].status = 1
                                task.position = positions[pos].id
                                task.machine = positions[pos].machine
                                break
                        if len(task.occupy) == 0:
                            task.machine = tasks[task.dependency].machine
                    # 当前任务释放设备
                    if len(task.release) > 1 and task.dependency2 != -1:
                        pos = tasks[task.dependency2].position
                        positions[pos].status = 0
                        task.release_position.append(pos)
                        task.release_machine.append(positions[pos].machine)
                    else:
                        for pos in task.release:
                            positions[pos].status = 0
                            task.release_position.append(pos)
                            task.release_machine.append(positions[pos].machine)
                    task.status = 1
                    while i < len(q):  # 输出一下当前优先队列里的内容，用于调试   先将优先队列内容清空，看所有任务当前是否能执行，后续可以对这里进行一定的优化工作。
                        print("task:", task.id, heapq.heappop(q).id)
                    for task in tasks:
                        flag = 1
                        if task.pre is not None:
                            for pre in task.pre:  # 判断是否所有前驱任务都完成
                                if tasks[pre].status == 0:
                                    flag = 0
                        # 判断所需要的资源是否都满足
                        flag2 = 0
                        if task.dependency == -1:
                            for pos in task.occupy:
                                if positions[pos-1].status == 0:
                                    flag2 = 1
                        else:
                            for pos in task.occupy:
                                if positions[pos-1].status == 0 and positions[pos-1].machine == tasks[task.dependency].machine:
                                    flag2 = 1
                        if len(task.occupy) == 0:
                            flag2 = 1
                        if flag == 1 and flag2 == 1 and task.status == 0:
                            tmp_cnt += 1
                    tasks = copy.deepcopy(tmp_Task)
                    positions = copy.deepcopy(tmp_position)
                    machines = copy.deepcopy(tmp_machine)
                    task.heuristic = copy.deepcopy(tmp_cnt)

        SAVED_PRIOR_QUEUE[step] = [t.id for t in q]
        Finished_task = []
        for t in tasks:
            if t.status == 1:
                Finished_task.append(t.id)
        # print("step:", step, "Selected_Task:", task.task_name, "Priority_queue:", pq_list, "Finished:", Finished_task, "occ:", task.occupy, "rel:", task.release)
        print("step:", step, "Selected_Task:", task.task_name, " id: ", t_idx_dic[task.id])
        
        posS = []
        for p in positions:
            posS.append(p.status)
        # print("pos_status:", posS)
        taskS = []
        for t in tasks:
            taskS.append(t.status)
        # print("task_status:", taskS)
        taskS = []
        for t in SAVED_CUR_TASK_STATUS[step-1]:
            taskS.append(t.status)
        # print("step:",step-1,"SAVED_TASK:", taskS)
        # print("len:",len(SAVED_CUR_TASK_STATUS))
        # print("\n")
        TASK_SELECT[step] = Finished_task
        # print("SAVED_TASK_ORDER:",SAVED_CUR_TASK)
        if len(q) == 0 and step != len(tasks):
            # CALL BACK
            if step == 0:
                print('请检查输入是否符合逻辑！')
                exit()
            step -= 1
            # 首先恢复到上一个step的状态。
            tasks = copy.deepcopy(SAVED_CUR_TASK_STATUS[step])  # task状态恢复
            positions = copy.deepcopy(SAVED_CUR_RESOURCE_POSITIONS[step])  # position状态恢复
            machines = copy.deepcopy(SAVED_CUR_RESOURCE_MACHINES[step])  # machine状态恢复
            # print("remove:", SAVED_CUR_TASK.pop())
            # Finished.pop()
            q_id = copy.deepcopy(SAVED_PRIOR_QUEUE[step])
            q = [tasks[t_idx_dic[t_id]] for t_id in q_id]
            print("RETURN　LAST　STEP !!", "任务ID:", t_idx_dic[task.id], "当前step:", step)
            DEAD = 2
            decision = []
            for dec in SAVED_PRE_DECISIONS[step]:
                decision.append(dec)
            # print("STEP:", step, "SAVED_PRE_DECISIONS:", decision)

#  --------------------------------------------- 启发式搜索完成 -------------------------------------------------------
    if step != len(tasks):
        print("exsiting deadlock !")
    else:
        print("successfully compute solution !")
    # print(len(Finished))

    print("step=", step)
    tasks_status = SAVED_CUR_TASK_STATUS[step-1]
    task_id = []
    # for task in tasks:
    #     print(task.id, ":", task.machine, task.position)
    # print("\n")
    # for pos in positions:
    #     print(pos.machine)
    # 当前知道了task分配资源的顺序，知道每个资源分配的设备，知道task时间，下面生成一个甘特图。
    print("Task SELECT:", TASK_SELECT)
    task_id = []
    for step in range(0, len(tasks)):
        l1 = TASK_SELECT[step]
        l2 = TASK_SELECT[step+1]
        for i in l2:
            if i not in l1:
                task_id.append(i)

    print("task分配资源的顺序为:")
    print(len(task_id))
    print(task_id)
    # colors = ['rgb(46, 137, 205)',
    #           'rgb(114, 44, 121)',
    #           'rgb(198, 47, 105)',
    #           'rgb(58, 149, 136)',
    #           'rgb(107, 127, 135)',
    #           'rgb(46, 180, 50)',
    #           'rgb(150, 44, 50)',
    #           'rgb(100, 47, 150)',
    #           'rgb(58, 100, 180)',
    #           'rgb(150, 127, 50)',
    #           'rgb(157, 127, 135)',
    #           'rgb(46, 180, 50)',
    #           'rgb(150, 44, 50)',
    #           'rgb(100, 47, 150)',
    #           'rgb(58, 100, 180)',
    #           'rgb(150, 127, 50)',
    #           'rgb(157, 127, 135)']
    colors = generate_colors(board_num * base_num)
    # print(type(colors))
    tmp = []
    for t in range(board_num * base_num):
        tmp.append(colors[t])
    colors = tmp
    # print(colors)
    # exit()
    Tm = [0 for _ in positions]  # Tm表示每个资源的可用时间
    PositionStatus = [0 for _ in positions]  # 表示资源当前被哪个任务占用
    gant = [[] for _ in Tm]
    
    for task in task_id:
        task = t_idx_dic[task]
        # 每个task 会占用多个设备，然后释放多个设备
        # print("taskID:", task, "occupy pos:", tasks[task].position, "release pos:", tasks[task].release_position)
        # 先处理占用的逻辑，再处理释放的逻辑。
        # 占用的逻辑
        begin_time = 0
        for pre in tasks[task].pre:  # 找所有前驱任务完成的最晚时间，还要取所有设备可用的最晚时间; 即前驱任务要完成，且资源要可用
            begin_time = max(begin_time, tasks[t_idx_dic[pre]].available)
        for occ_pos in tasks[task].position:
            pos_id = occ_pos  # task要占用的设备
            posAvailableTime = Tm[p_idx_dic[pos_id]]
            begin_time = max(begin_time, posAvailableTime)
            PositionStatus[p_idx_dic[pos_id]] = tasks[task].id
        for occ_pos in tasks[task].position:
            pos_id = occ_pos
            Tm[p_idx_dic[pos_id]] = begin_time
        tasks[task].available = begin_time + tasks[task].time   # available 记录设备完成工作时间
        tasks[task].start_time = begin_time  # start_time 记录开始占用该设备的时间

        for release_position in tasks[task].release_position:
            release_position = p_idx_dic[release_position]
            if release_position in tasks[task].position:
                gant[release_position].append([begin_time, begin_time+tasks[task].time, str(tasks[task].id)])
                Tm[release_position] = begin_time+tasks[task].time
                # print("S:", begin_time, "E:", begin_time+tasks[task].time, "ID:", str(tasks[task].id))
            else:
                gant[release_position].append([Tm[release_position], begin_time, str(PositionStatus[release_position])])
                # print("S:", Tm[release_position], "E:", begin_time, "ID:", str(PositionStatus[release_position]))
                Tm[release_position] = begin_time
        # print("gant:", gant)
        # print('TM=', Tm)
    print("gant中的信息如下: \n")
    for gan in gant:
        print(gan)
        # print("Tm:",Tm)
    
    print("最优时间：")
    print(max(Tm))

    # # 已知板子数量，每块板对应的任务
    # GantChartParent = []
    # BoardNum = args.task * 2
    # print("board_num:", BoardNum)
    # print("板子与任务之间的对应关系：")
    # board_to_task = [[0,1,2,3,4,5,6],[7,8,9,10,11,12,13]] # board[i]表示其对应的task
    # task_to_process = ["Get From Hotel","Transfer To Peel","Peel","Transfer To Seal",
    #                    "Seal","Transfer To Hotel","Put In Hotel",
    #                    "Get From Hotel","Transfer To Peel","Peel","Transfer To Seal",
    #                    "Seal","Transfer To Hotel","Put In Hotel",]  # 表示task与process之间的对应关系
    # for t in range(BoardNum):
    #     parent_id = t
    #     parent_name = "Plate"+str(t)
    #     AssayProcessId = "Exp_id"
    #     PlateProcessId = "process_id"
    #     StartTime = "start_time"
    #     EndTime = "end_time"
    #     Children = []
    #     for task in board_to_task[t]:
    #         child_id = str(t)+"_"+str(task)
    #         barcode = parent_name
    #         node_id = task
    #         name = task_to_process[task]
    #         instrument_name = tasks[task].machine
    #         start_time = -1
    #         end_time = -1
    #         for gan in gant:
    #             for g in gan:
    #                 if int(g[2])==task:
    #                     start_time = g[0]
    #                     end_time = g[1]
    #         estduration = tasks[task].time
    #         robotGetTime = 10
    #         robotPutTime = 0
    #         color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    #         sub_child = {
    #             "Id":child_id,
    #             "Barcode": barcode,
    #             "AssayProcessId": AssayProcessId,
    #             "PlateProcessId": PlateProcessId,
    #             "PlateProcessNodeId":node_id,
    #             "Name":name,
    #             "InstrumentName":instrument_name,
    #             "StartTime":start_time,
    #             "EndTime": end_time,
    #             "EstDuration": estduration,
    #             "RobotGetTime":robotGetTime,
    #             "RobotPutTime":robotPutTime,
    #             "Color":color
    #         }
    #         Children.append(sub_child)
    #     GantChartSub = {
    #         "Id" : parent_id,
    #         "Name": parent_name,
    #         "AssayProcessId": AssayProcessId,
    #         "PlateProcessId": PlateProcessId,
    #         "StartTime": StartTime,
    #         "EndTime": EndTime,
    #         "Children": Children
    #     }
    #     GantChartParent.append(GantChartSub)
    # a = json.dumps(GantChartParent)
    # print(a)
    # with open('GantChart.json','w', encoding='utf-8') as f:
    #     json.dump(GantChartParent, f, indent=2,ensure_ascii=False)
        
# 甘特图可视化模块
    pyplt = py.offline.plot
    df = create_draw_defination(gant)
    fig = ff.create_gantt(df, colors=colors, index_col='Resource', show_colorbar=True,
                          group_tasks=True)
    pyplt(fig, filename='tmp1.html')
    fig.show()
    end_time = time.time()
    run_time = end_time-start_time
    print("运行时间为：", run_time, "s")
# 算法整体逻辑：
# Generate Task Graph   任务图的构建，每个任务绑定对应的设备。维护设备队列。从初始数据结构中选择有用的属性信息，进行任务图的构建，每个任务表示在板位上的操作
# List Schedule Generate Priority  得到任务图、以及每个任务的执行时间，根据时间以及任务图计算优先级。
# Simulate According To Priority.  维护优先级队列，每次从任务中取优先级最高的任务进行调度，若出现队列为空情况则进行回溯。
