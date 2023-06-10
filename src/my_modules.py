# import plotly as py
import copy
import time
import json
# import plotly.figure_factory as ff
import random


class Task:
    """
    Base Task job.
    """
    def __init__(self, id, occupy) -> None:
        self.id = id
        self.idx = 0
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
        self.sequnceid = None

    def __lt__(self, other):
        """定义<比较操作符"""
        if self.priority == other.priority:
            return self.idx < other.idx
        return self.priority < other.priority


class MachineBase:
    def __init__(self, id) -> None:
        self.id = id
        self.name = ""
        self.type = ""


class Position:
    def __init__(self, id) -> None:
        self.id = id
        self.name = ""
        self.status = 0  # status为0表示空闲、为1表示忙碌。
        self.machine = None
        self.machineId = None


def dfs(tasks, task_id, t_idx_dic):
    max_time = 0
    for next in tasks[t_idx_dic[task_id]].next:
        max_time = max(max_time, tasks[t_idx_dic[next]].priority)
    tasks[t_idx_dic[task_id]].priority = tasks[t_idx_dic[task_id]].time + max_time
    for pre in tasks[t_idx_dic[task_id]].pre:
        tasks = dfs(tasks, pre, t_idx_dic)
    return tasks


def dfs2(tasks, task_id, t_idx_dic):
    max_time = 0
    for pre in tasks[t_idx_dic[task_id]].pre:
        max_time = max(max_time, tasks[t_idx_dic[pre]].priority)
    tasks[t_idx_dic[task_id]].priority = tasks[t_idx_dic[task_id]].time + max_time
    for next in tasks[t_idx_dic[task_id]].next:
        tasks = dfs2(tasks, next, t_idx_dic)
    return tasks


def generate_colors(x):
    colors = []
    for i in range(x):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        color = f'rgb({r}, {g}, {b})'
        colors.append(color)
    return colors


def data_read():
    # Data Input Process
    with open("./data_low.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    tasks_json = data['assaysmodel'][0]['tasks']
    positions_json = data['assaysmodel'][0]['positions']
    machines_json = data['assaysmodel'][0]['machines']
    assayssid = data['assaysmodel'][0]['assaysid']
    base_num = len(tasks_json)
    board_num = 50  # board_num 表示该任务图需要复制多少份
    window_size = 2  # window_size 表示每次排几个任务
    heuristics = 0
    t_idx_dic = {}  # 表示Id在数组中的下标
    p_idx_dic = {}
    m_idx_dic = {}
    tasks = [Task(t, []) for t in range(board_num * base_num)]
    positions = [Position(p) for p in range(len(positions_json))]
    machines = [MachineBase(m+1) for m in range(len(machines_json))]
    plateprocesses = []   # 用于记录有几个不同的泳道，一个泳道就是一个板子
    for tidx, task in enumerate(tasks):
        if tidx >= base_num:
            break
        task.id = tasks_json[tidx]['id']            # id表示task的id
        task.idx = tidx                             # tidx表示在数组中的下标
        t_idx_dic[task.id] = tidx                   # id到tidx的映射
        task.occupy = tasks_json[tidx]['occupy']
        task.release = tasks_json[tidx]['release']
        task.time = tasks_json[tidx]['time']
        task.task_name = tasks_json[tidx]['taskname']
        task.PlateProcess = tasks_json[tidx]['plateprocess']
        task.pre = tasks_json[tidx]['pre']
        task.next = tasks_json[tidx]['next']
        task.nodeid = tasks_json[tidx]['nodeid']
        task.robotgettime = tasks_json[tidx]['robotgettime']
        task.robotputtime = tasks_json[tidx]['robotputtime']
        task.sequnceid = tasks_json[tidx]['sequnceid']
        task.steps = tasks_json[tidx]['steps']
        task.sequncestepid = tasks_json[tidx]['sequncestepid']
        task.taskplates = tasks_json[tidx]['taskplates']
        task.color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        task.processtype = tasks_json[tidx]['processtype']
        # task.timeout = tasks_json[tidx]['timeout']
        task.bind = 0
        if "Incubator" in task.task_name or task.task_name == "Get From Hotel" or task.idx == 28 or task.idx == 27 or task.idx == 26:
            task.bind = 1
        if task.PlateProcess not in plateprocesses:
            plateprocesses.append(task.PlateProcess)
    for pidx, pos in enumerate(positions):
        pos.id = positions_json[pidx]['id']
        pos.name = positions_json[pidx]['positionname']
        pos.machine = positions_json[pidx]['machine']
        p_idx_dic[positions_json[pidx]['id']] = pidx
    for midx, mac in enumerate(machines):
        mac.id = machines_json[midx]['id']
        mac.name = machines_json[midx]['name']
        m_idx_dic[machines_json[midx]['id']] = midx
        mac.type = machines_json[midx]['type']
    for task in tasks:  # 通过occupy_dependency和release_dependency标识是否存在占用释放依赖关系
        for occ in task.occupy:
            task.occupy_dependency.append(-1)
        for rel in task.release:
            task.release_dependency.append(-1)
    for t in range(board_num * base_num):  # 复制的任务的前驱和后继的id对应加上
        if t >= base_num:
            task_base = tasks[t % base_num]
            tasks[t].id = task_base.id + str(t)
            t_idx_dic[tasks[t].id] = t
            tasks[t].idx = t
            tasks[t].occupy = task_base.occupy
            tasks[t].release = task_base.release
            tasks[t].time = task_base.time
            tasks[t].task_name = task_base.task_name
            tasks[t].PlateProcess = task_base.PlateProcess
            tasks[t].nodeid = task_base.nodeid
            tasks[t].robotgettime = task_base.robotgettime
            tasks[t].robotputtime = task_base.robotputtime
            tasks[t].sequnceid = task_base.sequnceid
            tasks[t].steps = task_base.steps
            tasks[t].sequncestepid = task_base.sequncestepid
            tasks[t].taskplates = task_base.taskplates
            tasks[t].color = task_base.color
            # tasks[t].timeout = task_base.timeout
            tasks[t].bind = task_base.bind
            tasks[t].processtype = task_base.processtype
    for t in range(board_num * base_num):  # 复制的任务的前驱和后继的id对应加上
        if t >= base_num:
            task_base = tasks[t % base_num]
            tasks[t].pre = [tasks[t_idx_dic[_]+int(t//base_num*base_num)].id for _ in task_base.pre]
            tasks[t].next = [tasks[t_idx_dic[_]+int(t//base_num*base_num)].id for _ in task_base.next]
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
    return tasks, positions, machines, plateprocesses, t_idx_dic, p_idx_dic, m_idx_dic, assayssid, base_num, board_num, heuristics, window_size


def list_scheduling(tasks, t_idx_dic):
    # L1 Scheduling
    pre_END = []  # pre_END 表示最终节点的前驱节点。
    for t in tasks:
        if not t.next:
            pre_END.append(t.id)
    for node in pre_END:
        tasks = dfs(tasks, node, t_idx_dic)
    print("tasks' priority are shown as follows:")
    return tasks


def list_scheduling2(tasks, t_idx_dic):
    # L1 Scheduling
    START_NEXT = []  # START_NEXT 表示开始节点的后续节点。
    for t in tasks:
        if not t.pre:
            START_NEXT.append(t.id)
    for node in START_NEXT:
        tasks = dfs2(tasks, node, t_idx_dic)
    print("tasks' priority are shown as follows:")
    for task in tasks:
        if task.bind == 1:
            for t in task.next:
                tasks[t_idx_dic[t]].priority = tasks[t_idx_dic[t]].priority + 100000000
    return tasks


def Initialize(tasks, begin, end, positions, t_idx_dic, p_idx_dic):
    # 1. Initialize
    import heapq
    q = []
    pq_tmp = []
    for idx, task in enumerate(tasks):
        if idx >= begin and idx < end:  # 初始化时只考虑窗口内的任务加入到队列里。
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
    return tasks, positions, i, Finished, SAVED_INFORMATION, SAVED_CUR_TASK, SAVED_CUR_TASK_STATUS, SAVED_CUR_RESOURCE_POSITIONS, SAVED_CUR_RESOURCE_MACHINES, SAVED_PRE_DECISIONS, SAVED_PRIOR_QUEUE, TASK_SELECT, DEAD, step, q, pq_tmp


def Run(tasks, start_task_id, end_task_id, positions, machines, q, heapq, SAVED_PRE_DECISIONS, step, SAVED_CUR_TASK_STATUS, SAVED_CUR_RESOURCE_POSITIONS, SAVED_CUR_RESOURCE_MACHINES, SAVED_PRIOR_QUEUE, t_idx_dic, p_idx_dic, DEAD, heuristics, TASK_SELECT, SAVED_CUR_TASK, base_num):
    task_list = [i for i in range(0, end_task_id)]
    PROGRAM_REP_TIME = 0
    cnt = 0
    while len(q) > 0:
        cnt += 1
        task = heapq.heappop(q)  # EACH TIME GET THE PRIOR TASK
        SIG_CALL_BACK = 0  # SIG_CALL_BACK用于记录是否在当前队列中找到了可执行的任务。
        PROGRAM_IN_TIME = time.time()
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
            task = tasks[t_idx_dic[SAVED_CUR_TASK[step]]]  # 需要返回上一步，修改上一步的选择
            # task恢复
            task.status = 0  # 任务设为未执行
            task_list.append(task.idx)
            SAVED_CUR_TASK.pop()
            # position恢复
            # 释放position 恢复
            for pos in task.release_position:
                if "Incubator" in task.task_name:
                    continue
                elif task.task_name == "robot":
                    is_next = 0
                    for pre in task.pre:
                        if "Incubator" in tasks[t_idx_dic[pre]].task_name:
                            is_next = 1
                    if is_next == 1:
                        continue
                positions[p_idx_dic[pos]].status = 1
            # 占用position恢复
            for pos in task.position:  
                if "Incubator" in task.task_name:
                    continue
                elif task.task_name == "robot":
                    is_pre = 0
                    for nxt in task.next:
                        if "Incubator" in tasks[t_idx_dic[nxt]].task_name:
                            is_pre = 1
                    if is_pre == 1:
                        continue
                positions[p_idx_dic[pos]].status = 0
            q_id = copy.deepcopy(SAVED_PRIOR_QUEUE[step])
            q = [tasks[t_idx_dic[t_id]] for t_id in q_id]
            task = heapq.heappop(q)

            SIG_tmp = 0
            while task.id in SAVED_PRE_DECISIONS[step]:
                if task.status == 0:  # 如果上一步选择task，但是不行这个任务没有执行完
                    for idx in range(t_idx_dic[task.id], len(tasks), base_num):
                        if tasks[idx].id not in SAVED_PRE_DECISIONS[step]:
                            SAVED_PRE_DECISIONS[step].append(tasks[idx].id)
                if len(q) == 0 and (task.id in SAVED_PRE_DECISIONS[step]):
                    SIG_tmp = 1
                    break
                task = heapq.heappop(q)
            if SIG_tmp == 0:
                break
        PROGRAM_OUT_TIME = time.time()
        PROGRAM_REP_TIME += PROGRAM_OUT_TIME-PROGRAM_IN_TIME
        SAVED_PRE_DECISIONS[step].append(copy.deepcopy(task.id))
        SAVED_CUR_TASK.append(task.id)
        # print("!!!", task.idx, " ", task.task_name, t_idx_dic[task.id], "base:", len(SAVED_PRE_DECISIONS[step]), "mod:", t_idx_dic[task.id] % base_num)
        step += 1
        # 如果当前task没有设置依赖约束，则从所有可用设备中选择第一个可用设备占用
        tmp_machine = []
        tmp_position = []
        for occ_id, occ in enumerate(task.occupy):
            if task.occupy_dependency[occ_id] == -1:
                for pos in occ:  # 从里面选择第一个可用的
                    if positions[p_idx_dic[pos]].status == 0:
                        tmp_position.append(positions[p_idx_dic[pos]].id)
                        tmp_machine.append(positions[p_idx_dic[pos]].machine)
                        if "Incubator" in task.task_name:
                            continue
                        elif task.task_name == "robot":
                            is_pre = 0
                            for nxt in task.next:
                                if "Incubator" in tasks[t_idx_dic[nxt]].task_name:
                                    is_pre = 1
                            if is_pre == 1:
                                continue
                        positions[p_idx_dic[pos]].status = 1
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
        tmp_rel_machine = []
        tmp_rel_position = []
        # 当前任务释放设备
        for rel_id, rel in enumerate(task.release):
            if len(rel) > 1 and task.release_dependency[rel_id] != -1:
                pre_pos = tasks[task.release_dependency[t_idx_dic[rel_id]]].position
                for cur_pos in rel:
                    if cur_pos in pre_pos:
                        positions[cur_pos].status = 0
                        tmp_rel_position.append(cur_pos)
                        tmp_rel_machine.append(positions[cur_pos].machine)
            else:
                for pos in rel:
                    tmp_rel_position.append(pos)
                    tmp_rel_machine.append(positions[p_idx_dic[pos]].machine)
                    if "Incubator" in task.task_name:
                        continue
                    elif task.task_name == "robot":
                        is_next = 0
                        for pre in task.pre:
                            if "Incubator" in tasks[t_idx_dic[pre]].task_name:
                                is_next = 1
                        if is_next == 1:
                            continue
                    positions[p_idx_dic[pos]].status = 0
        task.release_machine = copy.deepcopy(tmp_rel_machine)
        task.release_position = copy.deepcopy(tmp_rel_position)
        task.status = 1
        tmpPos = [0 for _ in range(len(positions))]
        for pos_id, pos in enumerate(positions):
            tmpPos[pos_id] = pos.status
        # print("pos状态为:", tmpPos)
        # 更新队列
        i = 0
        other_q = []
        while i < len(q):  # 输出一下当前优先队列里的内容，用于调试   先将优先队列内容清空，看所有任务当前是否能执行，后续可以对这里进行一定的优化工作。
            other_q.append(t_idx_dic[heapq.heappop(q).id])
        # task_state = []
        # for t in tasks:
        #     task_state.append(t.status)
        # if DEAD == 1:
        #     print(task.id, "其他优先队列中的task:", other_q)
        #     print("task_state:", task_state)
        pq_list = []
        # end_task_id = min(end_task_id, len(tasks))
        for idx in task_list:  #
            t = tasks[idx]
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
        SAVED_PRIOR_QUEUE[step] = [t.id for t in q]
        Finished_task = []
        for idx in task_list:
            if tasks[idx].status == 1:
                Finished_task.append(t.id)
        # print("step:", step, "Selected_Task:", task.task_name, "Priority_queue:", pq_list, "Finished:", Finished_task, "occ:", task.occupy, "rel:", task.release)
        # print("step:", step, "Selected_Task:", task.task_name, " id: ", t_idx_dic[task.id])
        # posS = []
        # for p in positions:
        #     posS.append(p.status)
        # print("pos_status:", posS)
        TASK_SELECT[step] = Finished_task
        flag = 0
        if len(q) == 0 and step == len(tasks):
            break
        task_list.remove(task.idx)
        if task.idx + base_num < len(tasks) and (task.idx+base_num not in task_list):
            task_list.append(task.idx + base_num)
        if len(q) == 0 and step != len(tasks):  # 这里主要做的是撤销操作、。
            # CALL BACK
            # print("CALL BACK", len(task_list))
            if step == 0:
                print('请检查输入是否符合逻辑！')
                exit()
            step -= 1
            # 首先恢复到执行该task之前的状态
            task.status = 0  # 任务设为未执行
            task_list.append(task.idx)
            SAVED_CUR_TASK.pop()
            # position恢复
            # 释放position 恢复
            for pos in task.release_position:
                if "Incubator" in task.task_name:
                    continue
                elif task.task_name == "robot":
                    is_next = 0
                    for pre in task.pre:
                        if "Incubator" in tasks[t_idx_dic[pre]].task_name:
                            is_next = 1
                    if is_next == 1:
                        continue
                positions[p_idx_dic[pos]].status = 1
            # 占用position恢复
            for pos in task.position:
                if "Incubator" in task.task_name:
                    continue
                elif task.task_name == "robot":
                    is_pre = 0
                    for nxt in task.next:
                        if "Incubator" in tasks[t_idx_dic[nxt]].task_name:
                            is_pre = 1
                    if is_pre == 1:
                        continue
                positions[p_idx_dic[pos]].status = 0
            # 优先队列恢复
            heapq.heappush(q, task)
        # if step % 1 == 0:
        #     print("进度：", step, "/", len(tasks))
        # PROGRAM_END_TIME = time.time()
        # sum += PROGRAM_END_TIME - PROGRAM_START_TIME
        # print("sum Time:", PROGRAM_END_TIME - PROGRAM_START_TIME, "AVG Time:",  sum/cnt, "s")
    print("回溯:", PROGRAM_REP_TIME, " ", cnt)
    return tasks, positions, machines, step, SAVED_CUR_TASK_STATUS, TASK_SELECT, SAVED_CUR_TASK


def ResultsOutput(tasks, step, SAVED_CUR_TASK_STATUS, TASK_SELECT, SAVED_CUR_TASK, board_num, base_num, positions, machines, t_idx_dic, p_idx_dic, m_idx_idc, plateprocesses):
    if step != len(tasks):
        print("exsiting deadlock !")
    else:
        print("successfully compute solution !")
    task_id = SAVED_CUR_TASK
    tsk_tmp = []
    for tid in task_id:
        tsk_tmp.append(t_idx_dic[tid])
    print("task分配资源的顺序为:")
    print(len(task_id))
    print(tsk_tmp)
    colors = generate_colors(board_num * base_num)
    tmp = []
    for t in range(board_num * base_num):
        tmp.append(colors[t])
    colors = tmp
    Tm = [0 for _ in positions]  # Tm表示每个资源的可用时间
    PositionStatus = [0 for _ in positions]  # 表示资源当前被哪个任务占用
    for task in task_id:
        task = t_idx_dic[task]  # task表示当前任务的下标
        begin_time = 0
        for pre in tasks[task].pre:  # 找所有前驱任务完成的最晚时间，还要取所有设备可用的最晚时间; 即前驱任务要完成，且资源要可用
            begin_time = max(begin_time, tasks[t_idx_dic[pre]].available)
        for occ_pos in tasks[task].position:
            pos_id = occ_pos  # task要占用的资源
            posAvailableTime = Tm[p_idx_dic[pos_id]]
            begin_time = max(begin_time, posAvailableTime)
            PositionStatus[p_idx_dic[pos_id]] = t_idx_dic[tasks[task].id]
        for occ_pos in tasks[task].position:
            pos_id = occ_pos
            if "Incubator" in tasks[task].task_name:
                continue
            elif tasks[task].task_name == "robot":
                is_pre = 0
                for nxt in tasks[task].next:
                    if "Incubator" in tasks[t_idx_dic[nxt]].task_name:
                        is_pre = 1
                if is_pre == 1:
                    continue
            Tm[p_idx_dic[pos_id]] = begin_time
        if "Incubator" in tasks[task].task_name:
            begin_time = tasks[t_idx_dic[tasks[task].pre[0]]].start_time
            tasks[task].available = begin_time + tasks[task].time   # available 记录设备完成工作时间
            tasks[task].start_time = begin_time  # start_time 记录开始占用该设备的时间
        else:
            tasks[task].available = begin_time + tasks[task].time   # available 记录设备完成工作时间
            tasks[task].start_time = begin_time  # start_time 记录开始占用该设备的时间
        for release_position in tasks[task].release_position:
            release_id = p_idx_dic[release_position]
            if "Incubator" in tasks[task].task_name:
                continue
            elif tasks[task].task_name == "robot":
                is_next = 0
                for pre in tasks[task].pre:
                    if "Incubator" in tasks[t_idx_dic[pre]].task_name:
                        is_next = 1
                if is_next == 1:
                    continue
            if release_position in tasks[task].position:
                Tm[release_id] = begin_time+tasks[task].time
            else:
                Tm[release_id] = begin_time
        # print("task_id:", task, " begin:", tasks[task].start_time, " end:", tasks[task].available)
    print("最优时间：", max(Tm))

    #  1 按顺序遍历所有任务，记录资源占用时间段。
    #  Firstly ， data structure
    TimeOcc = [[] for _ in positions]
    for task in task_id: 
        task = t_idx_dic[task]  # task表示当前任务的下标
        for occ_pos in tasks[task].position:
            pos_id = occ_pos
            if "Incubator" not in tasks[task].task_name:
                TimeOcc[p_idx_dic[pos_id]].append([tasks[task].start_time, tasks[task].available, tasks[task].idx])
    for L in TimeOcc:
        L.sort()
    print("TimeOcc:")
    print(TimeOcc[19])
    board_to_task = [[] for _ in range(len(plateprocesses) * board_num)]
    for tid, task in enumerate(tasks):
        if tid < base_num:
            for idx, process in enumerate(plateprocesses):
                for plate in task.taskplates:
                    if plate == process:
                        board_to_task[idx].append(t_idx_dic[task.id])
        else:
            b_idx = tid // base_num * len(plateprocesses)
            for idx, process in enumerate(plateprocesses):
                for plate in task.taskplates:
                    if plate == process:
                        board_to_task[idx + b_idx].append(t_idx_dic[task.id])
    for idx in range(len(board_to_task)-1, -1, -1):
        boardi = board_to_task[idx]
        for task_idx in range(len(boardi)-1, -1, -1):  # 倒序遍历任务
            task_idx = boardi[task_idx]
            if len(tasks[task_idx].next) > 0:  # 如果有后续节点，首先计算出后续任务最早开始时间
                min_start_time = 9999999999999
                for nxt in tasks[task_idx].next:
                    min_start_time = min(min_start_time, tasks[t_idx_dic[nxt]].start_time)
                if tasks[task_idx].available < min_start_time:  # 当前任务和后续任务没有连接起来，则判断是否能向后推迟
                    # 需要看该任务占用的所有资源，是否都能向后推迟。
                    min_pos_time = 9999999999999
                    for pos_id in tasks[task_idx].position:
                        pos_id = p_idx_dic[pos_id]
                        for idx, occ in enumerate(TimeOcc[pos_id]):
                            if tasks[task_idx].idx == occ[2] and idx+1 < len(TimeOcc[pos_id]):
                                nxt = TimeOcc[pos_id][idx+1][0]
                                min_pos_time = min(min_pos_time, nxt)
                            elif tasks[task_idx].idx == occ[2] and idx+1 == len(TimeOcc[pos_id]):
                                min_pos_time = min_start_time
                    during = tasks[task_idx].available - tasks[task_idx].start_time
                    pre = tasks[task_idx].start_time
                    pree = tasks[task_idx].available
                    tasks[task_idx].available = min(min_start_time, min_pos_time)
                    tasks[task_idx].start_time = tasks[task_idx].available - during
                    if tasks[task_idx].start_time != pre:
                        print(tasks[task_idx].start_time, " ", tasks[task_idx].available, " != ", pre, " ", pree)
                    for pos_id in tasks[task_idx].position:
                        pos_id = p_idx_dic[pos_id]
                        for idx, occ in enumerate(TimeOcc[pos_id]):
                            if tasks[task_idx].idx == occ[2]:
                                occ[0] = tasks[task_idx].start_time
                                occ[1] = tasks[task_idx].available
    for idx in range(len(board_to_task)-1, -1, -1):
        boardi = board_to_task[idx]
        # print(boardi)
        for t_idx in range(len(boardi)-1, -1, -1):  # 倒序遍历任务
            task_idx = boardi[t_idx]
            if t_idx-1 > 0 and "robot" not in tasks[task_idx].task_name:
                if "robot" in tasks[boardi[t_idx-1]].task_name:
                    pre_task_end_time = tasks[boardi[t_idx-2]].available
                    tasks[task_idx].start_time = pre_task_end_time
                else:
                    pre_task_end_time = tasks[boardi[t_idx-1]].available
                    tasks[task_idx].start_time = pre_task_end_time
    # exit()

def GenerateGant(plateprocesses, board_num, tasks, base_num, t_idx_dic, assayssid, machines):
    GantChartParent = []
    BoardNum = len(plateprocesses)
    print("board_num:", BoardNum)
    board_to_task = [[] for _ in range(len(plateprocesses) * board_num)]  # board[i]表示第i块板对应的任务
    for tid, task in enumerate(tasks):
        if tid < base_num:
            for idx, process in enumerate(plateprocesses):
                for plate in task.taskplates:
                    if plate == process:
                        board_to_task[idx].append(t_idx_dic[task.id])
        else:
            b_idx = tid // base_num * len(plateprocesses)
            for idx, process in enumerate(plateprocesses):
                for plate in task.taskplates:
                    if plate == process:
                        board_to_task[idx + b_idx].append(t_idx_dic[task.id])
    task_to_process = []
    for task in tasks:
        task_to_process.append(task.task_name)
    import datetime
    now = datetime.datetime.now()
    sequences = {}
    for task in tasks:
        if task.sequnceid != "":
            if task.sequnceid not in sequences:
                sequences[task.sequnceid] = task.steps
    for idx in range(len(plateprocesses) * board_num):   # 遍历一块板的，一块板是一个结构体。需要知道有几块板，以及每个任务属于哪块板
        t = plateprocesses[idx % len(plateprocesses)]   # plateprocess中共有两个元素
        parent_id = t + "board" + str(idx + 1)
        parent_name = "Plate"+str(idx+1)
        AssayProcessId = assayssid
        PlateProcessId = t
        StartTime = "start_time"
        mintime = datetime.datetime.max
        maxtime = now
        EndTime = "end_time"
        Children = []
        for task in board_to_task[idx]:
            if tasks[task].task_name == "robot":
                continue
            if tasks[task].sequnceid != "":
                continue
            child_id = tasks[task].id
            pid = parent_id
            barcode = parent_name
            node_id = tasks[task].nodeid
            name = task_to_process[task]
            instrument_name = tasks[task].machine
            for m in machines:
                if len(tasks[task].machine) > 0 and m.id == tasks[task].machine[0]:
                    instrument_name = m.name

            mintime = min(mintime, datetime.timedelta(seconds=int(tasks[task].start_time)) + now)
            maxtime = max(maxtime, datetime.timedelta(seconds=int(tasks[task].available)) + now)
            start_time = (datetime.timedelta(seconds=int(tasks[task].start_time)) + now).strftime("%Y-%m-%d %H:%M:%S")
            end_time = (datetime.timedelta(seconds=int(tasks[task].available)) + now).strftime("%Y-%m-%d %H:%M:%S")

            estduration = tasks[task].time
            timespan = int((tasks[task].available - tasks[task].start_time))
            minute, second = divmod(timespan, 60)
            hour, minute = divmod(minute, 60)
            hour = hour % 60
            timespan_trans = '{:02}:{:02}:{:02}'.format(hour, minute, second)
            robotGetTime = tasks[task].robotgettime
            robotPutTime = tasks[task].robotputtime
            color = tasks[task].color  # "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
            sub_child = {
                "Id": child_id,
                "Barcode": barcode,
                "Pid": pid,
                "Percentage": 0,
                "AssayProcessId": AssayProcessId,
                "PlateProcessId": PlateProcessId,
                "PlateProcessNodeId": node_id,
                "Name": name,
                "InstrumentName": instrument_name,
                "StartTime": start_time,
                "EndTime": end_time,
                "EstDuration": estduration,
                "TimeSpan": timespan_trans,
                "RobotGetTime": robotGetTime,
                "RobotPutTime": robotPutTime,
                "Color": color,
                "TaskStartTime": int(tasks[task].start_time),
                "TaskEndTime": int(tasks[task].available)
            }
            Children.append(sub_child)
        # print("mintime = ", mintime.strftime("%Y-%m-%d %H:%M:%S"))
        # 每个plateProcess中所有任务记录下
        for seq in sequences:  # 对于每个sequence中如果有task属于该sequence，则生成一个任务
            # print("!!!", sequences[seq])
            stask = []
            for task in board_to_task[idx]:  # 遍历这块板，也就是这个泳道包含的任务
                if tasks[task].sequnceid == seq:
                    for _ in sequences[seq]:
                        if tasks[task].sequncestepid == _['id']:
                            stask.append([task, _['sortnum']])
                            break
            stask = sorted(stask, key=lambda x: x[1])
            if stask != []:
                child_id = tasks[stask[0][0]].id
                node_id = tasks[stask[0][0]].nodeid
                instrument_name = seq
                start_time = (datetime.timedelta(seconds=int(tasks[stask[0][0]].start_time)) + now).strftime("%Y-%m-%d %H:%M:%S")
                end_time = (datetime.timedelta(seconds=int(tasks[stask[len(stask)-1][0]].available)) + now).strftime("%Y-%m-%d %H:%M:%S")
                timespan = int((tasks[stask[len(stask)-1][0]].available - tasks[stask[0][0]].start_time))
                estduration = timespan
                minute, second = divmod(timespan, 60)
                hour, minute = divmod(minute, 60)
                hour = hour % 60
                timespan_trans = '{:02}:{:02}:{:02}'.format(hour, minute, second)
                color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                sub_child = {
                    "Id": child_id,
                    "Barcode": parent_name,
                    "Pid": parent_id,
                    "Percentage": 0,
                    "AssayProcessId": AssayProcessId,
                    "PlateProcessId": PlateProcessId,
                    "PlateProcessNodeId": node_id,
                    "Name": "Run Sequences",
                    "InstrumentName": instrument_name,
                    "StartTime": start_time,
                    "EndTime": end_time,
                    "EstDuration": estduration,
                    "TimeSpan": timespan_trans,
                    "RobotGetTime": 0,
                    "RobotPutTime": 0,
                    "Color": color,
                    "TaskStartTime": int(tasks[stask[0][0]].start_time),
                    "TaskEndTime": int(tasks[stask[len(stask)-1][0]].available)
                }
                Children.append(sub_child)
        for child in Children:
            time_cur = child['EstDuration']
            child['Percentage'] = time_cur / ((maxtime-mintime).total_seconds())
        # for idx in range(len(Children)-1, -1, -1):
        #     child = Children[idx]   
        #     if idx + 1 < len(Children):
        #         TaskStartTime = child['TaskStartTime']
        #         TaskEndTime = child['TaskEndTime']
        #         nextChild = Children[idx + 1]
        #         nTaskStartTime = nextChild['TaskStartTime']
        #         if TaskEndTime < nTaskStartTime:
        #             TaskStartTime = TaskStartTime + nTaskStartTime - TaskEndTime
        #             TaskEndTime = nTaskStartTime
        #             child['TaskStartTime'] = TaskStartTime
        #             child['TaskEndTime'] = TaskEndTime
        #             child['StartTime'] = (datetime.timedelta(seconds=int(TaskStartTime)) + now).strftime("%Y-%m-%d %H:%M:%S")
        #             child['EndTime'] = (datetime.timedelta(seconds=int(TaskEndTime)) + now).strftime("%Y-%m-%d %H:%M:%S")
        StartTime = mintime.strftime("%Y-%m-%d %H:%M:%S")
        EndTime = maxtime.strftime("%Y-%m-%d %H:%M:%S")
        GantChartSub = {
            "Id": parent_id,
            "Name": parent_name,
            "AssayProcessId": AssayProcessId,
            "PlateProcessId": PlateProcessId,
            "StartTime": StartTime,
            "EndTime": EndTime,
            "Children": Children
        }
        GantChartParent.append(GantChartSub)
    with open('GantChart.json', 'w', encoding='utf-8') as f:
        json.dump(GantChartParent, f, indent=2, ensure_ascii=False)
    return GantChartParent


def TestGant(GantChartParent):
    print(GantChartParent)


def create_draw_defination(tasks):
    df = []
    for task in tasks:
        operation = {}
        operation['Task'] = task.task_name
        operation['Start'] = task.start_time
        operation['Finish'] = task.available
        operation['Resource'] = task.idx
        df.append(operation)
    return df


def display(tasks):
    # pyplt = py.offline.plot
    import plotly.figure_factory as ff
    df = create_draw_defination(tasks)
    fig = ff.create_gantt(df)
    # pyplt(fig, filename='tmp1.html')
    fig.show()

def record_task(tasks):
    my_list = []
    for task in tasks:
        my_list.append([t_idx_dic[task.id], task.task_name, task.start_time, task.available])
    with open('my_file.json', 'w') as f:
        json.dump(my_list, f)