# import plotly as py
import copy
import heapq
import random
import string
import json
import datetime


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
        self.platename = None

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
        self.taskstoragedetail = None


class Position:
    def __init__(self, id) -> None:
        self.id = id
        self.name = ""
        self.status = 0  # status为0表示空闲、为1表示忙碌。
        self.machine = None
        self.machineId = None


class Plate:
    def __init__(self) -> None:
        self.id = None
        self.idx = None
        self.hotelname = None
        self.storageconfigid = None
        self.projectinstrumentid = None
        self.barcode = None
        self.previousbarcode = None
        self.labware = None
        self.labwareid = None
        self.physicalposition = None
        self.shelf = None
        self.zone = None
        self.status = None

    def __lt__(self, other):
        """定义<比较操作符"""
        if self.physicalposition == other.physicalposition:
            return self.idx < other.idx
        return self.physicalposition < other.physicalposition


class PlateProcess:
    def __init__(self) -> None:
        self.id = None
        self.storage_idx = None
        self.zone = None


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


def create_graph(tasks_json, tidx, plateprocesses, allboard):  # allboard 表示每个泳道的板子的信息， idx表示在tasks数组中的起始下标。
    tasks = [Task(t, []) for t in range(len(tasks_json))]
    for tidx, task in enumerate(tasks):  # 复制任务基本信息
        task.PlateProcess = tasks_json[tidx]['plateprocess']
        b_idx = plateprocesses.index(task.PlateProcess)
        task.platename = allboard[b_idx]
    return tasks


def generate_random_string(length):
    letters = string.ascii_lowercase  # 可以根据需要使用其他字符集
    return ''.join(random.choice(letters) for _ in range(length))


def data_read(data):
    global t_idx_dic, p_idx_dic, m_idx_dic, tasks, step, board_num, base_num, positions, machines, plateprocesses, assayssid
    # Data Input Process
    # with open("./data_low_h.json", "r", encoding="utf-8") as f:
    #     data = json.load(f)
    # headers = {'content-type': 'application/json'}
    # upload_url = 'http://192.168.1.107:5089/home/GetTaskList?projectId=B48A055EF802466DBFC34B1600993BFA'  # first
    # # upload_url = 'http://192.168.1.107:5089/home/GetTaskList?projectId=3ABB3B0A4914479CB0BD3D2889F50EDA'  # second
    # r = requests.get(upload_url, headers=headers)
    # if r.status_code != 200:
    #     print(f'数据接收异常，返回码: {r.status_code}，重新接收数据')
    # if r.content:
    #     recvData = r.content.decode('utf-8')
    #     jsonData = json.loads(recvData)
    #     readData = jsonData
    # else:
    #     print(f'数据为空，返回码: {r.status_code}，请检查数据')
    # data = readData
    tasks_json = data['assaysmodel'][0]['tasks']
    positions_json = data['assaysmodel'][0]['positions']
    machines_json = data['assaysmodel'][0]['machines']
    assayssid = data['assaysmodel'][0]['assaysid']
    base_num = len(tasks_json)
    p_idx_dic = {}
    m_idx_dic = {}
    positions = [Position(p) for p in range(len(positions_json))]
    machines = [MachineBase(m+1) for m in range(len(machines_json))]
    for pidx, pos in enumerate(positions):  # position资源基本信息
        pos.id = positions_json[pidx]['id']
        pos.name = positions_json[pidx]['positionname']
        pos.machine = positions_json[pidx]['machine']
        p_idx_dic[positions_json[pidx]['id']] = pidx
    for midx, mac in enumerate(machines):
        mac.id = machines_json[midx]['id']
        mac.name = machines_json[midx]['name']
        m_idx_dic[machines_json[midx]['id']] = midx
        mac.type = machines_json[midx]['type']
        mac.taskstoragedetail = machines_json[midx]['taskstoragedetail']

    plateprocesses = []  # 这里面存储的是泳道的id
    for tidx in range(base_num):
        task_plates = tasks_json[tidx]['taskplates']
        for plate in task_plates:
            if plate not in plateprocesses:
                plateprocesses.append(plate)
    process_task = [[] for _ in plateprocesses]  # 这里存储的是每个泳道包含的任务的json
    main_process = []
    process_start_task = [[] for _ in process_task]  # 存储每个泳道最开始的任务
    process_end_task = [[] for _ in process_task]  # 存储每个泳道结尾的任务
    for tidx in range(base_num):
        task_plates = tasks_json[tidx]['taskplates']
        for plate in task_plates:
            if tidx == 0:
                main_process.append(tasks_json[tidx])
            else:
                idx = plateprocesses.index(plate)
                process_task[idx].append(tasks_json[tidx])
                if len(tasks_json[tidx]['pre']) == 0:
                    process_start_task[idx].append(tasks_json[tidx])
                elif len(tasks_json[tidx]['next']) == 0:
                    process_end_task[idx].append(tasks_json[tidx])
    # 首先维护泳道信息、存储设备信息，泳道与存储设备对应信息，存储设备与板子对应信息
    storages = []  # storage 存储的是存储设备的name，记录有多少个存储设备
    for machine in machines_json:
        if machine['instrumentcategory'] == 2:
            storages.append(machine['id'])
    store_board = [[] for _ in storages]  # store_board存储每个存储设备中的板子信息 [] list中存储每个板子的信息，并根据某些关键字进行排序

    print(storages)
    # 填充store_board数组，首先遍历所有存储设备
    for machine in machines_json:
        if machine['instrumentcategory'] == 2:
            taskstoragedetail = machine['taskstoragedetail']
            for board in taskstoragedetail:
                idx = storages.index(machine['id'])
                if board['barcode'] is not None:
                    cur_plate = Plate()
                    cur_plate.id = board['id']
                    cur_plate.idx = len(store_board[idx])
                    cur_plate.hotelname = board['hotelname']
                    cur_plate.storageconfigid = board['storageconfigid']
                    cur_plate.projectinstrumentid = board['projectinstrumentid']
                    cur_plate.barcode = board['barcode']
                    cur_plate.previousbarcode = board['previousbarcode']
                    cur_plate.labware = board['labware']
                    cur_plate.labwareid = board['labwareid']
                    cur_plate.physicalposition = board['physicalposition']
                    cur_plate.shelf = board['shelf']
                    cur_plate.zone = board['zone']
                    cur_plate.status = 0  # 0 表示未被使用 ， 1表示已被使用
                    store_board[idx].append(cur_plate)
    for store in store_board:  # 对每个存储区域中的板进行排序,按照位置作为第一优先级，下标作为第二优先级。
        store = sorted(store)
    # 遍历主泳道，找到其对应的存储区域，从该存储区域中取数据
    main_machine = []  # 主泳道的存储设备
    for store in storages:
        for occ in main_process[0]['occupy']:
            if positions[p_idx_dic[occ[0]]].machine == store:
                if machines[m_idx_dic[store]].id not in main_machine:
                    main_machine.append(machines[m_idx_dic[store]].id)
    main_zone = main_process[0]['zone']

    # 每个泳道对应一个存储设备的idx和存储区域的string
    print("main_machine:", main_machine[0])
    print("main_zone:", main_zone)
    # 找到该区域，维护该区域存储的板子遍历的位置。如果能找到下一个可用的板子，则生成一个任务图。 同时计算其他耗材板的数量，不足需要进行记录哪些不够。
    # 通过main_machine和main_zone找到位置，每个存储设备有一个遍历下标。看当前设备遍历到的板子。
    main_idx = storages.index(main_machine[0])  # main_idx表示主泳道占用的设备在storage数组中的位置
    pointer = [0 for _ in plateprocesses]  # pointer表示每个泳道的指针，指向当前下一个要取的内容  pointer[i]表示当前泳道i的指针
    # 每个泳道需要对应计算出一块板子。
    tasks = []
    while pointer[0] < len(store_board[main_idx]):  # 主泳道是第 0 个
        while pointer[0] < len(store_board[main_idx]) and (store_board[main_idx][pointer[0]].zone != main_zone or store_board[main_idx][pointer[0]].status == 1):
            pointer[0] += 1
        if pointer[0] < len(store_board[main_idx]):
            main_board = store_board[main_idx][pointer[0]]
            store_board[main_idx][pointer[0]].status = 1
            allboard = [main_board]
            for p_idx, res_process in enumerate(process_start_task):
                if p_idx == 0:
                    continue
                res_machine = []
                for store in storages:
                    for occ in res_process[0]['occupy']:
                        if positions[p_idx_dic[occ[0]]].machine == store:
                            if machines[m_idx_dic[store]].id not in res_machine:
                                res_machine.append(machines[m_idx_dic[store]].id)
                res_zone = res_process[0]['zone']
                res_idx = storages.index(res_machine[0])
                while pointer[p_idx] < len(store_board[res_idx]) and (store_board[res_idx][pointer[p_idx]].zone != res_zone or store_board[res_idx][pointer[p_idx]].status == 1):
                    pointer[p_idx] += 1
                if pointer[p_idx] < len(store_board[res_idx]):
                    store_board[res_idx][pointer[p_idx]].status = 1
                    res_board = store_board[res_idx][pointer[p_idx]]
                    allboard.append(res_board)
                else:
                    print("耗材板不足")
                    res_board = Plate()
                    res_board.barcode = "待补充耗材板" + generate_random_string(5)
                    allboard.append(res_board)
                    exit()
            # allboard表示每个泳道用到的板子
            new_graph = create_graph(tasks_json, len(tasks), plateprocesses, allboard)
            tasks.extend(new_graph)
        else:
            break
        pointer[0] += 1
    # 其他泳道每个泳道的板子从哪个地方出来。

    board_num = int(len(tasks)/base_num)
    window_size = 1  # window_size 表示每次排几个任务
    t_idx_dic = {}  # 表示Id在数组中的下标
    for tidx, task in enumerate(tasks):  # 复制任务基本信息
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
    return tasks, positions, machines, plateprocesses, t_idx_dic, p_idx_dic, m_idx_dic, base_num, board_num, window_size


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
    global SAVED_CUR_TASK, step
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
            # flag3 = 1
            # bind_task_list = []
            # tmp = task
            # while tmp.bind == 1:
            #     bind_task_list.append[tmp]
            #     if len(tmp.next) == 0:
            #         break
            #     tmp = tasks[t_idx_dic[tmp.next[0]]]
            # for tmp in bind_task_list:
                # 如果都满足，flag3才等于1，但是也有可能绑定任务链比较长，前面的会释放资源任务链后面的可以执行了。 
                # 判断list 中的任务是否占用的资源都满足，如果都满足，那么这个绑定任务才可以执行。

            if flag == 1 and flag2 == 1:
                heapq.heappush(q, task)
                pq_tmp.append(t_idx_dic[task.id])
    i = 0
    Finished = []
    SAVED_CUR_TASK = []  # 存储的是当前步选择的设备
    SAVED_PRE_DECISIONS = [[] for _ in range(len(tasks)+1)]  # 记录每一个step曾经做过的决策
    SAVED_PRIOR_QUEUE = [[] for _ in range(len(tasks)+1)]  # 记录在做决策前的优先队列的内容
    SAVED_PRIOR_QUEUE[0] = [t.id for t in q]
    step = 0  # step 用于表示当前的步骤
    print("step:", step, "Priority_queue:", pq_tmp)
    return tasks, positions, i, Finished, SAVED_PRE_DECISIONS, SAVED_PRIOR_QUEUE, q


def Run(tasks, end_task_id, positions, machines, q, SAVED_PRE_DECISIONS, SAVED_PRIOR_QUEUE, t_idx_dic, p_idx_dic, base_num):
    task_list = [i for i in range(0, end_task_id)]
    global SAVED_CUR_TASK, step
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
        pq_list = []
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
        # if step > len(q)/2:
        #     GenerateGant(plateprocesses, board_num, tasks, base_num, t_idx_dic, assayssid, machines)
    return tasks, positions, machines


def ResultsOutput(tasks, board_num, base_num, positions, t_idx_dic, p_idx_dic, m_idx_idc, plateprocesses):
    global SAVED_CUR_TASK
    if step != len(tasks):
        print("Still Scheduling !")
    else:
        print("Finished Scheduling !")
    task_id = SAVED_CUR_TASK  # 遍历SAVED_CUR_TASK
    tsk_tmp = []
    for tid in task_id:
        tsk_tmp.append(t_idx_dic[tid])
    print("task分配资源的顺序为:")
    print(len(task_id))
    print(tsk_tmp)
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

    # TimeOcc = [[] for _ in positions]
    # for task in task_id:
    #     task = t_idx_dic[task]  # task表示当前任务的下标
    #     for occ_pos in tasks[task].position:
    #         pos_id = occ_pos
    #         if "Incubator" not in tasks[task].task_name:
    #             TimeOcc[p_idx_dic[pos_id]].append([tasks[task].start_time, tasks[task].available, tasks[task].idx])
    # for L in TimeOcc:
    #     L.sort()
    # board_to_task = [[] for _ in range(len(plateprocesses) * board_num)]
    # for tid, task in enumerate(tasks):
    #     if tid < base_num:
    #         for idx, process in enumerate(plateprocesses):
    #             for plate in task.taskplates:
    #                 if plate == process:
    #                     board_to_task[idx].append(t_idx_dic[task.id])
    #     else:
    #         b_idx = tid // base_num * len(plateprocesses)
    #         for idx, process in enumerate(plateprocesses):
    #             for plate in task.taskplates:
    #                 if plate == process:
    #                     board_to_task[idx + b_idx].append(t_idx_dic[task.id])
    # for idxx in range(len(board_to_task)-1, -1, -1):
    #     boardi = board_to_task[idxx]
    #     for t_idx in range(len(boardi)-1, -1, -1):  # 倒序遍历任务
    #         task_idx = boardi[t_idx]
    #         if len(tasks[task_idx].next) > 0:  # 如果有后续节点，首先计算出后续任务最早开始时间
    #             min_start_time = 9999999999999
    #             for nxt in tasks[task_idx].next:
    #                 min_start_time = min(min_start_time, tasks[t_idx_dic[nxt]].start_time)
    #             if tasks[task_idx].available < min_start_time:  # 当前任务和后续任务没有连接起来，则判断是否能向后推迟
    #                 # 需要看该任务占用的所有资源，是否都能向后推迟。
    #                 min_pos_time = 9999999999999
    #                 for pos_id in tasks[task_idx].position:
    #                     pos_id = p_idx_dic[pos_id]
    #                     for idx, occ in enumerate(TimeOcc[pos_id]):
    #                         if tasks[task_idx].idx == occ[2] and idx+1 < len(TimeOcc[pos_id]):
    #                             nxt = TimeOcc[pos_id][idx+1][0]
    #                             min_pos_time = min(min_pos_time, nxt)
    #                         elif tasks[task_idx].idx == occ[2] and idx+1 == len(TimeOcc[pos_id]):
    #                             min_pos_time = min_start_time
    #                 during = tasks[task_idx].available - tasks[task_idx].start_time
    #                 pre = tasks[task_idx].start_time
    #                 pree = tasks[task_idx].available
    #                 tasks[task_idx].available = min(min_start_time, min_pos_time)
    #                 tasks[task_idx].start_time = tasks[task_idx].available - during
    #                 for pos_id in tasks[task_idx].position:
    #                     pos_id = p_idx_dic[pos_id]
    #                     for idx, occ in enumerate(TimeOcc[pos_id]):
    #                         if tasks[task_idx].idx == occ[2]:
    #                             occ[0] = tasks[task_idx].start_time
    #                             occ[1] = tasks[task_idx].available
    #         if idxx == 1 and task_idx == 40:
    #             print(tasks[task_idx].task_name)
    #             print(boardi)
    #             print(tasks[task_idx].start_time)
    #             print(tasks[task_idx].available)
    #             for b in boardi:
    #                 print(tasks[b].task_name)
    # for idx in range(len(board_to_task)-1, -1, -1):
    #     boardi = board_to_task[idx]
    #     for t_idx in range(len(boardi)-1, -1, -1):  # 倒序遍历任务
    #         task_idx = boardi[t_idx]  # task_idx表示要处理的任务。
    #         if "robot" not in tasks[task_idx].task_name:  # 如果不是robot节点，则和前驱任务链接起来。
    #             min_end_time = 0
    #             for pre in tasks[task_idx].pre:
    #                 if "robot" in tasks[t_idx_dic[pre]].task_name:
    #                     ppre = tasks[t_idx_dic[tasks[t_idx_dic[pre]].pre[0]]]
    #                     min_end_time = max(min_end_time, ppre.available)
    #                 else:
    #                     min_end_time = max(min_end_time, tasks[t_idx_dic[pre]].available)
    #                 tasks[task_idx].start_time = min_end_time


def GenerateGant(plateprocesses, board_num, tasks, base_num, t_idx_dic, machines, now):
    global assayssid
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
    sequences = {}
    for task in tasks:
        if task.sequnceid != "":
            if task.sequnceid not in sequences:
                sequences[task.sequnceid] = task.steps
    for idx in range(len(plateprocesses) * board_num):   # 遍历一块板的，一块板是一个结构体。需要知道有几块板，以及每个任务属于哪块板
        t = plateprocesses[idx % len(plateprocesses)]   # plateprocess中共有两个元素
        parent_id = t + "board" + str(idx + 1)
        for task in board_to_task[idx]:
            if len(tasks[task].pre) == 0:
                parent_name = tasks[task].platename.barcode
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
    # with open('D:\\apache-tomcat-10.1.9\\webapps\\gantchart\\GantChart.json', 'w', encoding='utf-8') as f:
    #     json.dump(GantChartParent, f, indent=2, ensure_ascii=False)
    return GantChartParent


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


def getTmpGant():
    global t_idx_dic, p_idx_dic, m_idx_dic, tasks, step, board_num, \
        base_num, positions, machines, plateprocesses, assayssid, SAVED_CUR_TASK, now
    task_id = SAVED_CUR_TASK  # 遍历SAVED_CUR_TASK
    tsk_tmp = []
    for tid in task_id:
        tsk_tmp.append(t_idx_dic[tid])
    print("task分配资源的顺序为:")
    print(len(task_id))
    print(tsk_tmp)
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

    TimeOcc = [[] for _ in positions]
    for task in task_id:
        task = t_idx_dic[task]  # task表示当前任务的下标
        for occ_pos in tasks[task].position:
            pos_id = occ_pos
            if "Incubator" not in tasks[task].task_name:
                TimeOcc[p_idx_dic[pos_id]].append([tasks[task].start_time, tasks[task].available, tasks[task].idx])
    for L in TimeOcc:
        L.sort()
    #  这里暂时省略的内容： 甘特图向后推迟的内容和补成连续的部分逻辑
    #   ---------------------------------------
    #  待补充逻辑，暂时对流程无影响，优化需要加  。
    GantChartParent = []
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

    sequences = {}
    for task in tasks:
        if task.sequnceid != "":
            if task.sequnceid not in sequences:
                sequences[task.sequnceid] = task.steps
    for idx in range(len(plateprocesses) * board_num):   # 遍历一块板的，一块板是一个结构体。需要知道有几块板，以及每个任务属于哪块板
        t = plateprocesses[idx % len(plateprocesses)]   # plateprocess中共有两个元素
        parent_id = t + "board" + str(idx + 1)
        for task in board_to_task[idx]:
            if len(tasks[task].pre) == 0:
                parent_name = tasks[task].platename.barcode
        AssayProcessId = assayssid
        PlateProcessId = t
        StartTime = "start_time"
        mintime = datetime.datetime.max
        now = datetime.datetime.now()
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
                if tasks[task].machine != -1 and len(tasks[task].machine) > 0 and m.id == tasks[task].machine[0]:
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
            if child_id in task_id and int(tasks[task].available) != int(tasks[task].start_time):
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
                if child_id in task_id and int(tasks[task].available) != int(tasks[task].start_time):
                    Children.append(sub_child)
        for child in Children:
            time_cur = child['EstDuration']
            if ((maxtime-mintime).total_seconds()) != 0:
                child['Percentage'] = time_cur / ((maxtime-mintime).total_seconds())
            else:
                child['Percentage'] = 0
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
        if len(GantChartSub['Children']) != 0:
            GantChartParent.append(GantChartSub)
    with open('D:\\apache-tomcat-10.1.9\\webapps\\gantchart\\GantChart.json', 'w', encoding='utf-8') as f:
        json.dump(GantChartParent, f, indent=2, ensure_ascii=False)
    return GantChartParent
