import plotly as py
import copy
import time
import plotly.figure_factory as ff
class Task:
    """
    Base Task job.
    """
    def __init__(self, id, occupy) -> None:
        self.id = id
        # self.occupy表示占用的设备资源，例如 [1,2,3] 表示从1，2，3中选择一个
        self.occupy = occupy
        self.release = []
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
        self.release_position = [] # 表示当前task释放的position
        self.dependency = -1  # dependency 表示当前task和哪个task需要有相同的设备
        self.available = 0
        self.start_time = 0
        self.heuristic = 0
    def __lt__(self, other):
        """定义<比较操作符"""
        if self.priority == other:
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
    parser.add_argument('--task', type=int, nargs='+')
    parser.add_argument('--machine', type=int, nargs='+')
    parser.add_argument('--position', type=int, nargs='+')
    parser.add_argument('--heuristics', type=int)
    return parser
def dfs(tasks,task_id):
    max_time = 0
    for next in tasks[task_id].next:
        max_time = max(max_time, tasks[next].priority)
    tasks[task_id].priority = tasks[task_id].time + max_time
    for pre in tasks[task_id].pre :
        tasks = dfs(tasks, pre)
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

if __name__ == "__main__":
    args = make_parser().parse_args()
    # 初始化tasks
    tasks = [Task(t, []) for t in args.task]
    positions = [Position(p) for p in args.position]
    heuristics = args.heuristics
    # 设置每个task需要的资源,暂时手动设置一个例子，测试算法是否可行，后面会从数据结构中生成下面的内容。
    tasks[0].occupy = [1]   # 0：冰箱1出Plate
    tasks[0].release = []
    tasks[1].occupy = [2]   # 机械臂取Plate
    tasks[1].release = [1]
    tasks[2].occupy = [3,5,7]  # 工作站加载Plate
    tasks[2].release = [2]
    tasks[3].occupy = [9]    # 冰箱2出Mix1
    tasks[3].release = []
    tasks[4].occupy = [2]    # 机械臂取Mix1
    tasks[4].release = [9]
    tasks[5].occupy = [4,6,8]   # 工作站加载Mix1
    tasks[5].release = [2]
    tasks[6].occupy = [11, 12]        # 工作站工作
    tasks[6].release = [11, 12]
    tasks[7].occupy = [2]        # 工作站卸载对应的Plate
    tasks[7].release = [3,5,7]
    tasks[8].occupy = [1]        # 机械臂放Plate到Hotel
    tasks[8].release = [2,1]
    tasks[9].occupy = [2]        # 工作站unload mix1
    tasks[9].release = [4,6,8]
    tasks[10].occupy = [10]       # 机械臂放Plate到trash
    tasks[10].release = [2,10]
    tasks[11].occupy = [1]   # 0：冰箱1出Plate1
    tasks[11].release = []
    tasks[12].occupy = [2]   # 机械臂取Plate1
    tasks[12].release = [1]
    tasks[13].occupy = [3,5,7]  # 工作站加载Plate1
    tasks[13].release = [2]
    tasks[14].occupy = [9]    # 冰箱2出Mix1_1
    tasks[14].release = []
    tasks[15].occupy = [2]    # 机械臂取Mix1_1
    tasks[15].release = [9]
    tasks[16].occupy = [4,6,8]   # 工作站加载Mix1_1
    tasks[16].release = [2]
    tasks[17].occupy = [11,12]        # 工作站工作
    tasks[17].release = [11,12]
    tasks[18].occupy = [2]        # 工作站卸载对应的Plate1
    tasks[18].release = [3,5,7] 
    tasks[19].occupy = [1]        # 机械臂放Plate1到Hotel
    tasks[19].release = [2,1]
    tasks[20].occupy = [2]        # 工作站unload mix1_1
    tasks[20].release = [4,6,8]
    tasks[21].occupy = [10]       # 机械臂放Plate1到trash
    tasks[21].release = [2,10]

    # 构建TaskGraph，连接task之间的有向图，每个task维护其前驱task集合。 有向图使用前驱进行表示，便于查找任务的前驱任务是否完成。
    tasks[0].pre = []
    tasks[1].pre = [0]
    tasks[2].pre = [1]
    tasks[3].pre = []
    tasks[4].pre = [3]
    tasks[5].pre = [2, 4]
    tasks[6].pre = [5]
    tasks[7].pre = [6]
    tasks[8].pre = [7]
    tasks[9].pre = [7]
    tasks[10].pre = [9]
    tasks[11].pre = []
    tasks[12].pre = [11]
    tasks[13].pre = [12]
    tasks[14].pre = []
    tasks[15].pre = [14]
    tasks[16].pre = [13, 15]
    tasks[17].pre = [16]
    tasks[18].pre = [17]
    tasks[19].pre = [18]
    tasks[20].pre = [18]
    tasks[21].pre = [20]

    tasks[0].next = [1]
    tasks[1].next = [2]
    tasks[2].next = [5]
    tasks[3].next = [4]
    tasks[4].next = [5]
    tasks[5].next = [6]
    tasks[6].next = [7]
    tasks[7].next = [8,9]
    tasks[8].next = []
    tasks[9].next = [10]
    tasks[10].next = []
    tasks[11].next = [12]
    tasks[12].next = [13]
    tasks[13].next = [16]
    tasks[14].next = [15]
    tasks[15].next = [16]
    tasks[16].next = [17]
    tasks[17].next = [18]
    tasks[18].next = [19,20]
    tasks[19].next = []
    tasks[20].next = [21]
    tasks[21].next = []
    
    pre_END = [8,10,19,21]  # pre_END 表示最终节点的前驱节点。
    machines = [MachineBase(m) for m in args.machine]
    task_graph = TaskGragh(tasks)
    # 每个task占用的设备资源，和释放的设备资源  这里每个任务
    # task的时间
    tasks[0].time = 5
    tasks[1].time = 10
    tasks[2].time = 7
    tasks[3].time = 3
    tasks[4].time = 10
    tasks[5].time = 10
    tasks[6].time = 20
    tasks[7].time = 5
    tasks[8].time = 5
    tasks[9].time = 5
    tasks[10].time = 5
    tasks[11].time = 5
    tasks[12].time = 10
    tasks[13].time = 7
    tasks[14].time = 3
    tasks[15].time = 10
    tasks[16].time = 10
    tasks[17].time = 20
    tasks[18].time = 5
    tasks[19].time = 5
    tasks[20].time = 5
    tasks[21].time = 5

    # task.dependency 的设置  dependency约束表示占用时要求在同一个设备，dependency2要求同一位置
    tasks[5].dependency = 2
    tasks[6].dependency = 5
    tasks[17].dependency = 16
    tasks[16].dependency = 13
    tasks[6].dependency2 = 6
    tasks[17].dependency2 = 17
    tasks[7].dependency2 = 2
    tasks[9].dependency2 = 5
    tasks[18].dependency2 = 13
    tasks[20].dependency2 = 16

    # 根据L1列表调度算法计算优先级
    # L1列表调度算法通过计算到终点节点的时间来作为优先级。
    # 这是一个图，从根节点向上搜索
    for node in pre_END:
        tasks = dfs(tasks,node)
    print("tasks' priority are shown as follows:")
    for id,task in enumerate(tasks):
        print(f"task {id} priority:", task.priority)
    # L1 scheduling to compute priority 优先级计算时考虑启发式规则，优先选择执行后prioqueue长度更长的
    if heuristics == 1:
        print("--------------------------启发式模块开启--------------------------")
    else :
        print("--------------------------未启用启发式模块--------------------------")
    print("\n")
    # 对资源的建模
    positions[0].machine = 0
    positions[1].machine = 1
    positions[2].machine = 2
    positions[3].machine = 2
    positions[4].machine = 3
    positions[5].machine = 3
    positions[6].machine = 4
    positions[7].machine = 4
    positions[8].machine = 5
    positions[9].machine = 6
    positions[10].machine = 2
    positions[11].machine = 3
    # 维护 q 中元素表示当前可以执行的任务 
    from queue import PriorityQueue
    q = PriorityQueue()
    # 初始化任务：判断当前哪些任务可以执行
    for task in tasks:
        flag = 1
        if task.pre is not None:
            for pre in task.pre:  # 判断是否所有前驱任务都完成
                if tasks[pre].status == 0:
                    flag = 0
        # 判断所需要的资源是否都满足
        flag2 = 0
        for pos in task.occupy:
            if(positions[pos-1].status==0):
                flag2 = 1
        if len(task.occupy)==0:
            flag2 = 1
        # 如果前驱任务完成了并且资源可满足，那么当前任务可执行 PUT INTO PRIORITY QUEUE
        if flag == 1 and flag2 == 1:    #   !!!
            q.put(task)
        

    i = 0 
    Finished = []
    SAVED_INFORMATION = []  # 用于存储每步选择过的task，当前这步选择的task，选择前的状态
    SAVED_CUR_TASK = [] # 存储的是当前步选择的设备
    SAVED_CUR_TASK_STATUS = []
    SAVED_CUR_RESOURCE_POSITIONS = []
    SAVED_CUR_RESOURCE_MACHINES = []
    SAVED_PRE_DECISIONS = [[] for _ in range(len(tasks))] # 用于记录之前在该步骤选择过的task
    step = 0
    while not q.empty():
        task = q.get() # EACH TIME GET THE PRIO TASK
        # task选择时不能选择之前尝试过的，如果所有的都尝试过了，那么就得再回溯一步
        SIG_CALL_BACK = 0
        while(task in SAVED_PRE_DECISIONS[step]):
            task = q.get()
            if q.size()==0 and (task in SAVED_PRE_DECISIONS[step]):
                SIG_CALL_BACK = 1
        
        if SIG_CALL_BACK ==1:
            SAVED_PRE_DECISIONS[step] = []
            if step == 0:
                print("PLEASE CHECK INPUT NO RESULT AVAILABLE")
                exit()
            step -= 1
            tasks = SAVED_CUR_TASK_STATUS[step]
            positions = SAVED_CUR_RESOURCE_POSITIONS[step]
            machines = SAVED_CUR_RESOURCE_MACHINES[step]
            SAVED_CUR_TASK.pop()
            continue
        # SAVE CURRENT STATUS FOR FURTHER CALL BACK
        # 需要保存的内容有：任务的状态、Tasks的status  2. 资源的状态 。 3.每一步的选择
        SAVED_CUR_TASK.append(task)
        SAVED_CUR_TASK_STATUS.append(tasks)
        SAVED_CUR_RESOURCE_POSITIONS.append(positions)
        SAVED_CUR_RESOURCE_MACHINES.append(machines)
        SAVED_PRE_DECISIONS[step].append(task)
        step += 1

        # remain to be done
    
        Finished.append(task) # TASK FINISH
        
       
        # 如果当前task没有设置依赖约束，则从所有可用设备中选择第一个可用设备占用
        if task.dependency == -1:
            for pos in task.occupy:
                if positions[pos-1].status == 0:
                    positions[pos-1].status = 1
                    task.position = positions[pos-1].id
                    task.machine = positions[pos-1].machine
                    break
        # 若当前task 选择了依赖约束：和其他task在相同的machine上执行   处理某些任务必须在相同设备上执行的约束。
        else:
            for pos in task.occupy:
                if positions[pos-1].status == 0 and positions[pos-1].machine==tasks[task.dependency].machine:
                    positions[pos-1].status = 1
                    task.position = positions[pos-1].id
                    task.machine = positions[pos-1].machine
                    break
            if len(task.occupy)== 0:
                task.machine = tasks[task.dependency].machine
        # 当前任务释放设备
        if len(task.release)>1 and task.dependency2!=-1:
            pos = tasks[task.dependency2].position
            positions[pos-1].status = 0
            task.release_position.append(pos)
            task.release_machine.append(positions[pos-1].machine)
        else :
            for pos in task.release:    
                positions[pos-1].status = 0
                task.release_position.append(pos)
                task.release_machine.append(positions[pos-1].machine)
        task.status = 1
        print("task:",task.id,"task_machine",task.machine)
        # 更新队列
        i = 0
        while i < q.qsize():  # 输出一下当前优先队列里的内容，用于调试   先将优先队列内容清空，看所有任务当前是否能执行，后续可以对这里进行一定的优化工作。
            print("task:",task.id,q.get().id)
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
                    if positions[pos-1].status == 0 :
                        flag2 = 1
            else:
                for pos in task.occupy:
                    if positions[pos-1].status == 0 and positions[pos-1].machine==tasks[task.dependency].machine:
                        flag2 = 1
            # 如果前驱任务完成了并且资源可满足，那么当前任务可执行
            # if task.id == 1:
            #     print(flag,flag2)
            #     print(tasks[task.pre[0]].status)
            if len(task.occupy)==0:
                flag2 = 1
            if flag == 1 and flag2 == 1 and task.status == 0:
                q.put(task)
                print("--",task.id)
                if heuristics == 1:   # 启发式规则计算第二优先级
                    tmp_Task = tasks   # 暂存状态，假设执行一步，计算下一步可执行的任务数量作为优先级
                    tmp_position = positions
                    tmp_machine = machines
                    tmp_cnt = 0
                    #-----------------simulate exe task-------------------------------------------------------------------------------------------------------------
                                # 如果当前task没有设置依赖约束，则从所有可用设备中选择第一个可用设备占用
                    if task.dependency == -1:
                        for pos in task.occupy:
                            if positions[pos-1].status == 0:
                                positions[pos-1].status = 1
                                task.position = positions[pos-1].id
                                task.machine = positions[pos-1].machine
                                break
                    # 若当前task 选择了依赖约束：和其他task在相同的machine上执行   处理某些任务必须在相同设备上执行的约束。
                    else:
                        for pos in task.occupy:
                            if positions[pos-1].status == 0 and positions[pos-1].machine==tasks[task.dependency].machine:
                                positions[pos-1].status = 1
                                task.position = positions[pos-1].id
                                task.machine = positions[pos-1].machine
                                break
                        if len(task.occupy)== 0:
                            task.machine = tasks[task.dependency].machine
                    # 当前任务释放设备
                    if len(task.release)>1 and task.dependency2!=-1:
                        pos = tasks[task.dependency2].position
                        positions[pos-1].status = 0
                        task.release_position.append(pos)
                        task.release_machine.append(positions[pos-1].machine)
                    else :
                        for pos in task.release:    
                            positions[pos-1].status = 0
                            task.release_position.append(pos)
                            task.release_machine.append(positions[pos-1].machine)
                    task.status = 1
                    while i < q.qsize():  # 输出一下当前优先队列里的内容，用于调试   先将优先队列内容清空，看所有任务当前是否能执行，后续可以对这里进行一定的优化工作。
                        print("task:",task.id,q.get().id)
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
                                if positions[pos-1].status == 0 :
                                    flag2 = 1
                        else:
                            for pos in task.occupy:
                                if positions[pos-1].status == 0 and positions[pos-1].machine==tasks[task.dependency].machine:
                                    flag2 = 1
                        # 如果前驱任务完成了并且资源可满足，那么当前任务可执行
                        # if task.id == 1:
                        #     print(flag,flag2)
                        #     print(tasks[task.pre[0]].status)
                        if len(task.occupy)==0:
                            flag2 = 1
                        if flag == 1 and flag2 == 1 and task.status == 0:
                            tmp_cnt +=1
                    tasks = tmp_Task
                    positions = tmp_position
                    machines = tmp_machine
                    task.heuristic = tmp_cnt
                    #-------------------------------------------------------------------------------------------------------------------------------
        if q.qsize() == 0 and len(Finished)!= len(tasks):
            print("!!!EORROR:qsize=0,and Finished not full!")
            pass


    if len(Finished)!= len(tasks):
        print("exsiting deadlock !")
    else :
        print("successfully compute solution !")
    print(len(Finished))
    
    print("step=",step)
    tasks_status = SAVED_CUR_TASK_STATUS[step-1]
    task_id = []
    for task in tasks:
        print(task.id,":",task.machine,task.position)
    print("\n")
    # for pos in positions:
    #     print(pos.machine)
    # 当前知道了task分配资源的顺序，知道每个资源分配的设备，知道task时间，下面生成一个甘特图。
    print("task分配资源的顺序为:")
    task_id = []
    for task in SAVED_CUR_TASK:
        task_id.append(task.id)
    print(task_id)
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
              'rgb(150, 47, 50)',
              'rgb(157, 47, 135)',
              'rgb(146, 40, 50)',
              'rgb(50, 144, 50)',
              'rgb(150, 147, 150)',
              'rgb(158, 50, 180)',
              'rgb(50, 147, 50)')
    Tm = [0 for _ in positions]  # Tm表示每个资源的可用时间
    PositionStatus = [0 for _ in positions]  # 表示资源当前被哪个任务占用
    gant = [[] for _ in Tm]
    for task in task_id:
        # 每个task 会占用一个设备，然后释放一个设备
        pos_id = tasks[task].position  # task占用的设备
        release_id = tasks[task].release_position   # task释放的设备
        print("taskID:",task,"occupy pos:",tasks[task].position,"release pos:",tasks[task].release_position)
        begin_time = Tm[tasks[task].position-1]
        for pre in tasks[task].pre:
            begin_time = max(begin_time, tasks[pre].available)
        tasks[task].available = begin_time+tasks[task].time   # available 记录设备完成工作时间
        tasks[task].start_time = begin_time  # start_time 记录开始占用该设备的时间
        PositionStatus[pos_id-1] = tasks[task].id
        # 任务执行顺序已知，每个任务占用一个资源，释放若干资源 ，需要知道占用的开始时间、结束时间、占用的资源是什么
        Tm[tasks[task].position-1] = begin_time  # Tm表示设备什么时候开始可用
        if tasks[task].next == []:  # 如果当前是最后一个节点，当前任务gant图计算
            for release_position in tasks[task].release_position:
                gant[release_position-1].append([Tm[release_position-1], begin_time, str(PositionStatus[release_position-1])])
                Tm[release_position-1] = begin_time
                print("S:", Tm[release_position-1], "E:", begin_time, "ID:",str(PositionStatus[release_position-1]))
            gant[pos_id-1].append([begin_time, begin_time+tasks[task].time, str(tasks[task].id)])
            print("S:", begin_time, "E:", begin_time+tasks[task].time, "ID:", str(tasks[task].id))
            Tm[pos_id-1] = begin_time+tasks[task].time
        else:
            for release_position in tasks[task].release_position:
                if release_position == pos_id:
                    gant[release_position-1].append([begin_time, begin_time+tasks[task].time, str(tasks[task].id)])
                    Tm[release_position-1] = begin_time+tasks[task].time
                else:
                    gant[release_position-1].append([Tm[release_position-1], begin_time, str(PositionStatus[release_position-1])])
                    print("S:", Tm[release_position-1], "E:", begin_time, "ID:", str(PositionStatus[release_position-1]))
                    Tm[release_position-1] = begin_time
        
    for gan in gant:
        print(gan)
        # print("Tm:",Tm)
    # exit()
    print("最优时间：")
    print(max(Tm))
    pyplt = py.offline.plot
    df = create_draw_defination(gant)
    fig = ff.create_gantt(df, colors=colors, index_col='Resource', show_colorbar=True,
                          group_tasks=True)
    # pyplt(fig, filename='tmp1.html')
    fig.show()

# 算法整体逻辑：
# Generate Task Graph   任务图的构建，每个任务绑定对应的设备。维护设备队列。从初始数据结构中选择有用的属性信息，进行任务图的构建，每个任务表示在板位上的操作
# List Schedule Generate Priority  得到任务图、以及每个任务的执行时间，根据时间以及任务图计算优先级。
# Simulate According To Priority.  维护优先级队列，每次从任务中取优先级最高的任务进行调度，若出现队列为空情况则进行回溯。
