import pandas as pd
import json
import sys
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
        self.release_position = [] # 表示当前task释放的position
        self.dependency = -1  # dependency 表示当前task和哪个task需要有相同的设备
        self.available = 0
        self.start_time = 0
        self.heuristic = 0
        self.type = 0  # type为0时表示该任务为执行任务，type为1表示该任务为转移任务
        self.name = ""
    def __lt__(self, other):
        """定义<比较操作符"""
        if self.priority == other.priority:
            return self.heuristic > other.heuristic
        return self.priority > other.priority

with open("./projectInfo.txt","r",encoding = "utf-8") as f:
    data = json.load(f)

#  资源建模，设备资源：设备数量，每个设备对应的position，总共的position
instruments = data['instruments']
machine_num = len(instruments)
machines = [i for i in range(machine_num)]
print("设备数量为:", machine_num)
print("machines:", machines)
print("设备名称和id如下")
for instrument in instruments:
    print(instrument['instrumentCategoryText'], instrument['name'], instrument['id'])
print('--------------------------------------------------------------')
positions = []
positionsToMachine = {}
arm_id = None
for instrument_id,instrument in enumerate(instruments):   # 记录机械臂id，板位资源建模，以及对应设备
    if instrument['instrumentCategoryText'] == "机械臂":
        arm_id = instrument['id']
    if instrument['instrumentCategory']  in [1, 2]:
        pos_source = "pos"+instrument['id']
        work_source = "work"+instrument['id']
        positions.append(pos_source) # 首先如果是存储类型设备则板位设为 1
        positions.append(work_source) # work资源为另一个position
        positionsToMachine[pos_source] = instrument['id']
        positionsToMachine[work_source] = instrument['id']
    else :
        for _ in instrument['positions']:
            pos_source = "pos" + _['id'] 
            positions.append(pos_source)
            positionsToMachine[pos_source] = instrument['id']
        work_source = "work"+instrument['id']
        positions.append(work_source)
        positionsToMachine[work_source] = instrument['id']

print("机械臂的id为:", arm_id)
print("len pos:",len(positions))
# print("positions如下:", positions)
print("positions对应的 machine id 如下:",positionsToMachine)
# #########资源建模完成-----------------------------------------------------------------------------
#   以下是任务的建模
plate_processes = data['assayProcesses'][0]['plateProcesses']
assay_num = len(plate_processes)
print("泳道数量：",assay_num)
# 前两个泳道是开始执行前和执行结束后泳道，不需要处理
for process in plate_processes:
    if len(process['nodes'])==0:
        continue
    cur_nodes = process['nodes']
    for node in cur_nodes:
        print(node['name'], "node-processTypeText", node['processTypeText'])
    print('\n')


tasks = []
# 记录
for process in plate_processes:     # 遍历所有泳道
    plate_num = process['plate_num']  #  plate_num表示当前泳道的板子数量
    cur_nodes = process['nodes']    # cur_nodes 表示泳道中的节点
    for node_idx, node in enumerate(cur_nodes):   # 遍历泳道中的节点
        # 处理机械臂转移任务
        task_id = len(tasks)                # task_id表示tasks中的任务数量
        if node_idx-1 >= 0:                    # 如果当前node不是泳道中第一个节点
            pre_node = cur_nodes[node_idx-1]    # 则查找其上一个节点
            # 由pre_node转移到当前node的任务
            trans_task = Task(pre_node['id']+node['id'],[])  # 创建一个机械臂转移任务，由上一个节点转移到该节点
            trans_task.type = 1 # 表示该任务为转移任务
            trans_task.time = pre_node['robotGetTime'] + node['robotPutTime']  # 当前任务时间为从上一个取的时间加上放到当前节点的时间
            trans_task.pre = tasks[task_id-1].id  # 记录转移任务的前驱任务
            trans_task.name = pre_node['id'][0:4] + " transfer to " + node['id'][0:4]
            tasks[task_id-1].next = trans_task.id  # 记录一下上一个任务的后续任务为转移任务。
            tasks.append(trans_task) # 将转移任务加入任务列表中。

        # 每个节点初始化一个task: cur_task ,  另外还有一个transfer_task, 表示板子在设备之间的转移需要
        node_id = node['id']
        cur_task = Task(node_id,[])
        cur_task.type = 0
        cur_task.name = node_id[0:4]
        if node_idx - 1 >= 0:   # 如果存在前驱任务，则其上一个任务是转移任务，转移任务的后续，该任务的前驱进行设置
            cur_task.pre = trans_task.id
            trans_task.next = cur_task.id
        if node['processType'] == 2:  # 如果是个抓板工艺，那么需要机械臂协助工作，则需要占用的设备有机械臂和该工艺占用的设备
            cur_task.time = node['estDuration']
            cur_task.occupy = []
            cur_task.occupy.append([node['projectInstrumentId']])  # 将当前任务占用设备记录
            cur_task.occupy.append([arm_id])
        elif node['processType'] == 6:  # 如果是无板工艺，其中设备池属于无板工艺进行特判
            cur_task.time = node['estDuration']
            if node['projectInstrumentName'] == "Instrument Pool": # 如果是设备池工艺类型，则当前任务可由多个备选occ
                poolNodes = node['poolNodes']
                process_type = poolNodes[0]['processType']
                if process_type == 2: 
                    cur_task.occupy = []
                    pnode_occ = []
                    for pnode in poolNodes:
                        pnode_occ.append(pnode['projectInstrumentId'])
                    cur_task.occupy.append(pnode_occ)
                    cur_task.occupy.append(arm_id)
                elif process_type == 3:
                    print("在设备池中会出现放板工艺吗？如果会作者没有考虑，这里的逻辑没有做处理")
                    pass
                elif process_type == 1: # 如果是多板工艺
                    pass
                else:
                    cur_task.occupy.append()
        elif node['processType'] == 3: # 放板工艺
            cur_task.time = node['estDuration']
            cur_task.occupy = []    # 放板工艺释放设备，无占用设备，单纯是由冰箱回收的操作。
            cur_task.release = []
            cur_task.release.append([node['projectInstrumentId']])  # 将当前任务占用设备记录
            
        else:  # 如果是其他工艺，则
            pass
        tasks.append(cur_task)

# 实现一个功能，将id对应成在数组中的下标
map_taskId_to_Idx = {}
for idx, task in enumerate(tasks):
    map_taskId_to_Idx[task.id] = idx
# for task in tasks:
#     print(map_taskId_to_Idx[task.id])
print('\n')

print("任务数量为:", len(tasks))
for task in tasks:
    print(map_taskId_to_Idx[task.id], " name:", task.name," type", task.type, " pre:", task.pre, "next:", task.next)
    print("TASK OCCUPY:",task.occupy)
    print("--------------------------------------------------------------------------")
