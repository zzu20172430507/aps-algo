import time
import sys
import heapq
# import plotly as py
# import plotly.figure_factory as ff
# from pprint import pprint
sys.path.append("./src")
import my_modules
from my_modules import generate_colors
from my_modules import *
TEST_MODE = 1

#  21：46

if __name__ == "__main__":
    PROGRAM_START_TIME = time.time()
    # Data Read
    tasks, positions, machines, plateprocesses, t_idx_dic, p_idx_dic, m_idx_dic, assayssid, base_num, board_num, heuristics, window_size = my_modules.data_read()
    PROGRAM_READ_TIME = time.time()
    # L1 Scheduling
    # tasks = my_modules.list_scheduling(tasks, t_idx_dic)
    tasks = my_modules.list_scheduling2(tasks, t_idx_dic)
    PROGRAM_PRIOR_TIME = time.time()
    # L1 Result Output & Task Graph & Resource Output-- TEST CODE
    if TEST_MODE == 1:
        for _id, task in enumerate(tasks):
            print(f"task {_id} priority:", task.priority)
        for idx, task in enumerate(tasks):
            print("task Idx:", idx, " ID: ", task.id, " Name: ", task.task_name)
        for idx, pos in enumerate(positions):
            print("pos Idx:", idx, " ID: ", pos.id, " Name: ", pos.name)
    # Scheduling: 1. Initialize 2. Running  3. Results Output  4. Generating Output
    # 1. Initialize
    count = board_num // window_size + (0 if board_num % window_size == 0 else 1)
    print(count)
    tasks, positions, i, Finished, SAVED_INFORMATION, SAVED_CUR_TASK, \
        SAVED_CUR_TASK_STATUS, SAVED_CUR_RESOURCE_POSITIONS, \
        SAVED_CUR_RESOURCE_MACHINES, SAVED_PRE_DECISIONS, SAVED_PRIOR_QUEUE, \
        TASK_SELECT, DEAD, step, q, pq_tmp = my_modules.Initialize(tasks, 0, window_size * base_num, positions, t_idx_dic, p_idx_dic)
    PROGRAM_INIT_TIME = time.time()
    # 2. Running
    # PROGRAM_STEP2_TIME = time.time()
    tasks, positions, machines, step, SAVED_CUR_TASK_STATUS, TASK_SELECT, SAVED_CUR_TASK = \
        my_modules.Run(tasks, 0, window_size * base_num, positions, machines, q, heapq, SAVED_PRE_DECISIONS,
                       step, SAVED_CUR_TASK_STATUS, SAVED_CUR_RESOURCE_POSITIONS,
                       SAVED_CUR_RESOURCE_MACHINES, SAVED_PRIOR_QUEUE, t_idx_dic, p_idx_dic, DEAD, heuristics, TASK_SELECT, SAVED_CUR_TASK, base_num)
    PROGRAM_RUN_TIME = time.time()
    # 3. Results Output
    my_modules.ResultsOutput(tasks, step, SAVED_CUR_TASK_STATUS, TASK_SELECT, SAVED_CUR_TASK, board_num, base_num, positions, machines, t_idx_dic, p_idx_dic, m_idx_dic)
    for idx, task in enumerate(tasks):
        print("ID:", idx, "S:", task.start_time, "E:", task.available)

    # 4.Generating Output
    PROGRAM_OUT_TIME = time.time()
    my_modules.GenerateGant(plateprocesses, board_num, tasks, base_num, t_idx_dic, assayssid, machines)
    PROGRAM_END_TIME = time.time()
    # print("BEGIN:", PROGRAM_START_TIME, " STEP1:", PROGRAM_STEP1_TIME, "STEP2:", PROGRAM_STEP2_TIME, " STEP3:", PROGRAM_STEP3_TIME, "END:", PROGRAM_END_TIME)
    print("Total:", PROGRAM_END_TIME - PROGRAM_START_TIME, "s")
    print("读数据:", PROGRAM_READ_TIME - PROGRAM_START_TIME, "s")
    print("L1算法:", PROGRAM_PRIOR_TIME - PROGRAM_READ_TIME, "s")
    print("初始化:", PROGRAM_INIT_TIME - PROGRAM_PRIOR_TIME, "s")
    print("运行:", PROGRAM_RUN_TIME - PROGRAM_INIT_TIME, "s")
    print("计算甘特图:", PROGRAM_OUT_TIME - PROGRAM_RUN_TIME, "s")
    print("生成文件:", PROGRAM_END_TIME - PROGRAM_OUT_TIME, "s")
    
