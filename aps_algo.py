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

#  this is main

if __name__ == "__main__":
    PROGRAM_START_TIME = time.time()
    # Data Read
    tasks, positions, machines, plateprocesses, t_idx_dic, p_idx_dic, m_idx_dic, assayssid, base_num, board_num, heuristics = my_modules.data_read()

    # L1 Scheduling
    # tasks = my_modules.list_scheduling(tasks, t_idx_dic)
    tasks = my_modules.list_scheduling2(tasks, t_idx_dic)

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
    PROGRAM_STEP1_TIME = time.time()
    tasks, positions, i, Finished, SAVED_INFORMATION, SAVED_CUR_TASK, \
        SAVED_CUR_TASK_STATUS, SAVED_CUR_RESOURCE_POSITIONS, \
        SAVED_CUR_RESOURCE_MACHINES, SAVED_PRE_DECISIONS, SAVED_PRIOR_QUEUE, \
        TASK_SELECT, DEAD, step, q, pq_tmp = my_modules.Initialize(tasks, positions, t_idx_dic, p_idx_dic)

    # 2. Running
    PROGRAM_STEP2_TIME = time.time()
    tasks, positions, machines, step, SAVED_CUR_TASK_STATUS, TASK_SELECT = \
        my_modules.Run(tasks, positions, machines, q, heapq, SAVED_PRE_DECISIONS,
                       step, SAVED_CUR_TASK_STATUS, SAVED_CUR_RESOURCE_POSITIONS,
                       SAVED_CUR_RESOURCE_MACHINES, SAVED_PRIOR_QUEUE, t_idx_dic, p_idx_dic, DEAD, heuristics, TASK_SELECT)

    # 3. Results Output
    my_modules.ResultsOutput(tasks, step, SAVED_CUR_TASK_STATUS, TASK_SELECT, board_num, base_num, positions, t_idx_dic, p_idx_dic)

    # 4.Generating Output
    PROGRAM_STEP3_TIME = time.time()
    my_modules.GenerateGant(plateprocesses, board_num, tasks, base_num, t_idx_dic, assayssid, machines)
    PROGRAM_END_TIME = time.time()
    print("BEGIN:", PROGRAM_START_TIME, " STEP1:", PROGRAM_STEP1_TIME, "STEP2:", PROGRAM_STEP2_TIME, " STEP3:", PROGRAM_STEP3_TIME, "END:", PROGRAM_END_TIME)
    print("Total:", PROGRAM_END_TIME - PROGRAM_START_TIME, "s")
