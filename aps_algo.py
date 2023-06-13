import time
import sys
sys.path.append("./src")
import my_modules
from my_modules import *
from flask import Flask, request, jsonify
TEST_MODE = 1
app = Flask(__name__)
gantt_chart = None


def scheduler(data):
    PROGRAM_START_TIME = time.time()
    # Data Read
    tasks, positions, machines, plateprocesses, t_idx_dic, p_idx_dic, m_idx_dic, assayssid, base_num, board_num, window_size = my_modules.data_read(data)
    PROGRAM_READ_TIME = time.time()
    # L1 Scheduling
    tasks = my_modules.list_scheduling2(tasks, t_idx_dic)
    PROGRAM_PRIOR_TIME = time.time()
    # L1 Result Output & Task Graph & Resource Outp t-- TEST CODE
    if TEST_MODE == 1:
        for _id, task in enumerate(tasks):
            print(f"task {_id} priority:", task.priority)
        for idx, task in enumerate(tasks):
            print("task Idx:", idx, " ID: ", task.id, " Name: ", task.task_name)
        for idx, pos in enumerate(positions):
            print("pos Idx:", idx, " ID: ", pos.id, " Name: ", pos.name)
        # for idx, task in enumerate(tasks):
        #     print("\n")
        #     print("task Idx:", idx, " task name: ", task.task_name)
        #     print(" OCC: ")
        #     for _ in task.occupy:
        #         for __ in _:
        #             print(positions[p_idx_dic[__]].name)
        #     print(" REL: ")
        #     for _ in task.release:
        #         for __ in _:
        #             print(positions[p_idx_dic[__]].name)
    # 1. Initialize
    tasks, positions, i, Finished, SAVED_CUR_TASK, SAVED_PRE_DECISIONS, SAVED_PRIOR_QUEUE, \
        step, q = my_modules.Initialize(tasks, 0, window_size * base_num, positions, t_idx_dic, p_idx_dic)
    PROGRAM_INIT_TIME = time.time()
    # 2. Running
    tasks, positions, machines, step, SAVED_CUR_TASK = \
        my_modules.Run(tasks, window_size * base_num, positions, machines, q, SAVED_PRE_DECISIONS,
                       step, SAVED_PRIOR_QUEUE, t_idx_dic, p_idx_dic, SAVED_CUR_TASK, base_num)
    PROGRAM_RUN_TIME = time.time()
    
    # 3. Results Output
    my_modules.ResultsOutput(tasks, step, SAVED_CUR_TASK, board_num, base_num, positions, machines, t_idx_dic, p_idx_dic, m_idx_dic, plateprocesses)
    for task in tasks:
        if task.task_name == "Open Lid3":
            print("!!!!!!!!!")
            print(task.task_name, task.start_time, " ", task.available, task.time)
            for pre in task.pre:
                ptask = tasks[t_idx_dic[pre]]
                print(ptask.task_name, ptask.start_time, " ", ptask.available, ptask.time)
                for ppre in ptask.pre:
                    pptask = tasks[t_idx_dic[ppre]]
                    print(pptask.task_name, pptask.start_time, " ", pptask.available, pptask.time)
    if TEST_MODE == 1:
        for idx, task in enumerate(tasks):
            print("ID:", idx, "S:", task.start_time, "E:", task.available)

    # 4.Generating Outputo
    PROGRAM_OUT_TIME = time.time()
    gantchart = my_modules.GenerateGant(plateprocesses, board_num, tasks, base_num, t_idx_dic, assayssid, machines)
    PROGRAM_END_TIME = time.time()

    if TEST_MODE == 1:
        print("Total:", PROGRAM_END_TIME - PROGRAM_START_TIME, "s")
        print("读数据:", PROGRAM_READ_TIME - PROGRAM_START_TIME, "s")
        print("L1算法:", PROGRAM_PRIOR_TIME - PROGRAM_READ_TIME, "s")
        print("初始化:", PROGRAM_INIT_TIME - PROGRAM_PRIOR_TIME, "s")
        print("运行:", PROGRAM_RUN_TIME - PROGRAM_INIT_TIME, "s")
        print("计算甘特图:", PROGRAM_OUT_TIME - PROGRAM_RUN_TIME, "s")
        print("生成文件:", PROGRAM_END_TIME - PROGRAM_OUT_TIME, "s")
    return gantchart


@app.route('/dispatch', methods=['POST'])
def handle_dispatch_requeset():
    data = request.get_json()
    result = {"message": "调度请求已收到", "data": data}
    global gantt_chart
    gantt_chart = scheduler(data)
    return jsonify(result)


@app.route('/file', methods=['GET'])
def handle_file_request():
    return jsonify(gantt_chart)


if __name__ == "__main__":
    app.run(host='192.168.0.192', port=5000)
