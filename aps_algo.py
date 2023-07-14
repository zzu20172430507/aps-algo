import time
import my_modules
import threading
import datetime
from flask import Flask, request, jsonify
TEST_MODE = 1
app = Flask(__name__)
gantt_chart = None
global t_idx_dic, p_idx_dic, m_idx_dic, tasks, step, board_num, base_num, positions, machines, plateprocesses, SAVED_CUR_TASK, assayssid


def scheduler(data):
    global SAVED_CUR_TASK, gantt_chart, step
    now = datetime.datetime.now()
    # delta = datetime.timedelta(days=0, hours=0, minutes=5)  # 设置要添加的时间段
    # now = now + delta
    PROGRAM_START_TIME = time.time()
    # Data Read
    tasks, positions, machines, plateprocesses, t_idx_dic, p_idx_dic, m_idx_dic, base_num, board_num, window_size = my_modules.data_read(data)
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
    tasks, positions, i, Finished, SAVED_PRE_DECISIONS, SAVED_PRIOR_QUEUE, \
        q = my_modules.Initialize(tasks, 0, window_size * base_num, positions, t_idx_dic, p_idx_dic)
    PROGRAM_INIT_TIME = time.time()

    # 2. Running
    tasks, positions, machines = \
        my_modules.Run(tasks, window_size * base_num, positions, machines, q, SAVED_PRE_DECISIONS,
                    SAVED_PRIOR_QUEUE, t_idx_dic, p_idx_dic, base_num)
    PROGRAM_RUN_TIME = time.time()

    # 3. Results Output
    my_modules.ResultsOutput(tasks, board_num, base_num, positions, t_idx_dic, p_idx_dic, m_idx_dic, plateprocesses)
    for task in tasks:
        if task.task_name == "Open Lid4":
            print("!!!!!!!!!")
            print(task.task_name, task.start_time, " ", task.available, task.time)
            for pre in task.pre:
                ptask = tasks[t_idx_dic[pre]]
                print(ptask.task_name, ptask.start_time, " ", ptask.available, ptask.time, ptask.idx)
                for ppre in ptask.pre:
                    pptask = tasks[t_idx_dic[ppre]]
                    print(pptask.task_name, pptask.start_time, " ", pptask.available, pptask.time, pptask.idx)
    if TEST_MODE == 1:
        for idx, task in enumerate(tasks):
            print("ID:", idx, "S:", task.start_time, "E:", task.available)
    
    # 4.Generating Output
    PROGRAM_OUT_TIME = time.time()
    gantt_chart = my_modules.GenerateGant(plateprocesses, board_num, tasks, base_num, t_idx_dic, machines, now)
    PROGRAM_END_TIME = time.time()

    if TEST_MODE == 1:
        print("Total:", PROGRAM_END_TIME - PROGRAM_START_TIME, "s")
        print("读数据:", PROGRAM_READ_TIME - PROGRAM_START_TIME, "s")
        print("L1算法:", PROGRAM_PRIOR_TIME - PROGRAM_READ_TIME, "s")
        print("初始化:", PROGRAM_INIT_TIME - PROGRAM_PRIOR_TIME, "s")
        print("运行:", PROGRAM_RUN_TIME - PROGRAM_INIT_TIME, "s")
        print("计算甘特图:", PROGRAM_OUT_TIME - PROGRAM_RUN_TIME, "s")
        print("生成文件:", PROGRAM_END_TIME - PROGRAM_OUT_TIME, "s")


@app.route('/dispatch', methods=['POST'])
def handle_dispatch_requeset():
    data = request.get_json()
    result = {"message": "调度请求已收到, 任务启动成功"}
    thread = threading.Thread(target=scheduler, args=(data,))
    thread.start()
    return jsonify(result)


@app.route('/file', methods=['GET'])
def handle_file_request():
    if gantt_chart is not None:
        return jsonify(gantt_chart)
    else:
        result = {"message": "任务尚未调度完成"}
        return jsonify(result)


@app.route('/get_intermediate_result', methods=['GET'])
def get_intermediate_result():
    res = my_modules.getTmpGant()
    return jsonify(res)


if __name__ == "__main__":
    # ip = input("Input host IP  如 192.168.0.192 :")
    ip = '10.192.253.121'
    # ip = '192.168.214.11'
    ip = '192.168.200.103'
    ip = '30.137.80.111'
    ip = '10.192.204.237'
    app.run(host=ip, port=5000)
