def GenerateGant(plateprocesses, board_num, tasks, base_num, t_idx_dic, assayssid, machines):
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
    with open('D:\\apache-tomcat-10.1.9\\webapps\\gantchart\\GantChart.json', 'w', encoding='utf-8') as f:
        json.dump(GantChartParent, f, indent=2, ensure_ascii=False)
    return GantChartParent