def gen_task_list():
    return [1] * 5 + [2] * 5 + [7] * 4 + [10] * 5 + [13] * 5
    # return [13]

def params_list():
    return [
        {"data_id": 0},  {"data_id": 1}, {"data_id": 2}, {"data_id": 3}, {"data_id": 4},
        {"scale": 1}, {"scale": 2}, {"scale": 4}, {"scale": 8}, {"scale": 16},
        {"a": 5}, {"a": 10}, {"a": 20}, {"a": 40},
        {"a": 2}, {"a": 4}, {"a": 6}, {"a": 8}, {"a": 10},
        {"a": 2}, {"a": 4}, {"a": 6}, {"a": 8}, {"a": 10}
    ]