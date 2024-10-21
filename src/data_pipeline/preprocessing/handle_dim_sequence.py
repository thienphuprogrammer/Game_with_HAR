import numpy as np


def pad_length(ele, max_dim, metric_temp: list = None) -> np.array:
    if metric_temp is None:
        metric_temp = []
    if ele.shape[0] == max_dim:
        return ele

    step = max(1, ele.shape[0] // (max_dim - ele.shape[0]))

    for j in range(0, ele.shape[0]):
        metric_temp.append(ele[j])
        if len(metric_temp) == max_dim:
            break
        if j % step == 0:
            metric_temp.append(ele[j])
        if len(metric_temp) == max_dim:
            break
    return pad_length(np.array(metric_temp), max_dim)


def truncate_length(ele, max_dim, start=0) -> np.array:
    if ele.shape[0] == max_dim:
        return ele

    metrics_temp = []
    step = max(2, ele.shape[0] // (ele.shape[0] - max_dim))
    j = start
    while len(metrics_temp) != max_dim:
        metrics_temp.append(ele[j])
        j = (j + step) % ele.shape[0]
    return np.array(metrics_temp)


def pad_and_truncate(ele, max_dim: int = 35) -> np.array:
    list_elements = []
    current_length = ele.shape[0]

    if current_length == max_dim:
        return [ele]

    dev = max(int(round(current_length / max_dim)), 1)
    if dev * max_dim > current_length:
        new_element = pad_length(ele, max_dim * dev)

        if new_element.shape[0] > max_dim:
            for step in range(dev):
                list_elements.append(truncate_length(new_element, max_dim, step))
        else:
            list_elements.append(new_element)
    else:
        for step in range(dev):
            list_elements.append(truncate_length(ele, max_dim, step))
    return list_elements
