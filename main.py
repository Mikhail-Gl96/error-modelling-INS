import os
import numpy as np
import matplotlib.pyplot as plt
import TrajectoryModel as Trj


# переменные
freq = 100             # Частота алгоритма                      в Гц
dt = 1 / freq          # шаг по времени (наш шаг выполнения)    в сек
hours = 2              # Часы работы                            в часах
end_time = int(hours * 3600 * freq)  # время работы равное 2 часам   в сек
time = range(end_time)          # массив из времени


model_flight = Trj.Trajectory(hours, freq)
# data = ['v_east', 'v_north', 'W', 'epsilon', 'alpha', 'a_east', 'a_north', 'a_center']
model_flight.make_trajectory()
# model_flight.get_graphics(data)
# input()
a_north_arr = model_flight.get_a_north_arr()
a_east_arr = model_flight.get_a_east_arr()
a_center_arr = model_flight.get_a_center_arr()
w_up_arr = model_flight.get_w_up_arr()

# print(f'a_n= {len(a_north_arr)}  a_e= {len(a_east_arr)}  a_center= {len(a_center_arr)}  w_up= {len(w_up_arr)}')

# константы
gravity = 9.81      # Земная гравитация               в м/с**2
Radius = 6371000    # Радиус Земли                    в м
U = 7.292115 * pow(10, -5)  # Скорость вращения Земли в рад/сек


# пределы значений
limit_Heading = np.deg2rad(0.1)         # Курс      в рад
limit_Roll = np.deg2rad(0.01)           # Крен      в рад
limit_Pitch = np.deg2rad(0.01)          # Тангаж    в рад

limit_Lat = np.deg2rad(1)         # Широта          в рад
limit_Lon = np.deg2rad(0)         # Долгота         в рад

limit_delta_V = 0.7         # ограничение по скорости   в м/с

limit_coord = 1            # к  м (погрешность определения координат)

# константы приборов
_acc_null_param = 0.8 * pow(10, -4)           # вводим в м/с**2
B_n = _acc_null_param         # Смещение нулей акса по  северному направлению       в м/с**2
B_e = _acc_null_param         # Смещение нулей акса по  восточному направлению      в м/с**2

_mu_null_param = 500 / 1000000  # 0.0005
# print(_mu_null_param)
mu_n = _mu_null_param        # масштабный коэф аксов по северному направлению       безразм
mu_e = _mu_null_param        # масштабный коэф аксов по восточному направлению      безразм

_w_dr_null_param = np.deg2rad(0.01 / 3600)            # в град/ч  (делим на 3600 для гр/с)
w_North_dr = _w_dr_null_param  # угловая скорость дрейфа по восточному направлению  в рад/с
w_East_dr = _w_dr_null_param   # угловая скорость дрейфа по северному направлению   в рад/с
W_Up_dr = _w_dr_null_param     # угловая скорость дрейфа по высотному каналу        в рад/с

# ошибкив рад
error_Heading = np.deg2rad(0)     # ошибка по Курсу                                 в рад
error_Roll = np.deg2rad(0)        # ошибка по Крену                                 в рад
error_Pitch = np.deg2rad(0)       # ошибка по Тангажу                               в рад
error_Lat = np.deg2rad(0)         # ошибка по широте                                в рад
error_V_e = 0                     # ошибка по скорости север                        в м/с
error__V_n = 0                    # ошибка по скорости восток                       в м/с
error_coord = 0                   # ошибка по координатам                           в м/ч


fi = np.deg2rad(55.75222)   # широта                            в рад
Heading = np.deg2rad(0)     # Курс                              в рад
Roll = np.deg2rad(0)        # Крен                              в рад
Pitch = np.deg2rad(0)       # Тангаж                            в рад


Acc_North = 0   # ускорение по северному направлению            в м/с**2
Acc_East = 0    # ускорение по восточному направлению           в м/с**2


dot_V_east = 0          # Значение до интегрирования            в м/с**2
dot_V_North = 0         # Значение до интегрирования            в м/с**2

d_V_North = 0   # дельта скорости по  северному направлению  (значение после интегрирования)    в м/с
d_V_East = 0    # дельта скорости по  восточному направлению (значение после интегрирования)    в м/с


dot_F_North = 0     # угол между навигационной СК и приборной СК до интегрирования      в рад/с**2
dot_F_East = 0      # угол между навигационной СК и приборной СК до интегрирования      в рад/с**2
dot_F_Up = 0        # угол между навигационной СК и приборной СК до интегрирования      в рад/с**2

_F_null_param = _acc_null_param / gravity
F_North = _F_null_param  # угол между навигационной СК и приборной СК (значение после интегрирования)   в рад/с
F_East = _F_null_param   # угол между навигационной СК и приборной СК (значение после интегрирования)   в рад/с
F_Up = w_East_dr / (U * np.cos(fi))  # угол между навигационной СК и приборной СК (значение после интегрирования)
                                                                                # (гр/ч -> рад/с)

# Значения параметров для начальных условий
_null_d_V_North = d_V_North
_null_d_V_East = d_V_East
_null_F_North = F_North
_null_F_East = F_East
_null_F_Up = F_Up
_null_Lat = fi
_null_coord = 0
# _null_ = 0


def integrate_equation(prev_value, right_part):
    new_value = prev_value + right_part * dt
    return new_value


def solve_Stationary_East_chanel_SystemEquations(i):
    """Решаем канал East"""
    global dot_V_east, dot_F_North, Acc_North, Acc_East
    Acc_North = a_north_arr[i]
    Acc_East = a_east_arr[i]
    dot_V_east = -gravity * F_North + Acc_North * F_Up + B_e + Acc_East * mu_e
    dot_F_North = d_V_East / Radius + w_North_dr


def solve_Stationary_North_chanel_SystemEquations(i):
    """Решаем канал North"""
    global dot_V_North, dot_F_East, Acc_North, Acc_East
    Acc_North = a_north_arr[i]
    Acc_East = a_east_arr[i]
    dot_V_North = gravity * F_East - Acc_East * F_Up + B_e + Acc_North * mu_n
    dot_F_East = -d_V_North / Radius + w_East_dr


def solve_Stationary_Up_chanel_SystemEquations(i):
    """Решаем канал UP"""
    global dot_F_Up
    W_Up = np.deg2rad(w_up_arr[i])
    delta_F_gyro = 0.0005        # Пока так будет &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    dot_F_Up = d_V_East / Radius * np.tan(fi) + W_Up_dr + W_Up * delta_F_gyro


def solve_before_integration_Stationary_SystemEquations(i):
    """Решение уравнений мат модели ошибок """
    solve_Stationary_East_chanel_SystemEquations(i)
    solve_Stationary_North_chanel_SystemEquations(i)
    solve_Stationary_Up_chanel_SystemEquations(i)


def integrate_Stationary_SystemEquations():
    """Интегрирование уравнений мат модели ошибок """
    global d_V_North, d_V_East, F_North, F_East, F_Up
    d_V_North = integrate_equation(d_V_North, dot_V_North)
    d_V_East = integrate_equation(d_V_East, dot_V_east)
    F_North = integrate_equation(F_North, dot_F_North)
    F_East = integrate_equation(F_East, dot_F_East)
    F_Up = integrate_equation(F_Up, dot_F_Up)


def solve_Stationary_SystemEquations(i):
    """Решаем мат модель и интегрируем"""
    solve_before_integration_Stationary_SystemEquations(i)   # РЕШЕНИЕ УР-ИЙ
    integrate_Stationary_SystemEquations()                   # ИНТЕГРИРОВАНИЕ


def errors_count():
    """Считаем ошибки вычислений """
    global error_Heading, error_Roll, error_Pitch, error_Lat, error_V_e, error__V_n, error_coord
    error_Heading = _null_F_Up - F_Up
    # print(np.rad2deg(_null_F_Up), np.rad2deg(F_Up))
    error_Roll = _null_F_East - F_East
    error_Pitch = _null_F_North - F_North
    error_Lat = _null_Lat - fi
    error_V_e = _null_d_V_North - d_V_East
    error__V_n = _null_d_V_East - d_V_North
    error_coord = _null_coord - cord_error_gloabal
    

def check_limits():
    """Проверяем нарушение допустимых границ значений"""
    errors_count()              # считаем текущую ошибку
    round_num = 3

    local_error_V_e = np.abs(error_V_e).__round__(1)
    local_error__V_n = np.abs(error__V_n).__round__(1)
    local_error_Heading = np.abs(error_Heading).__round__(3)
    local_error_Pitch = np.abs(error_Pitch).__round__(2)
    local_error_Roll = np.abs(error_Roll).__round__(2)
    local_error_Lat = np.abs(error_Lat).__round__(round_num)
    local_error_coord = np.abs(error_coord).__round__(round_num)
    answer = True
    # print(error_Heading.__round__(round_num), limit_Heading.__round__(round_num))
    if local_error_V_e > limit_delta_V:
        # print(f'Attention, delta v_e = {error_V_e.__round__(round_num)}  > {limit_delta_V.__round__(round_num)} limit')
        answer = False
    if local_error__V_n > limit_delta_V:
        # print(f'Attention, delta v_n = {error__V_n.__round__(round_num)}  > {limit_delta_V.__round__(round_num)} limit')
        answer = False
    if local_error_Heading > limit_Heading:
        # print(f'Attention, курс = {error_Heading.__round__(round_num)}  > {limit_Heading.__round__(round_num)} limit')
        answer = False
        # input()
    if local_error_Pitch > limit_Pitch:
        # print(f'Attention, крен = {error_Pitch.__round__(round_num)}  > {limit_Pitch.__round__(round_num)} limit')
        answer = False
    if local_error_Roll > limit_Roll:
        # print(f'Attention, тангаж = {error_Roll.__round__(round_num)}  > {limit_Roll.__round__(round_num)} limit')
        answer = False
    if local_error_Lat > limit_Lat:
        # print(f'Attention, широта = {error_Lat.__round__(round_num)}  > {limit_Lat.__round__(round_num)} limit ')
        answer = False
    if local_error_coord > limit_coord:
        # print(f'Attention, ошибка координат = {error_coord.__round__(round_num)}  > '
        #       f'{limit_coord.__round__(round_num)} limit ')
        answer = False
    return answer



d_V_North_arr   = []
d_V_East_arr    = []
F_North_arr     = []
F_East_arr      = []
F_Up_arr        = []

dot_V_east_arr = []
dot_F_North_arr = []
dot_V_North_arr = []
dot_F_East_arr = []
dot_F_Up_arr = []


def reset_to_start_values():
    """Оичстка массивов и возврат к значениям при н.у."""
    global fi, Heading, Roll, Pitch, Acc_North, Acc_East, dot_V_east, dot_V_North, d_V_North, d_V_East, \
        dot_F_North, dot_F_East, dot_F_Up, F_North, F_East, F_Up

    fi = np.deg2rad(55.75222)
    Heading = np.deg2rad(0)
    Roll = np.deg2rad(0)
    Pitch = np.deg2rad(0)
    Acc_North = 0
    Acc_East = 0
    dot_V_east = 0
    dot_V_North = 0
    d_V_North = 0
    d_V_East = 0
    dot_F_North = 0
    dot_F_East = 0
    dot_F_Up = 0

    F_North = np.deg2rad(0.005)
    F_East = np.deg2rad(0)
    F_Up = np.deg2rad(2) / 3600

    d_V_North_arr.clear()
    d_V_East_arr.clear()
    F_North_arr.clear()
    F_East_arr.clear()
    F_Up_arr.clear()
    dot_V_east_arr.clear()
    dot_F_North_arr.clear()
    dot_V_North_arr.clear()
    dot_F_East_arr.clear()
    dot_F_Up_arr.clear()


def write_data():
    """Запись рез-ов на каждом шагу"""
    d_V_North_arr.append(d_V_North)
    d_V_East_arr.append(d_V_East)
    F_North_arr.append(np.rad2deg(F_North))
    F_East_arr.append(np.rad2deg(F_East))
    F_Up_arr.append(np.rad2deg(F_Up))

    dot_V_east_arr.append(dot_V_east)
    dot_F_North_arr.append(dot_F_North)
    dot_V_North_arr.append(dot_V_North)
    dot_F_East_arr.append(dot_F_East)
    dot_F_Up_arr.append(dot_F_Up)


colors_graph = ['red', 'blue', 'violet', 'green', 'black', 'brown', 'yellow', 'orange', 'purple', 'pink',
                'red', 'blue', 'violet', 'green', 'black', 'brown', 'yellow', 'orange', 'purple', 'pink',
                'red', 'blue', 'violet', 'green', 'black', 'brown', 'yellow', 'orange', 'purple', 'pink']


def make_graph(title, data_time, data_DATA, colors, labels):
    """Функция создает на выходе график с кривыми
        на вход: название,
                 массив[время графика 1, время графика 2, ...],
                 массив[информация график 1, итнформация график 2, ...],
                 цвета графиков, названия кривых"""
    plt.cla()
    plt.title(title)
    data_number = len(data_DATA)
    for i in range(data_number):
        plt.plot(data_time[i], data_DATA[i], color=colors[i], label=labels[i])
    plt.legend()
    plt.grid()
    plt.xlabel("Time")
    plt.ylabel('Data')
    plt.show()


def make_graph_from_dict(args):
    """Делаем графики из данных в словаре dict_all"""
    for name in args:
        local_data = dict_all.get(f'{name}')
        make_graph(f'{local_data.get("name")}', local_data.get("time"), local_data.get("data"), colors_graph, local_data.get("labels"))


# def print_status_bar(num, multipl=0, counter=20):
def print_status_bar(num):
    multipl = 1
    counter = 1
    print(f'  {counter}%')
    while True:
        num = yield
        if num // ((end_time / freq) * multipl) == 1:
            counter += 1
            os.system('cls')
            print(f' {counter}%')
            multipl += 1


x_cord_errors_arr, y_cord_errors_arr, vector_coord_errors_arr = [], [], []
cord_error_gloabal = 0
x_cord_error, y_cord_error = 0, 0


def compute_coord_errors():
    global cord_error_gloabal, x_cord_error, y_cord_error
    x_cord_error += dot_V_east * dt
    y_cord_error += dot_V_North * dt
    cord_error_gloabal = np.sqrt(pow(x_cord_error, 2) + pow(y_cord_error, 2))
    vector_coord_errors_arr.append(cord_error_gloabal)


def check_1hour_errors(time, step=[1], stepp=[1]):
    round_num = 3
    if time == (3600 * freq * stepp[0] - 1):
        print(f'{"*" * 100}\nОшибки за {step[0]} час работы:\n\n')
        status_answer = check_limits()  # Проверка на выходы за границы пределов

        if status_answer is True:
            print(f'Все значения в пределах нормы: \n')
            print(f'delta v_e = {error_V_e.__round__(round_num)}  '
                                                f'< {limit_delta_V.__round__(round_num) * step[0]} limit')
            print(f'delta v_n = {error__V_n.__round__(round_num)}  '
                                                f'< {limit_delta_V.__round__(round_num) * step[0]} limit')
            print(f'курс = {np.rad2deg(error_Heading).__round__(round_num)}  '
                                                f'< {np.rad2deg(limit_Heading).__round__(round_num) * step[0]} limit')
            print(f'тангаж = {np.rad2deg(error_Pitch).__round__(round_num)}  '
                                                f'< {np.rad2deg(limit_Pitch).__round__(round_num) * step[0]} limit')
            print(f'крен = {np.rad2deg(error_Roll).__round__(round_num)}  '
                                                f'< {np.rad2deg(limit_Roll).__round__(round_num) * step[0]} limit')
            print(f'ошибка координат = {error_coord.__round__(round_num)}  '
                                                f'< {limit_coord.__round__(round_num) * step[0]} limit ')
        else:
            print(f'Attention, delta v_e = {error_V_e.__round__(round_num)}  '
                                                f'> {limit_delta_V.__round__(round_num) * step[0]} limit')
            print(f'Attention, delta v_n = {error__V_n.__round__(round_num)}  '
                                                f'> {limit_delta_V.__round__(round_num) * step[0]} limit')
            print(f'Attention, курс = {np.rad2deg(error_Heading).__round__(round_num)}  '
                                                f'> {np.rad2deg(limit_Heading).__round__(round_num) * step[0]} limit')
            print(f'Attention, тангаж = {np.rad2deg(error_Pitch).__round__(round_num)}  '
                                                f'> {np.rad2deg(limit_Pitch).__round__(round_num) * step[0]} limit')
            print(f'Attention, крен = {np.rad2deg(error_Roll).__round__(round_num)}  '
                                                f'> {np.rad2deg(limit_Roll).__round__(round_num) * step[0]} limit')
            print(f'Attention, ошибка координат = {error_coord.__round__(round_num)}  '
                                                f'> {limit_coord.__round__(round_num) * step[0]} limit ')

        print("*" * 100)
        stepp[0] += 1


def start_modelling():
    """Запускаем прогон нашей программы в соотв с заданными параметрами"""
    status_bar = print_status_bar(0)
    next(status_bar)
    for i in range(end_time):  # Проходимся по мат моделе, пишем результаты для текущего В
        solve_Stationary_SystemEquations(i)  # Решенеие стац мат модели и интегрирование
        check_limits()  # Проверка на выходы за границы пределов
        compute_coord_errors()
        check_1hour_errors(i)
        write_data()  # Запись результатов
        # print(i)
        status_bar.send(i)




#
# min_step_B = 0.01       # шаг B
# n_steps = 2             # кол-во ненулевых шагов
# data_arr = [B_n*i for i in range(n_steps + 1)]     # формируем массив из n_steps шагов + нулевой шаг
# data_arr = data_arr[1:]
# arr_data_dot_V_east = []        # храним данные для графиков


# def change_B(index):
#     """Сменяем значение начальной скорости ухода"""
#     global _acc_null_param, B_n, B_e
#     # _acc_null_param = np.deg2rad(0.05)
#     _acc_null_param = data_arr[index]   # обновляем в соотв с массивом заданных значений В
#     B_n = _acc_null_param       # обновляем данные в соотв с _acc_null_param
#     B_e = _acc_null_param       # обновляем данные в соотв с _acc_null_param


start_modelling()

# print(f' d_v_n= {len(d_V_North_arr)}  len time= {len(time)}  time= {time[0:10], time[-10:-1]}')
# print(f'data = {d_V_North_arr[0:10]}')

time = [(i / (freq * 60)) for i in time]
# print(f' \n'
      # f'\nd_v_n= {len(d_V_North_arr)}  len time= {len(time)}  time= {time[0:10], time[-10:]}')
dict_all = {
    'delta_V': {
        'name': 'Ошибка определения скорости',
        'data': [d_V_North_arr, d_V_East_arr],
        'time': [time for i in range(2)],
        'labels': ['Северный канал', 'Восточный канал']
        },
    'F': {
        'name': 'Ошибки определения тангажа(γ) и крена(θ)',
        'data': [F_North_arr, F_East_arr],
        'time': [time for i in range(2)],
        'labels': ['θ', 'γ']
        },
    'F_up': {
            'name': 'Ошибки определения курса(Ψ)',
            'data': [F_Up_arr],
            'time': [time],
            'labels': ['']
        },
    'coord': {
            'name': 'Ошибка определения координат',
            'data': [vector_coord_errors_arr],
            'time': [time],
            'labels': ['']
        },
}

# data = ['delta_V', 'F', 'F_up']
data = ['delta_V', 'F', 'F_up', 'coord']

make_graph_from_dict(data)


# make_graph('d_V_North_arr', [time], [d_V_North_arr], colors_graph, ['d_V_North_arr'])
# make_graph('F_North_arr', [time], [F_North_arr], colors_graph, ['F_North'])
# make_graph('F_Up_arr', [time], [F_Up_arr], colors_graph, ['F_Up'])




