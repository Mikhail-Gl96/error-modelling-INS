import numpy as np
import matplotlib.pyplot as plt


class Trajectory:
    def __init__(self, hours_in, freq_in, ):
        # Вспомогательные переменные
        self.km_to_m = 1000     # Перевод из км в метры

        # Общие переменные
        self.hours = hours_in   # Часы работы, ч
        self.S_linear = 10      # Длина пути на линейном участке, км
        self.w_speed = 30       # Угловая скорость в градусах, град/с
        self.v_min = 200 / 3.6  # Минимальная скорость объекта, км/ч / 3.6 -> м/с
        self.v_max = 300 / 3.6  # Максимальная скорость объекта, км/ч / 3.6 -> м/с
        self.time_linear = 10 * self.km_to_m / ((self.v_max + self.v_min) / 2)  # Время движения на линейном участке, с
        self.time_angle = 6     # Время движения на повороте, с
        self.time_part1_3 = 1   # Время на 1 и 3 участке при угловом движении, с

        # -------------------------------------------------------------------------

        # Переменные участка АВ             ( ЛИНЕЙНОЕ )
        self.t_ab = self.time_linear    # Время движения на участке АВ, с
        self.v0_ab = self.v_min         # Начальная скорость на участке АВ, м/с
        self.v_end_ab = self.v_max      # Конечная скорость на участке АВ, м/с

        # Переменные участка BC             ( УГЛОВОЕ )
        self.w_bc = np.deg2rad(self.w_speed)    # Угловая скорость на участке BC, м/с
        self.v_bc = self.v_end_ab               # Линейная скорость на участке BC, м/с
        self.R_bc = self.v_bc / self.w_bc       # Радиус вектор участка BC, м
        self.t_bc = self.time_angle             # Время движения на участке BC, с
        self.t_bc_part1_3 = self.time_part1_3   # Время на участве BC участок времени 1 и 3, с
        self.t_bc_part2 = self.t_bc - 2 * self.t_bc_part1_3  # Время на участве BC участок времени 3, с

        # Переменные участка CD             ( ЛИНЕЙНОЕ )
        self.t_cd = self.time_linear    # Время движения на участке CD, с
        self.v0_cd = self.v_max         # Начальная скорость на участке CD, м/с
        self.v_end_cd = self.v_min      # Конечная скорость на участке CD, м/с

        # Переменные участка DE             ( УГЛОВОЕ )
        self.w_de = np.deg2rad(self.w_speed)    # Угловая скорость на участке DE, м/с
        self.v_de = self.v_end_cd               # Линейная скорость на участке DE, м/с
        self.R_de = self.v_de / self.w_de       # Радиус вектор участка DE, м
        self.t_de = self.time_angle             # Время движения на участке DE, с
        self.t_de_part1_3 = self.time_part1_3   # Время на участве DE участок времени 1 и 3, с
        self.t_de_part2 = self.t_de - 2 * self.t_de_part1_3  # Время на участве DE участок времени 3, с

        # Кол-во прогонов А-Е (каждый по 144 секунды)
        self.end_time = int(self.hours * 3600 / (self.t_ab + self.t_bc + self.t_cd + self.t_de))
        # -------------------------------------------------------------------------

        # Тестовые переменные
        self.freq = freq_in     # Частота работы алгоритма, Гц
        self.round_num = 4      # Количество знаков после запятой

        # ---------------------------------------------------------------------------------------------------------------
        # Скорости ЛИНЕЙНЫЕ
        self.v_e_previous = 0  # Скорость на прошлом шаге восточное направление, м/с
        self.v_n_previous = 0  # Скорость на прошлом шаге северное направление,  м/с

        # Скорости УГЛОВЫЕ
        self.w_e_previous = 0  # Скорость на прошлом шаге восточное направление, м/с
        self.w_n_previous = 0  # Скорость на прошлом шаге северное направление,  м/с

        # Ускорения
        self.a_e_previous = 0  # Ускорение на прошлом шаге восточное направление, м/с^2
        self.a_n_previous = 0  # Ускорение на прошлом шаге северное направление,  м/с^2

        # Время
        self.iter_linear = int(self.time_linear * self.freq)    # Кол-во итерация на линейном участке, шт
        self.global_time_add = 0                                # Добавочное время для каждой след итерации, с
        self.time_previous = 0                                  # Метка времени на предыдущем итерации, с

        # Переменные для хранения итоговых результатов
        self.arr_time = []      # Массив времени, с

        self.arr_v_east = []    # Массив линейных скоростей по восточному направлению, м/с
        self.arr_v_north = []   # Массив линейных скоростей по северному  направлению, м/с

        self.arr_W = []         # Массив угловых скоростей, м/с

        self.arr_a_center = []  # Массив центростремительных ускорений, м/с^2
        self.arr_a_east = []    # Массив линейных ускорений по восточному направлению, м/с^2
        self.arr_a_north = []   # Массив линейных ускорений по северному  направлению, м/с^2
        self.arr_epsilon = []   # Массив угловых ускорений, м/с^2

        self.arr_alpha = []     # Массив углов поворотов, градусы

        # ---------------------------------------------------------------------------------------------------------------
        self.log_get = 0        # Флаг записи лога (1 - вкл,  0 - выкл)
        if self.log_get == 1:
            self.file = open('log_output_data.txt', 'w')    # Файл отчет по прогону мат модели

    # Уравнение, которое решает задачу на линейном участке траектории (Величины равномерно распределены на участке)
    def linear_part_equation(self, v0_in, v_end_in, t_end_in):
        v_vector = np.linspace(v0_in, v_end_in, self.iter_linear + 1)   # Вектор строка последовательностей скоростей
        t_vector = np.linspace(0, t_end_in, self.iter_linear + 1)       # Вектор строка времени
        t_vector += self.global_time_add            # Доводим время до текущего при добавке предыдущей точки
        self.global_time_add = t_vector[-1]         # Изменяем метку времени для след итераций
        # Начинаем перебор по элементам в количестве время * частоту алгоритма
        for i in range(self.iter_linear + 1):
            v_e_current = v_vector[i]
            v_n_current = 0
            a_n_current = 0
            a_center_current = 0
            w_n_current = 0
            w_e_current = 0
            if len(self.arr_alpha) < 2:
                alpha_current = 0
            else:
                alpha_current = self.arr_alpha[-1].__round__()
            if i == 0:
                dt = t_vector[i] - self.time_previous
                a_e_current = ((v_e_current - self.v_e_previous) / dt).__round__(self.round_num)
                if abs(a_e_current) > abs(200):
                    # print('jjjjjjjjjjjjjjjjjj')
                    if v_e_current == 300:
                        a_e_current = 0.1929.__round__(self.round_num)
                    elif v_e_current == 200:
                        a_e_current = -0.1929.__round__(self.round_num)
                    else:
                        a_e_current = 0
                    # print(a_e_current)
            else:
                dt = t_vector[i] - t_vector[i - 1]
                a_e_current = ((v_vector[i] - v_vector[i - 1]) / dt).__round__(self.round_num)
            # Перезапись текущих значений в прошлые для последующих расчетов (кроме самой последней записи)
            if i != (self.iter_linear + 1 - 1):
                # --------------------------------------------------
                self.arr_time.append(t_vector[i])       # Записываем время в глобальный массив
                self.arr_v_east.append(v_e_current)     # Записываем скорость по восточному каналу в глобальный массив
                self.arr_v_north.append(0)              # Записываем скорость по северному  каналу в глобальный массив
                self.arr_a_east.append(a_e_current)     # Записываем ускорения по восточному каналу в глобальный массив
                self.arr_a_north.append(0)              # Записываем ускорения по северному  каналу в глобальный массив
                self.arr_a_center.append(a_center_current)  # Записываем центростремительное ускорение
                self.arr_alpha.append(alpha_current)    # Записываем углы поворота в глобальный массив
                self.arr_epsilon.append(0)              # Записываем угловые ускорения
                self.arr_W.append(0)                    # Записываем угловые скорости
                # --------------------------------------------------
                self.v_e_previous = v_e_current
                self.v_n_previous = v_n_current
                self.a_e_previous = a_e_current
                self.a_n_previous = a_n_current
                self.w_e_previous = w_e_current
                self.w_n_previous = w_n_current
                answer = f'{str(f"time= {self.arr_time[-1].__round__(self.round_num)},").ljust(25, " ")} ' \
                    f'{str(f"alpha_real= {self.arr_alpha[-1].__round__(self.round_num)},").ljust(25, " ")} ' \
                    f'{str(f"v_e= {v_e_current.__round__(self.round_num)},").ljust(25, " ")} ' \
                    f'{str(f"v_e_previous= {self.v_e_previous.__round__(self.round_num)},").ljust(25, " ")} ' \
                    f'{str(f"dt= {dt.__round__(self.round_num)},").ljust(25, " ")} ' \
                    f'{str(f"a_e = {a_e_current.__round__(self.round_num)}").ljust(25, " ")}' \
                    f'{str(f"v_n= {v_n_current.__round__(self.round_num)},").ljust(25, " ")} ' \
                    f'{str(f"v_n_previous= {self.v_n_previous.__round__(self.round_num)},").ljust(25, " ")} ' \
                    f'{str(f"a_n = {a_n_current.__round__(self.round_num)}").ljust(25, " ")}\n'
                if self.log_get == 1:
                    self.file.write(answer)
        self.time_previous = self.global_time_add

    # Уравнение, которое решает задачу на угловом участке траектории (Величины равномерно распределены на участке)
    def angle_part_equation(self, t_in, w_in, r_in):
        t1_in, t2_in, t3_in = t_in[0], t_in[1], t_in[2]
        w_1part, w_2part, w_3part = np.rad2deg(w_in[0]), np.rad2deg(w_in[1]), np.rad2deg(w_in[2])
        time_add = self.global_time_add                 # Добавочное время для каждого интервала в секундах
        alpha_prev = self.arr_alpha[-1].__round__()     # Предыдуще значение угла альфа в градусах
        v_linear_abs = abs(w_in[0] * r_in)              # Абсолютная скорость по модулю
        alpha_previous = 0                      # Значение угла, на который мы доповорачиваем наше значение до реального

        def part1(t1_inside, w_stop):
            nonlocal time_add, alpha_prev, alpha_previous
            if len(self.arr_time) < 2:
                time_prev = 0
            else:
                time_prev = self.arr_time[-1]       # Предыдущее значение метки времени
            epsilon = (w_stop / t1_inside).__round__(self.round_num)    # Угловое ускорение
            time_out = np.linspace(time_add, t1_inside + time_add, t1_inside * self.freq + 1)   # Временная прямая
            for i in time_out:
                dt = i - time_prev                  # Временной шаг
                w_current = ((i - time_add) * epsilon).__round__(self.round_num)    # Угловая скорость на итерации
                alpha = ((i - time_add) * w_current).__round__(self.round_num)      # Угол на итерации
                alpha_real = alpha + alpha_prev     # Угол с учетом предыдущего роста
                alpha_previous = alpha_real         # Угол текущий с учетом предыдущего поворота
                v_linear = (w_in[0] * r_in)         # Линейная скорость на повороте
                # Считаем линейную скорость по восточному каналу
                v_e = v_linear_abs - abs(v_linear * np.sin(np.deg2rad(alpha_real)))
                # Считаем линейную скорость по восточному каналу
                v_n = v_linear_abs - abs(v_linear * np.cos(np.deg2rad(alpha_real)))
                # Линейное ускорение по восточному каналу
                a_e = ((v_e - self.v_e_previous) / dt).__round__(self.round_num)
                if dt == 0:
                    a_e = 0
                # Линейное ускорение по северному  каналу
                a_n = ((v_n - self.v_n_previous) / dt).__round__(self.round_num)
                a_center = r_in * pow(np.deg2rad(w_current), 2)     # Центростремительное ускорение
                if i != time_out[-1]:           # Пока не дошли до последней итерации, то выполняем
                    self.arr_epsilon.append(epsilon)    # Пишем угловое ускорение в глобальный масссив
                    self.arr_W.append(w_current)        # Пишем угловую скорость в глобальный массив
                    self.arr_time.append(i)             # Пишем время в массив
                    self.arr_alpha.append(alpha_real)   # Пишем угол в глобальный массив
                    self.arr_v_east.append(v_e)         # Запись скорости по восточному каналу
                    self.arr_v_north.append(v_n)        # Запись скорости по северному канал
                    self.arr_a_east.append(a_e)         # Запись линейного ускорения по восточному каналу
                    self.arr_a_north.append(a_n)        # Запись линейного ускорения по северному  каналу
                    self.arr_a_center.append(a_center)  # Запись центростремительного ускорения
                    answer = f'{str(f"time= {self.arr_time[-1].__round__(self.round_num + 1)},").ljust(25, " ")}' \
                        f'{str(f"alpha_real= {alpha_real.__round__(self.round_num)},").ljust(25, " ")} ' \
                        f'{str(f"v_e= {v_e.__round__(self.round_num)},").ljust(25, " ")} ' \
                        f'{str(f"v_e_previous= {self.v_e_previous.__round__(self.round_num)},").ljust(25, " ")} ' \
                        f'{str(f"dt= {dt.__round__(self.round_num)},").ljust(25, " ")} ' \
                        f'{str(f"a_e = {a_e.__round__(self.round_num)}").ljust(25, " ")}' \
                        f'{str(f"v_n= {v_n.__round__(self.round_num)},").ljust(25, " ")} ' \
                        f'{str(f"v_n_previous= {self.v_n_previous.__round__(self.round_num)},").ljust(25, " ")} ' \
                        f'{str(f"a_n = {a_n.__round__(self.round_num)}").ljust(25, " ")}\n'
                    # print(answer)
                    # input()
                    if self.log_get == 1:
                        self.file.write(answer)     # Запись лога в файл
                    self.v_e_previous = v_e     # Перезапись предыдущего значения скорости по восточному каналу
                    self.v_n_previous = v_n     # Перезапись предыдущего значения скорости по северному  каналу
                    time_prev = i               # Перезапись предыдущего значения локальной метки времени
            alpha_prev = alpha_previous             # Перезапись предудущего значения угла поворота
            time_add = time_out[-1]                 # Делаем новую метку сдвига времени

        def part2(t2_inside, w_stop):
            nonlocal time_add, alpha_prev, alpha_previous
            time_prev = self.arr_time[-1]
            time_out = np.linspace(time_add, t2_inside + time_add, t2_inside * self.freq + 1)
            for i in time_out:
                dt = i - time_prev                  # Временной шаг
                w_current = w_stop.__round__(self.round_num)    # Угловая скорость на итерации
                alpha = ((i - time_add) * w_current).__round__(self.round_num)      # Угол на итерации
                alpha_real = alpha + alpha_prev     # Угол с учетом предыдущего роста
                alpha_previous = alpha_real         # Угол текущий с учетом предыдущего поворота
                v_linear = (w_in[1] * r_in)         # Линейная скорость на повороте
                # Считаем линейную скорость по восточному каналу
                v_e = v_linear_abs - abs(v_linear * np.sin(np.deg2rad(alpha_real)))
                # Считаем линейную скорость по восточному каналу
                v_n = v_linear_abs - abs(v_linear * np.cos(np.deg2rad(alpha_real)))
                # Линейное ускорение по восточному каналу
                a_e = ((v_e - self.v_e_previous) / dt).__round__(self.round_num)
                # Линейное ускорение по северному  каналу
                a_n = ((v_n - self.v_n_previous) / dt).__round__(self.round_num)
                a_center = r_in * pow(np.deg2rad(w_current), 2)     # Центростремительное ускорение
                if a_n == 0:
                    a_n = self.arr_a_north[-1]
                if i != time_out[-1]:
                    self.arr_epsilon.append(0)          # Пишем угловое ускорение в глобальный масссив
                    self.arr_W.append(w_current)        # Пишем угловую скорость в глобальный массив
                    self.arr_time.append(i)             # Пишем время в массив
                    self.arr_alpha.append(alpha_real)   # Пишем угол в глобальный массив
                    self.arr_v_east.append(v_e)         # Запись скорости по восточному каналу
                    self.arr_v_north.append(v_n)        # Запись скорости по северному каналу
                    self.arr_a_east.append(a_e)         # Запись линейного ускорения по восточному каналу
                    self.arr_a_north.append(a_n)        # Запись линейного ускорения по северному  каналу
                    self.arr_a_center.append(a_center)  # Запись центростремительного ускорения
                    answer = f'{str(f"time= {self.arr_time[-1].__round__(self.round_num + 1)},").ljust(25, " ")}' \
                        f'{str(f"alpha_real= {alpha_real.__round__(self.round_num)},").ljust(25, " ")} ' \
                        f'{str(f"v_e= {v_e.__round__(self.round_num)},").ljust(25, " ")} ' \
                        f'{str(f"v_e_previous= {self.v_e_previous.__round__(self.round_num)},").ljust(25, " ")} ' \
                        f'{str(f"dt= {dt.__round__(self.round_num)},").ljust(25, " ")} ' \
                        f'{str(f"a_e = {a_e.__round__(self.round_num)}").ljust(25, " ")}' \
                        f'{str(f"v_n= {v_n.__round__(self.round_num)},").ljust(25, " ")} ' \
                        f'{str(f"v_n_previous= {self.v_n_previous.__round__(self.round_num)},").ljust(25, " ")} ' \
                        f'{str(f"a_n = {a_n.__round__(self.round_num)}").ljust(25, " ")}\n'
                    # print(answer)
                    # input()
                    if self.log_get == 1:
                        self.file.write(answer)     # Запись лога в файл
                    self.v_e_previous = v_e     # Перезапись предыдущего значения скорости по восточному каналу
                    self.v_n_previous = v_n     # Перезапись предыдущего значения скорости по северному  каналу
                    time_prev = i               # Перезапись предыдущего значения локальной метки времени
            alpha_prev = alpha_previous             # Перезапись предудущего значения угла поворота
            time_add = time_out[-1]                 # Делаем новую метку сдвига времени

        def part3(t3_inside, w_stop):
            nonlocal time_add, alpha_prev, alpha_previous
            time_prev = self.arr_time[-1]
            add_pos_above_zero_line = w_stop    # Установка значения нуля системы от начальной приянтой угловой скорости
            epsilon = -(w_stop / t3_inside).__round__(self.round_num)   # Угловое ускорение
            time_out = np.linspace(time_add, t3_inside + time_add, t3_inside * self.freq + 1)
            for i in time_out:
                dt = i - time_prev      # Временной шаг
                w_current = ((i - time_add) * epsilon + add_pos_above_zero_line).__round__(
                    self.round_num)     # Угловая скорость
                alpha = ((i - time_add) * (w_current + add_pos_above_zero_line)).__round__(
                    self.round_num)     # Угол на итерации
                alpha_real = alpha + alpha_prev     # Угол с учетом предыдущего роста
                alpha_previous = alpha_real         # Угол текущий с учетом предыдущего поворота
                v_linear = (w_in[2] * r_in)         # Линейная скорость на повороте
                # Считаем линейную скорость по восточному каналу
                v_e = v_linear_abs - abs(v_linear * np.sin(np.deg2rad(alpha_real)))
                # Считаем линейную скорость по восточному каналу
                v_n = v_linear_abs - abs(v_linear * np.cos(np.deg2rad(alpha_real)))
                # Линейное ускорение по восточному каналу
                a_e = ((v_e - self.v_e_previous) / dt).__round__(self.round_num)
                # Линейное ускорение по северному  каналу
                a_n = ((v_n - self.v_n_previous) / dt).__round__(self.round_num)
                a_center = r_in * pow(np.deg2rad(w_current), 2)     # Центростремительное ускорение
                if i != time_out[-1]:
                    self.arr_epsilon.append(epsilon)    # Пишем угловое ускорение в глобальный масссив
                    self.arr_W.append(w_current)        # Пишем угловую скорость в глобальный массив
                    self.arr_time.append(i)             # Пишем время в массив
                    self.arr_alpha.append(alpha_real)   # Пишем угол в глобальный массив
                    self.arr_v_east.append(v_e)         # Запись скорости по восточному каналу
                    self.arr_v_north.append(v_n)        # Запись скорости по северному каналу
                    self.arr_a_east.append(a_e)         # Запись линейного ускорения по восточному каналу
                    self.arr_a_north.append(a_n)        # Запись линейного ускорения по северному  каналу
                    self.arr_a_center.append(a_center)  # Запись центростремительного ускорения
                    answer = f'{str(f"time= {self.arr_time[-1].__round__(self.round_num + 1)},").ljust(25, " ")}' \
                        f'{str(f"alpha_real= {alpha_real.__round__(self.round_num)},").ljust(25, " ")} ' \
                        f'{str(f"v_e= {v_e.__round__(self.round_num)},").ljust(25, " ")} ' \
                        f'{str(f"v_e_previous= {self.v_e_previous.__round__(self.round_num)},").ljust(25, " ")} ' \
                        f'{str(f"dt= {dt.__round__(self.round_num)},").ljust(25, " ")} ' \
                        f'{str(f"a_e = {a_e.__round__(self.round_num)}").ljust(25, " ")}' \
                        f'{str(f"v_n= {v_n.__round__(self.round_num)},").ljust(25, " ")} ' \
                        f'{str(f"v_n_previous= {self.v_n_previous.__round__(self.round_num)},").ljust(25, " ")} ' \
                        f'{str(f"a_n = {a_n.__round__(self.round_num)}").ljust(25, " ")}\n'
                    # print(answer)
                    # input()
                    if self.log_get == 1:
                        self.file.write(answer)     # Запись лога в файл
                    self.v_e_previous = v_e     # Перезапись предыдущего значения скорости по восточному каналу
                    self.v_n_previous = v_n     # Перезапись предыдущего значения скорости по северному  каналу
                    time_prev = i               # Перезапись предыдущего значения локальной метки времени
            alpha_prev = alpha_previous             # Перезапись предудущего значения угла поворота
            time_add = time_out[-1]                 # Делаем новую метку сдвига времени
        part1(t1_in, w_1part)   # Просчет 1 из 3 частей угловой траектории (угловой разгон)
        part2(t2_in, w_2part)   # Просчет 2 из 3 частей угловой траектории (без ускорения)
        part3(t3_in, w_3part)   # Просчет 3 из 3 частей угловой траектории (угловое торможение)
        self.global_time_add = time_add     # Обновляем метку времени
        self.time_previous = self.arr_time[-1]

    def make_trajectory(self):
        for i in range(self.end_time):
            self.linear_part_equation(self.v0_ab, self.v_end_ab, self.t_ab)  # Участок АВ
            self.angle_part_equation([self.t_bc_part1_3, self.t_bc_part2, self.t_bc_part1_3], [self.w_bc, self.w_bc, self.w_bc], self.R_bc)  # Участок BC
            self.linear_part_equation(self.v0_cd, self.v_end_cd, self.t_cd)  # Участок CD
            self.angle_part_equation([self.t_de_part1_3, self.t_de_part2, self.t_de_part1_3], [-self.w_de, -self.w_de, -self.w_de], self.R_de)  # Участок DE
        if self.log_get == 1:
            self.file.close()

    def get_graphics(self, args):
        time = self.arr_time
        dict_all = {
            'v_east': {
                'name': 'Линейная скорость по восточному каналу',
                'data': self.arr_v_east,
                'time': time,
                'ylabel': 'Скорость, м/с'
                },
            'v_north': {
                'name': 'Линейная скорость по северному каналу',
                'data': self.arr_v_north,
                'time': time,
                'ylabel': 'Скорость, м/с'
            },
            'W': {
                'name': 'Угловая скорость',
                'data': self.arr_W,
                'time': time,
                'ylabel': 'Скорость, м/с'
            },
            'a_east': {
                'name': 'Линейноу ускорение по восточному каналу',
                'data': self.arr_a_east,
                'time': time,
                'ylabel': 'Ускорение, м/с^2'
            },
            'a_north': {
                'name': 'Линейноу ускорение по северному каналу',
                'data': self.arr_a_north,
                'time': time,
                'ylabel': 'Ускорение, м/с^2'
            },
            'epsilon': {
                'name': 'Угловое ускорение',
                'data': self.arr_epsilon,
                'time': time,
                'ylabel': 'Ускорение, м/с^2'
            },
            'alpha': {
                'name': 'Угол поворота',
                'data': self.arr_alpha,
                'time': time,
                'ylabel': 'Угол, град'
            },
            'a_center': {
                'name': 'Центростремительное ускорение',
                'data': self.arr_a_center,
                'time': time,
                'ylabel': 'Ускорение, м/с^2'
            },
        }

        for name in args:
            print(f'Вывод графика {name} ')
            local_data = dict_all.get(f'{name}')
            plt.plot(local_data.get("time"),  local_data.get("data"))  # График F(t)
            plt.title(f'{local_data.get("name")}')
            plt.xlabel("Время, с")
            plt.ylabel(f'{local_data.get("ylabel")}')
            plt.show()

    def get_a_north_arr(self):
        return self.arr_a_north

    def get_a_east_arr(self):
        return self.arr_a_east

    def get_a_center_arr(self):
        return self.arr_a_center

    def get_w_up_arr(self):
        return self.arr_W

# data = ['v_east', 'v_north', 'W', 'epsilon', 'alpha', 'a_east', 'a_north', 'a_center']
# q = Trajectory()
# q.make_trajectory()
# q.get_graphics(data)
