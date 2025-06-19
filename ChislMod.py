import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import uniform_filter1d

# Отключаем смещение меток и научную нотацию
mpl.rcParams['axes.formatter.useoffset'] = False
mpl.rcParams['axes.formatter.limits'] = [-5, 5]

# === 1) Вводимые параметры ===
nx, ny, nz = 130, 52, 52  # исходное разрешение
Lx = 1.0  # длина канала, м
S0, S1 = 0.05, 0.30  # сторона квадрата на входе/выходе, м

T_in = 378.15  # температура пара на входе, K
T_sat = 373.15  # температура насыщения, K (фиксированная)
T_wall_init = 294.15  # начальная температура стенок
u_flow = 0.5  # скорость потока на входе, м/с

rho_v0, cp_v = 0.25, 1500.0  # плотность пара (кг/м³), теплоёмкость (Дж/(кг·К))
kappa_v0 = 0.015  # теплопроводность пара, Вт/(м·К)
latent_h = 2.257e6  # скрытая теплота конденсации, Дж/кг
R_v = 461.5  # газовая постоянная для водяного пара, Дж/(кг·К)
M_v = 0.018  # молярная масса воды, кг/моль

dt = 0.0001  # шаг по времени, с
total_t = 300.0  # время моделирования, с
# 300 / 0.0001 = 3.000.000

rho_w = 7800.0  # плотность стенки (сталь), кг/м³
cp_w = 500.0  # теплоёмкость стенки (сталь), Дж/(кг·K)
th_w = 0.006  # толщина стенки, м
kappa_w = 0.6  # теплопроводность плёнки воды, Вт/(м·К)
rho_water = 1000.0  # плотность воды, кг/м³

P0 = 120790.0  # начальное давление, Па
P_in = 142000.0  # начальное давление насыщенного пара при T_in = 378.15 K
R = 8.314  # универсальная газовая постоянная, Дж/(моль·К)
M_air = 0.029  # молярная масса воздуха, кг/моль
gamma = 1.3  # показатель адиабаты для смеси

# === 2) Функция моделирования ===
simulation_counter = 0
total_simulations = 10  # 1 основной + 3 сценария + 6 сходимость

def simulate(nx, ny, nz, T_in, T_sat, T_wall_init, u_flow, dt, total_t, run_label="Main Run"):
    global simulation_counter
    simulation_counter += 1
    print(f"\n=== Начало симуляции {simulation_counter}/{total_simulations}: {run_label} ===")

    # Инициализация координат и геометрических параметров
    x = np.linspace(0, Lx, nx)  # Создание массива координат x вдоль канала
    dx = x[1] - x[0]  # Вычисление шага по оси x
    steps = int(total_t / dt)  # Вычисление общего числа временных шагов

    i_arr = np.arange(nx)  # Создание массива индексов для оси x
    Si_arr = S0 + (S1 - S0) * i_arr / (nx - 1)  # Линейное расширение сечения канала от S0 до S1
    dy_arr = Si_arr / (ny - 1)  # Вычисление шага по y для каждого сечения
    dz_arr = Si_arr / (nz - 1)  # Вычисление шага по z для каждого сечения

    dy_arr = np.broadcast_to(np.expand_dims(np.expand_dims(dy_arr, axis=1), axis=2), (nx, ny, nz))  # Расширение dy_arr для 3D-решетки
    dz_arr = np.broadcast_to(np.expand_dims(np.expand_dims(dz_arr, axis=1), axis=2), (nx, ny, nz))  # Расширение dz_arr для 3D-решетки

    # Установка начальных условий
    rho_loc = 1.2 * (S0 / Si_arr) ** 2  # Начальная плотность воздуха с учетом сужения канала
    rho_loc = np.broadcast_to(np.expand_dims(np.expand_dims(rho_loc, axis=1), axis=2), (nx, ny, nz))  # Расширение rho_loc в 3D
    rho_loc = rho_loc.copy()  # Создание копии для предотвращения изменения исходных данных
    print(f"Форма rho_loc после копирования: {rho_loc.shape}")  # Вывод формы массива rho_loc

    u_loc = u_flow * (S0 / Si_arr) ** 2  # Начальная скорость потока с учетом расширения
    u_loc = np.broadcast_to(np.expand_dims(np.expand_dims(u_loc, axis=1), axis=2), (nx, ny, nz))  # Расширение u_loc в 3D
    u_loc = u_loc.copy()  # Создание копии для предотвращения изменения исходных данных
    print(f"Форма u_loc после копирования: {u_loc.shape}")  # Вывод формы массива u_loc

    P_loc = P0 * (rho_loc[:, :, 0] * u_loc[:, :, 0] ** 2) / (rho_v0 * u_flow ** 2)  # Начальное давление на основе плотности и скорости
    P_loc = np.mean(P_loc, axis=1)  # Среднее давление по сечению
    P_loc = np.broadcast_to(np.expand_dims(P_loc, axis=(1, 2)), (nx, ny, nz))  # Расширение P_loc в 3D

    T_sat_loc = np.full((nx, ny, nz), T_sat)  # Создание массива фиксированной температуры насыщения
    print(f"Форма T_sat_loc: {T_sat_loc.shape}")  # Вывод формы массива T_sat_loc

    T_wall = np.full(nx, T_wall_init)  # Инициализация температуры стенок
    delta_film = np.zeros(nx)  # Инициализация толщины конденсатной пленки

    cell_vol = dx * dy_arr * dz_arr  # Вычисление объема ячейки для каждого узла
    total_vol = dx * Si_arr ** 2  # Общий объем сечения

    T = np.full((nx, ny, nz), T_wall_init).copy()  # Инициализация температуры поля холодным состоянием
    cond_mass = np.zeros((nx, ny, nz))  # Инициализация массива массы конденсата
    M_tot = np.zeros(steps)  # Массив для накопления общей массы конденсата

    Cw = rho_w * cp_w * th_w  # Вычисление теплоёмкости стенки
    prev_Mx = np.zeros(nx)  # Предыдущее значение массы конденсата по оси x

    kappa_v = kappa_v0 * (T_wall_init / T_in) ** 0.5  # Начальная теплопроводность пара
    alpha = kappa_v / (rho_loc * cp_v)  # Начальный коэффициент диффузии
    max_alpha = np.max(alpha)  # Максимальное значение коэффициента диффузии
    dt_max = (dx ** 2) / (2 * max_alpha)  # Максимально допустимый шаг времени для стабильности
    print(f"Максимальное alpha: {max_alpha:.2e} м²/с")  # Вывод максимального alpha
    print(f"Максимально допустимый dt для диффузии: {dt_max:.2e} с")  # Вывод максимального dt
    if dt > dt_max:
        print(f"ВНИМАНИЕ: dt={dt} превышает допустимое значение {dt_max:.2e} с. Возможна нестабильность!")  # Предупреждение о нестабильности

    prev_M_tot = 0.0  # Предыдущее значение общей массы конденсата

    # Основной цикл симуляции
    for step in range(steps):
        # Применение граничных условий
        T[0, :, :] = T_in  # Установка температуры на входе

        T_clipped = np.clip(T, 273.15, T_in)  # Ограничение температуры в допустимом диапазоне
        speed_factor = 1 + 0.5 * (u_loc / u_flow) ** 2  # Корректировка скорости потока
        kappa_v = kappa_v0 * (T_clipped / T_in) ** 0.5 * speed_factor  # Обновление теплопроводности с учетом температуры и скорости
        alpha = kappa_v / (rho_loc * cp_v)  # Обновление коэффициента диффузии
        alpha = np.where(np.isfinite(alpha), alpha, 0.0)  # Замена нечисловых значений нулями

        # Применение адиабатической поправки
        if step > 0:  # Пропуск на первом шаге
            volume_ratio = (Si_arr / S0) ** 2  # Отношение объемов сечений
            volume_ratio_3d = np.broadcast_to(np.expand_dims(np.expand_dims(volume_ratio, axis=1), axis=2), (nx, ny, nz))  # Расширение в 3D
            T_adiabatic = T * (volume_ratio_3d ** (1 - gamma))  # Вычисление адиабатической температуры
            T = np.where(T_adiabatic > 273.15, T_adiabatic, T)  # Обновление температуры с учетом адиабаты

        # Вычисление производных для теплопроводности
        d2_x = (np.roll(T, -1, axis=0) - 2 * T + np.roll(T, 1, axis=0))[1:-1, 1:-1, 1:-1] / dx ** 2  # Вторая производная по x
        d2_x = np.where(np.isfinite(d2_x), d2_x, 0.0)  # Замена нечисловых значений нулями
        T_diff = T[1:-1, 1:-1, 1:-1] - np.roll(T, 1, axis=0)[1:-1, 1:-1, 1:-1]  # Разность температур для конвекции
        conv = u_loc[1:-1, 1:-1, 1:-1] * T_diff / dx  # Конвективный член
        conv = np.where(np.isfinite(conv), conv, 0.0)  # Замена нечисловых значений нулями
        d2_y = (np.roll(T, -1, axis=1) - 2 * T + np.roll(T, 1, axis=1))[1:-1, 1:-1, 1:-1] / (dy_arr[1:-1, 1:-1, 1:-1] ** 2)  # Вторая производная по y
        d2_y = np.where(np.isfinite(d2_y), d2_y, 0.0)  # Замена нечисловых значений нулями
        d2_z = (np.roll(T, -1, axis=2) - 2 * T + np.roll(T, 1, axis=2))[1:-1, 1:-1, 1:-1] / (dz_arr[1:-1, 1:-1, 1:-1] ** 2)  # Вторая производная по z
        d2_z = np.where(np.isfinite(d2_z), d2_z, 0.0)  # Замена нечисловых значений нулями

        T[1:-1, 1:-1, 1:-1] += dt * (alpha[1:-1, 1:-1, 1:-1] * (d2_x + d2_y + d2_z) - conv)  # Обновление температуры с учетом диффузии и конвекции
        T = np.clip(T, 273.15, T_in)  # Ограничение температуры в допустимом диапазоне

        # Моделирование конденсации
        mask_vol = (T < T_sat_loc) & (step > 100)  # Условие конденсации после прогрева
        deltaT_vol = np.where(mask_vol, np.minimum(T_sat_loc - T, 50.0), 0.0)  # Разность температур для конденсации
        mcd_vol = rho_loc * cp_v * deltaT_vol / latent_h  # Масса конденсата на единицу объема
        cond_mass_vol = mcd_vol * dt * cell_vol  # Объемный прирост массы конденсата
        cond_mass += cond_mass_vol  # Накопление массы конденсата
        T += (latent_h * mcd_vol * dt) / (rho_loc * cp_v)  # Корректировка температуры из-за теплоты конденсации
        T = np.clip(T, 273.15, T_in)  # Ограничение температуры
        T[mask_vol] = T_sat_loc[mask_vol]  # Установка температуры насыщения в зонах конденсации

        # Вычисление конденсата на стенках
        cond_mass_wall = np.sum(cond_mass[:, 0, :] + cond_mass[:, -1, :], axis=(1,)) + \
                         np.sum(cond_mass[:, 1:-1, 0] + cond_mass[:, 1:-1, -1], axis=(1,))  # Суммирование массы на границах
        wall_fraction = 0.8  # Доля конденсата, прилипающего к стенкам
        cond_mass_wall *= wall_fraction  # Учет доли конденсата на стенках
        delta_film = cond_mass_wall / (rho_water * Si_arr)  # Вычисление толщины пленки

        # Вычисление теплового потока на стенках
        q_wall = np.zeros(nx)  # Инициализация массива теплового потока
        for i in range(1, nx - 1):  # Цикл по внутренним узлам
            dy = dy_arr[i, 0, 0]  # Шаг по y в текущем сечении
            T_mean = np.mean(T[i, 1:-1, 1:-1])  # Средняя температура в сечении
            grad = (T_mean - T_wall[i]) / dy  # Градиент температуры
            kappa_v_i = kappa_v0 * (T_mean / T_in) ** 0.5 * speed_factor[i, 1, 1]  # Локальная теплопроводность
            R_gas = 5 * dy / kappa_v_i  # Сопротивление газа
            R_film = delta_film[i] / kappa_w  # Сопротивление пленки
            q_wall[i] = grad / max(R_gas + R_film, 1e-10) if grad > 0 else 0.0  # Тепловой поток с защитой от деления на ноль
            if not np.isfinite(q_wall[i]):  # Проверка на конечность значения
                q_wall[i] = 0.0  # Установка нуля для нечисловых значений

        T_wall += dt * q_wall / Cw  # Обновление температуры стенок
        T_wall = np.clip(T_wall, T_wall_init, T_in)  # Ограничение температуры стенок

        T[:, 0, :] = T_wall[:, None]  # Установка температуры на границе y=0
        T[:, -1, :] = T_wall[:, None]  # Установка температуры на границе y=S
        T[:, :, 0] = T_wall[:, None]  # Установка температуры на границе z=0
        T[:, :, -1] = T_wall[:, None]  # Установка температуры на границе z=S

        M_tot[step] = np.sum(cond_mass)  # Накопление общей массы конденсата

        Mx = np.sum(cond_mass, axis=(1, 2))  # Суммирование массы по сечениям
        dM = Mx - prev_Mx  # Прирост массы
        prev_Mx = Mx.copy()  # Обновление предыдущего значения
        rho_loc -= (dM / total_vol)[:, None, None]  # Корректировка плотности из-за конденсации
        rho_loc = np.maximum(rho_loc, 1e-6)  # Ограничение плотности снизу

        # Вывод промежуточных результатов
        if step % 250000 == 0:  # Проверка через каждые 4000 шагов
            current_M_tot = M_tot[step] if step < steps else M_tot[-1]  # Текущая масса конденсата
            mass_increment = current_M_tot - prev_M_tot  # Прирост массы
            prev_M_tot = current_M_tot  # Обновление предыдущего значения
            avg_T = np.mean(T)  # Средняя температура
            avg_T_sat = np.mean(T_sat_loc)  # Средняя температура насыщения
            avg_q_wall = np.mean(q_wall) if np.any(q_wall) else 0.0  # Средний тепловой поток
            avg_R_total = np.mean(R_gas + R_film) if np.any(R_gas + R_film) else 0.0  # Среднее сопротивление
            cond_mass_in_flow = np.sum(cond_mass[:, 1:-1, 1:-1])  # Масса конденсата в потоке
            elapsed_time = step * dt  # Прошедшее время
            print(f"Шаг {step}:")
            print(f"  Прошедшее время:\t\t\t\t\t\t\t{elapsed_time:.2f} с")
            print(f"  Средняя температура стенки:\t\t\t\t{np.mean(T_wall):.1f} К")
            print(f"  Средняя температура пара:\t\t\t\t\t{avg_T:.1f} К")
            print(f"  Средняя температура насыщения:\t\t\t{avg_T_sat:.1f} К")
            print(f"  Средний тепловой поток:\t\t\t\t\t{avg_q_wall:.2e} Вт/м²")
            print(f"  Среднее термическое сопротивление:\t\t{avg_R_total:.2e} м²·К/Вт")
            print(f"  Масса конденсата в потоке (без стенок):\t{cond_mass_in_flow:.10f} кг")
            print(f"  Суммарная масса конденсата:\t\t\t\t{current_M_tot:.10f} кг")
            print(f"  Прирост массы конденсата:\t\t\t\t\t{mass_increment:.10f} кг")

    # Финальные вычисления
    cond_density = cond_mass / cell_vol  # Вычисление плотности конденсата
    avg_density = np.sum(cond_density * cell_vol, axis=(1, 2)) / np.sum(cell_vol, axis=(1, 2))  # Средняя плотность
    M_x = np.sum(cond_mass, axis=(1, 2))  # Масса по оси x

    print("Моделирование завершено успешно!")
    print(f"Финальная суммарная масса конденсата: {M_tot[-1]:.10f} кг")
    print(f"Средняя температура стенки: {np.mean(T_wall):.1f} К")
    print(f"Финальная масса конденсата в потоке: {np.sum(cond_mass[:, 1:-1, 1:-1]):.10f} кг")
    print(f"=== Конец симуляции: {run_label} ===\n")
    return x, T, cond_density, avg_density, M_tot, M_x, T_wall, delta_film, Si_arr, cond_mass

# === 3) Запуск модели ===
x, T, cond_density, avg_density, M_tot, M_x, T_wall, delta_film, Si_arr, cond_mass = simulate(
    nx, ny, nz, T_in, T_sat, T_wall_init, u_flow, dt, total_t, run_label="Основной запуск"
)

# Обрезаем шумы
cond_density = np.clip(cond_density, 0, None)
M_x = np.clip(M_x, 0, None)

# Сохранение данных Base
M_base = M_tot[-1]

figs = []
captions = []

# === 4) 2D-срезы в поперечном сечении ===
i_mid = nx // 2
Si_mid = S0 + (S1 - S0) * (i_mid / (nx - 1))

y = np.linspace(0, Si_mid, ny)
z = np.linspace(0, Si_mid, nz)

T_slice = T[i_mid]
rho_slice = cond_density[i_mid]

y_p = np.concatenate([[y[0]], y, [y[-1]]])
z_p = np.concatenate([[z[0]], z, [z[-1]]])
T_p = np.pad(T_slice, ((1, 1), (1, 1)), mode='edge')
rho_p = np.pad(rho_slice, ((1, 1), (1, 1)), mode='edge')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# График температуры
vmin_T = 294.0  # Начальная температура стенок
vmax_T = 378.15  # Температура на входе
levels_T = np.linspace(vmin_T, vmax_T, 512)
im1 = ax1.contourf(y_p, z_p, T_p.T, levels=levels_T, cmap='magma', vmin=vmin_T, vmax=vmax_T)
ax1.set_title(f"Температура, x={x[i_mid]:.2f} м")
ax1.set_xlabel('Y [м]')
ax1.set_ylabel('Z [м]')
ax1.set_xlim(0, Si_mid)
ax1.set_ylim(0, Si_mid)
cbar1 = fig.colorbar(im1, ax=ax1, ticks=np.arange(300, 380, 20))
cbar1.set_label('T [K]')

# График плотности конденсата
vmax_rho = np.max(rho_p)
if vmax_rho <= 0:
    ax2.text(0.5, 0.5, 'Нет конденсата', ha='center', va='center', fontsize=12)
else:
    levels_rho = np.linspace(0, vmax_rho, 256)
    im2 = ax2.contourf(y_p, z_p, rho_p.T, levels=levels_rho, cmap='Blues', vmin=0, vmax=vmax_rho)
    ax2.set_title(f"Плотность конденсата, x={x[i_mid]:.2f} м")
    ax2.set_xlabel('Y [м]')
    ax2.set_ylabel('Z [м]')
    ax2.set_xlim(0, Si_mid)
    ax2.set_ylim(0, Si_mid)
    ticks_rho = np.linspace(0, vmax_rho, 6)
    cbar2 = fig.colorbar(im2, ax=ax2, ticks=ticks_rho, format='%.3f')
    cbar2.set_label('ρ_cond [кг/м³]')

figs.append(fig)
plt.show()
plt.close(fig)

# === 5) Продольный анализ ===
fig, (p1, p2, p3) = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
avg_density_smooth = uniform_filter1d(avg_density, size=5)
M_x_smooth = uniform_filter1d(M_x, size=5)
T_wall_smooth = uniform_filter1d(T_wall, size=5)
p1.plot(x, avg_density_smooth)
p1.set_title('Средняя плотность конденсата')
p1.set_xlabel('x [м]')
p1.set_ylabel('ρ_cond [кг/м³]')
p1.ticklabel_format(style='plain', axis='y')

p2.plot(x, M_x_smooth)
p2.set_title('Масса конденсата на сечении')
p2.set_xlabel('x [м]')
p2.set_ylabel('M_x [кг]')
p2.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

p3.plot(x, T_wall_smooth)
p3.set_title('Температура стенки вдоль канала')
p3.set_xlabel('x [м]')
p3.set_ylabel('T_wall [K]')
p3.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

figs.append(fig)
plt.close(fig)

# === 6) Тепловой поток на стенке Y=0 ===
q_wall = np.zeros(nx)
for i in range(nx):
    Si = Si_arr[i]
    dy = Si / (ny - 1)
    T_mean = np.mean(T[i, 1:-1, 1:-1])
    grad = (T_mean - T_wall[i]) / dy
    kappa_v = kappa_v0 * (T_mean / T_in) ** 0.5
    R_gas = 5 * dy / kappa_v
    R_film = delta_film[i] / kappa_w
    q_wall[i] = grad / max(R_gas + R_film, 1e-10)

fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
ax.plot(x, q_wall, '-', lw=2)
ax.set_xticks(x[::10])
ax.set_xticklabels([f"{val:.2f}" for val in x[::10]])
ax.set_title('Тепловой поток на стенке Y=0 вдоль канала')
ax.set_xlabel('x [м]')
ax.set_ylabel('q$_w$ [Вт/м²]')
ax.ticklabel_format(style='plain', axis='y')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid(True)

figs.append(fig)
plt.close(fig)

# === 7) Временная эволюция суммарной массы конденсата ===
t = np.linspace(0, total_t, len(M_tot))
fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
ax.plot(t, M_tot, '-')
ax.set_title('Суммарная масса конденсата vs время')
ax.set_xlabel('t [с]')
ax.set_ylabel('M_tot [кг]')
ax.grid(True)

figs.append(fig)
plt.close(fig)

# === 8) Сходимость по сетке (шаг по nx=20) ===
ny0, nz0 = ny, nz
nx_vals = list(range(50, 151, 20))
M_end = []
for idx, nx2 in enumerate(nx_vals):
    ny2 = max(3, int(ny0 * nx2 / nx))
    nz2 = max(3, int(nz0 * nx2 / nx))
    _, _, _, _, M2, _, _, _, _, _ = simulate(nx2, ny2, nz2, T_in, T_sat, T_wall_init, u_flow, dt, total_t,
                                          run_label=f"Сходимость nx={nx2}")
    M_end.append(M2[-1])

fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
ax.plot(nx_vals, M_end, marker='o', linewidth=2)
ax.set_title('Сходимость массы конденсата по сетке (шаг=20)')
ax.set_xlabel('nx')
ax.set_ylabel('M_tot [кг]')
ax.grid(True)

figs.append(fig)
plt.close(fig)

# === 9) Коэффициент теплоотдачи вдоль канала ===
fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
h = np.zeros(nx)
for i in range(nx):
    Si = Si_arr[i]
    dy = Si / (ny - 1)
    T_mean = np.mean(T[i, 1:-1, 1:-1])
    kappa_v = kappa_v0 * (T_mean / T_in) ** 0.5
    Rgas = 5 * dy / kappa_v
    Rfilm = delta_film[i] / kappa_w
    h[i] = 1.0 / max(Rgas + Rfilm, 1e-10)

ax.plot(x, h, '-')
ax.set_title('Коэффициент теплоотдачи вдоль канала')
ax.set_xlabel('x [м]')
ax.set_ylabel('h [Вт/(м²·K)]')
ax.ticklabel_format(style='plain', axis='y')
ax.grid(True)

figs.append(fig)
plt.close(fig)

# === 10) Сравнение конечной массы конденсата в сценариях ===
scenarios = [
    ('Base', T_in, T_sat, T_wall_init, u_flow, M_base),
    ('T_in+10', T_in + 10, T_sat, T_wall_init, u_flow, None),
    ('T_wall+10', T_in, T_sat, T_wall_init + 10, u_flow, None),
    ('u_flow×2', T_in, T_sat, T_wall_init, u_flow * 2, None),
]
names, vals_wall, vals_flow = [], [], []
for idx, (name, Tin, Tsat, Tw, uf, M_val) in enumerate(scenarios):
    if name == 'Base' and M_val is not None:
        total_mass = M_val
        cond_mass_wall = np.sum(np.sum(cond_mass[:, 0, :], axis=1) + np.sum(cond_mass[:, -1, :], axis=1) +
                                np.sum(cond_mass[:, 1:-1, 0], axis=1) + np.sum(cond_mass[:, 1:-1, -1], axis=1)) * 0.8
        cond_mass_flow = total_mass - cond_mass_wall
    else:
        x, T, cond_density, avg_density, M_tot, M_x, T_wall, delta_film, Si_arr, cond_mass = simulate(
            nx, ny, nz, Tin, Tsat, Tw, uf, dt, total_t, run_label=f"Сценарий: {name}")
        total_mass = M_tot[-1]
        cond_mass_wall = np.sum(np.sum(cond_mass[:, 0, :], axis=1) + np.sum(cond_mass[:, -1, :], axis=1) +
                                np.sum(cond_mass[:, 1:-1, 0], axis=1) + np.sum(cond_mass[:, 1:-1, -1], axis=1)) * 0.8
        cond_mass_flow = total_mass - cond_mass_wall
    names.append(name)
    vals_wall.append(cond_mass_wall)
    vals_flow.append(cond_mass_flow)

fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
bar_width = 0.35
x_pos = np.arange(len(names))
ax.bar(x_pos - bar_width/2, vals_wall, bar_width, label='Стенки', color='b')
ax.bar(x_pos + bar_width/2, vals_flow, bar_width, label='Поток', color='#01cbfe', bottom=vals_wall)
ax.set_xticks(x_pos)
ax.set_xticklabels(names, rotation=20)
ax.set_title('Конечная масса конденсата в сценариях')
ax.set_ylabel('M_tot [кг]')
ax.legend()
ax.tick_params(axis='x', rotation=20)

figs.append(fig)
plt.close(fig)

# === 11) Создание PDF-отчёта ===
with PdfPages('report.pdf') as pdf:
    fig_params = plt.figure(figsize=(8.27, 11.69))
    plt.axis('off')

    def wrap_text(text, max_length=20):
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        for word in words:
            word_length = len(word)
            if current_length + word_length + (1 if current_line else 0) > max_length:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = word_length
            else:
                current_line.append(word)
                current_length += word_length + 1
        if current_line:
            lines.append(" ".join(current_line))
        return "\n".join(lines)

    table_data = [
        ["Обозначение", "Значение", "Единица", "Описание"],
        ["nx", f"{nx}", "-", wrap_text("число ячеек вдоль X")],
        ["ny", f"{ny}", "-", wrap_text("число ячеек вдоль Y")],
        ["nz", f"{nz}", "-", wrap_text("число ячеек вдоль Z")],
        ["Lx", f"{Lx:.3f}", "м", wrap_text("длина канала")],
        ["S0, S1", f"{S0:.3f}, {S1:.3f}", "м", wrap_text("сторона квадрата в начале/конце")],
        ["T_in", f"{T_in:.2f}", "K", wrap_text("температура пара на входе")],
        ["T_sat", f"{T_sat:.2f}", "K", wrap_text("температура насыщения")],
        ["T_wall_init", f"{T_wall_init:.2f}", "K", wrap_text("начальная температура стенок")],
        ["u_flow", f"{u_flow:.3f}", "м/с", wrap_text("скорость потока на входе")],
        ["dt", f"{dt:.3f}", "с", wrap_text("шаг по времени")],
        ["total_t", f"{total_t:.1f}", "с", wrap_text("время моделирования")],
        ["ρ_v0", f"{rho_v0}", "кг/м³", wrap_text("плотность пара")],
        ["c_p", f"{cp_v:.1f}", "Дж/(кг·K)", wrap_text("теплоёмкость пара")],
        ["κ_v", f"{kappa_v0:.3f}", "Вт/(м·К)", wrap_text("теплопроводность пара")],
        ["λ", f"{latent_h:.2e}", "Дж/кг", wrap_text("скрытая теплота конденсации")],
        ["ρ_w", rho_w, "кг/м³", wrap_text("плотность стенки")],
        ["c_w", f"{cp_w:.1f}", "Дж/(кг·K)", wrap_text("теплоёмкость стенки")],
        ["ws", f"{th_w:.3f}", "м", wrap_text("толщина стенки")],
        ["κ_w", f"{kappa_v0:.3f}", "Вт/(м·К)", wrap_text("теплопроводность плёнки воды")],
        ["ρ_water", f"{rho_water:.1f}", "кг/м³", wrap_text("плотность воды")],
        ["P0", f"{P0:.0f}", "Па", wrap_text("начальное давление")],
    ]

    tbl = plt.table(cellText=table_data, loc='center', colWidths=[0.2, 0.2, 0.2, 0.4])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2)

    for key, cell in tbl.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(1)

    pdf.savefig(fig_params)
    plt.close(fig_params)

    for fig in figs:
        pdf.savefig(fig)
        plt.close(fig)