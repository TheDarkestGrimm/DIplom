import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

# Отключаем смещение меток и научную нотацию
mpl.rcParams['axes.formatter.useoffset'] = False
mpl.rcParams['axes.formatter.limits']    = [-5, 5]

# === 1) Вводимые параметры ===
nx, ny, nz = 100, 40, 40       # ячейки вдоль X и в поперечных Y,Z
Lx = 1.0                       # длина канала, м
S0, S1 = 0.05, 0.15            # сторона квадрата на входе/выходе, м

T_in        = 393.15           # температура пара на входе, K
T_sat       = 373.15           # температура насыщения, K
T_wall_init = 300.0            # начальная температура стенок, K
u_flow      = 0.5              # скорость потока на входе, м/с

rho_v0, cp_v = 0.6, 2000.0     # исходная плотность пара (кг/м³), теплоёмкость (Дж/(кг·K))
kappa_v      = 0.025           # теплопроводность пара, Вт/(м·K)
latent_h     = 2.257e6         # скрытая теплота конденсации, Дж/кг

dt      = 0.01                 # шаг по времени, с
total_t = 120.0                # время моделирования, с

rho_w     = 7800.0             # плотность стенки, кг/м³
cp_w      = 500.0              # теплоёмкость стенки, Дж/(кг·K)
th_w      = 0.005              # толщина стенки, м
kappa_w   = 0.6                # теплопроводность плёнки воды, Вт/(м·K)
rho_water = 1000.0             # плотность воды, кг/м³

# === 2) Функция моделирования ===
def simulate(nx, ny, nz, T_in, T_wall_init, u_flow):
    x  = np.linspace(0, Lx, nx)
    dx = x[1] - x[0]
    steps = int(total_t / dt)

    # 2.0) Профиль расширения
    i_arr   = np.arange(nx)
    Si_arr  = S0 + (S1 - S0) * i_arr/(nx-1)
    # рассчитываем поперечные шаги
    dy_arr  = Si_arr/(ny-1)
    dz_arr  = Si_arr/(nz-1)

    # исходная плотность и скорость пара (по закону сохранения массы)
    rho_loc = rho_v0 * (S0/Si_arr)**2
    u_loc   = u_flow * (S0/Si_arr)**2
    # диффузионный коэффициент
    alpha   = kappa_v/(rho_loc*cp_v)

    # начальные поля стенок и плёнки
    T_wall     = np.full(nx, T_wall_init)
    delta_film = np.zeros(nx)

    # объёмы ячеек
    cell_vol = np.zeros((nx, ny, nz))
    for i in range(nx):
        dy, dz = dy_arr[i], dz_arr[i]
        cell_vol[i] = dx * dy * dz
    total_vol = dx * Si_arr**2  # суммарный объём секции i

    # поля
    T         = np.full((nx, ny, nz), T_in)
    cond_mass = np.zeros((nx, ny, nz))
    M_tot     = np.zeros(steps)

    # теплоёмкость стенки (на единицу площади)
    Cw = rho_w * cp_w * th_w

    # чтобы отслеживать, сколько уже конденсировалось в каждом i
    prev_Mx = np.zeros(nx)

    for step in range(steps):
        Tn = T.copy()

        # 2.1) диффузия+конвекция по X (векторно)
        d2   = (
            T[2:,1:-1,1:-1]
            - 2*T[1:-1,1:-1,1:-1]
            + T[:-2,1:-1,1:-1]
        ) / dx**2
        conv = u_loc[1:-1,None,None] * (
            T[1:-1,1:-1,1:-1] - T[:-2,1:-1,1:-1]
        ) / dx
        Tn[1:-1,1:-1,1:-1] = (
            T[1:-1,1:-1,1:-1]
            + dt*(alpha[1:-1,None,None]*d2 - conv)
        )

        # 2.2) классическая конденсация при Tn < T_sat
        mask = Tn < T_sat
        deltaT = np.where(mask, T_sat - Tn, 0.0)
        mcd = rho_v0 * cp_v * deltaT / latent_h  # кг/(м³·с)
        cond_mass += mcd * dt * cell_vol  # кг
        Tn[mask] = T_sat

        # 2.3) BC: после всего «закатываем» T_wall на четырёх гранях
        Tn[:, 0, :] = T_wall[:, None]
        Tn[:, -1, :] = T_wall[:, None]
        Tn[:, :, 0] = T_wall[:, None]
        Tn[:, :, -1] = T_wall[:, None]

        T = Tn
        M_tot[step] = cond_mass.sum()

        # модель истощения пара:
        Mx = cond_mass.sum(axis=(1,2))
        dM = Mx - prev_Mx
        prev_Mx = Mx.copy()
        rho_loc -= dM / total_vol
        rho_loc = np.maximum(rho_loc, 1e-6)
        alpha = kappa_v/(rho_loc*cp_v)

    # пост-обработка
    cond_density = cond_mass / cell_vol
    avg_density  = (cond_density*cell_vol).sum(axis=(1,2)) / cell_vol.sum(axis=(1,2))
    M_x          = cond_mass.sum(axis=(1,2))

    return x, T, cond_density, avg_density, M_tot, M_x, T_wall, delta_film

# === 3) Запуск модели ===
x, T, cond_density, avg_density, M_tot, M_x, T_wall, delta_film = simulate(
    nx, ny, nz, T_in, T_wall_init, u_flow
)

# Обрезаем шумы
cond_density = np.clip(cond_density, 0, None)
M_x          = np.clip(M_x, 0, None)

figs = []
captions = []

# === 4) 2D-срезы в поперечном сечении ===
i_mid  = nx // 2
Si_mid = S0 + (S1 - S0) * (i_mid / (nx - 1))

# координаты центров ячеек
y = np.linspace(0, Si_mid, ny)
z = np.linspace(0, Si_mid, nz)

# срезы
T_slice   = T[i_mid]                   # shape (ny, nz)
rho_slice = cond_density[i_mid]        # shape (ny, nz)

y_p = np.concatenate([[y[0]], y, [y[-1]]])
z_p = np.concatenate([[z[0]], z, [z[-1]]])
T_p = np.pad(T_slice,   ((1,1),(1,1)), mode='edge')

rho2 = np.pad(rho_slice, ((1,1),(1,1)), mode='edge')

y2 = np.concatenate([[y[0]], y, [y[-1]]])
z2 = np.concatenate([[z[0]], z, [z[-1]]])
vmax_rho = np.percentile(rho2, 99)
levels_rho = np.linspace(0, vmax_rho, 256)
ticks_rho  = np.linspace(0, vmax_rho, 6)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5), constrained_layout=True)

# --- Контур температуры ---
levels_T = np.linspace(T_sat, T_in, 256)
im1 = ax1.contourf(
    y_p, z_p, T_p.T,
    levels=levels_T, cmap='inferno',
    vmin=T_sat, vmax=T_in
)
ax1.set_title(f"Температура, x={x[i_mid]:.2f} м")
ax1.set_xlabel('Y [м]'); ax1.set_ylabel('Z [м]')
ax1.set_xlim(0, Si_mid); ax1.set_ylim(0, Si_mid)
cbar1 = fig.colorbar(im1, ax=ax1,
    ticks=np.arange(int(T_sat), int(T_in)+1, 1),
    format='%d'
)
cbar1.set_label('T [K]')

# --- Контур плотности конденсата  ---
rho_p = np.pad(rho_slice, ((1,1),(1,1)), mode='edge')

y_p   = np.concatenate([[y[0]], y, [y[-1]]])
z_p   = np.concatenate([[z[0]], z, [z[-1]]])

vmax_rho = np.percentile(rho_p, 99)
if vmax_rho <= 0:
    ax2.text(0.5, 0.5, 'Нет конденсата',
             ha='center', va='center', fontsize=12)
else:
    levels_rho = np.linspace(0, vmax_rho, 256)
    im2 = ax2.contourf(
        y2, z2, rho2.T,
        levels=levels_rho, cmap='Blues',
        vmin=0, vmax=vmax_rho
    )
    ax2.set_title(f"Плотность конденсата, x={x[i_mid]:.2f} м")
    ax2.set_xlabel('Y [м]'); ax2.set_ylabel('Z [м]')
    ax2.set_xlim(0, Si_mid); ax2.set_ylim(0, Si_mid)
    ticks = np.linspace(0, vmax_rho, 6)
    cbar2 = fig.colorbar(im2, ax=ax2,
                         ticks=ticks, format='%.3f')
    cbar2.set_label('ρ_cond [кг/м³]')

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5), constrained_layout=True)
# … рисуем температуру и плотность …
figs.append(fig)
captions.append(
    "Рисунок 1. Поперечное распределение:\n"
    "- слева – поле температуры в среднем сечении x = Lx/2;\n"
    "- справа – плотность конденсата на гранях этого сечения."
)

plt.show()

# === 5) Продольный анализ ===
fig, (p1, p2, p3) = plt.subplots(1,3,figsize=(12,4), constrained_layout=True)
p1.plot(x, avg_density)
p1.set_title('Средняя плотность конденсата')
p1.set_xlabel('x [м]'); p1.set_ylabel('ρ_cond [кг/м³]')
p1.ticklabel_format(style='plain', axis='y')

p2.plot(x, M_x)
p2.set_title('Масса конденсата на сечении')
p2.set_xlabel('x [м]'); p2.set_ylabel('M_x [кг]')
p2.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

p3.plot(x, T_wall)
p3.set_title('Температура стенки вдоль канала')
p3.set_xlabel('x [м]'); p3.set_ylabel('T_wall [K]')
p3.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

fig, (p1,p2,p3) = plt.subplots(1,3, figsize=(12,4), constrained_layout=True)
# … рисуем avg_density, M_x, T_wall …
figs.append(fig)
captions.append(
    "Рисунок 2. Продольные профили вдоль x:\n"
    "• средняя плотность конденсата ρ_cond(x),\n"
    "• масса конденсата на сечении M_x(x),\n"
    "• температура стенки T_wall(x)."
)

plt.show()


# === 6) Тепловой поток на стенке Y=0 ===
q_wall = np.zeros(nx)
for i in range(nx):
    Si = S0 + (S1 - S0)*i/(nx-1)
    dy = Si/(ny-1)
    grad = (T_wall[i] - T[i, 1, :]) / dy
    q_wall[i] = kappa_v * grad.mean()

fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
ax.plot(x, q_wall, '-', lw=2)
ax.set_xticks(x[::10])
ax.set_xticklabels([f"{val:.2f}" for val in x[::10]])
ax.set_title('Тепловой поток на стенке Y=0 вдоль канала')
ax.set_xlabel('x [м]'); ax.set_ylabel('q$_w$ [Вт/м²]')
ax.ticklabel_format(style='plain', axis='y')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid(True)

fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
# … рисуем q_wall …
figs.append(fig)
captions.append("Рисунок 3. Тепловой поток q_w(x) на стенке Y=0 вдоль канала.")

plt.show()


# === 7) Временная эволюция суммарной массы конденсата ===
t = np.linspace(0, total_t, len(M_tot))
plt.figure(figsize=(6,4), constrained_layout=True)
plt.plot(t, M_tot, '-')
plt.title('Суммарная масса конденсата vs время')
plt.xlabel('t [с]'); plt.ylabel('M_tot [кг]')
plt.grid(True)

fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
# … рисуем M_tot(t) …
figs.append(fig)
captions.append("Рисунок 4. Суммарная масса конденсата M_tot(t) vs t.")

plt.show()


# === 8) Сходимость по сетке (шаг по nx=20) ===
ny0, nz0 = ny, nz
nx_vals = list(range(50, 151, 20))
M_end = []
for nx2 in nx_vals:
    # масштабируем поперечные размеры пропорционально
    ny2 = max(3, int(ny0 * nx2 / nx))
    nz2 = max(3, int(nz0 * nx2 / nx))
    _, _, _, _, M2, _, _, _ = simulate(nx2, ny2, nz2, T_in, T_wall_init, u_flow)
    M_end.append(M2[-1])

plt.figure(figsize=(6,4), constrained_layout=True)
plt.plot(nx_vals, M_end, marker='o', linewidth=2)
plt.title('Сходимость массы конденсата по сетке (шаг=20)')
plt.xlabel('nx')
plt.ylabel('M_tot [кг]')
plt.grid(True)

fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
# … рисуем M_end vs nx …
figs.append(fig)
captions.append("Рисунок 5. Сходимость массы конденсата по размерности сетки nx.")

plt.show()

# === 9) Коэффициент теплоотдачи вдоль канала ===
fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
h = np.zeros(nx)
for i in range(nx):
    Si    = S0 + (S1 - S0)*i/(nx-1)
    dy    = Si/(ny-1)
    Rgas  = dy / kappa_v
    Rfilm = delta_film[i] / kappa_w
    h[i]  = 1.0/(Rgas + Rfilm)

ax.plot(x, h, '-')
ax.set_title('Коэффициент теплоотдачи вдоль канала')
ax.set_xlabel('x [м]'); ax.set_ylabel('h [Вт/(м²·K)]')
ax.ticklabel_format(style='plain', axis='y')
ax.grid(True)

fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
# … рисуем столбчатую диаграмму финальной массы …
figs.append(fig)
captions.append("Рисунок 6. Влияние параметров (T_in, T_wall_init, u_flow) на финальную массу конденсата.")

plt.show()

# === 10) Сравнение конечной массы конденсата в изначальном и модифицированных сценариях (наглядность влияние параметров) ===
scenarios = [
    ('Base',      T_in,        T_wall_init, u_flow),
    ('T_in+10',   T_in+10,     T_wall_init, u_flow),
    ('T_wall+10', T_in,        T_wall_init+10, u_flow),
    ('u_flow×2',  T_in,        T_wall_init,    u_flow*2),
]
names, vals = [], []
for name, Tin, Tw, uf in scenarios:
    _,_,_,_,M2,_,_,_ = simulate(nx, ny, nz, Tin, Tw, uf)
    names.append(name); vals.append(M2[-1])

plt.figure(figsize=(6,4), constrained_layout=True)
plt.bar(names, vals)
plt.title('Конечная масса конденсата в сценариях')
plt.ylabel('M_tot [кг]')
plt.xticks(rotation=20)

fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
# … рисуем h(x) …
figs.append(fig)
captions.append("Рисунок 7. Коэффициент теплоотдачи h(x) вдоль канала.")

plt.show()


# === 11) Создание PDF-отчета ===
with PdfPages('report.pdf') as pdf:

    # — Страница 1: таблица вводных параметров —
    fig_params = plt.figure(figsize=(8.27,11.69))  # формат A4 в портрете
    plt.axis('off')
    table_data = [
        ["Обозначение", "Значение",     "Единица",         "Описание"],
        ["nx",          f"{nx}",        "-",               "число ячеек вдоль X"],
        ["ny",          f"{ny}",        "-",               "число ячеек вдоль Y"],
        ["nz",          f"{nz}",        "-",               "число ячеек вдоль Z"],
        ["Lx",          f"{Lx:.3f}",    "м",               "длина канала"],
        ["S0, S1",      f"{S0:.3f}, {S1:.3f}", "м",      "сторона квадрата в начале/конце"],
        ["T_in",        f"{T_in:.2f}",  "K",               "температура пара на входе"],
        ["T_sat",       f"{T_sat:.2f}", "K",               "температура насыщения"],
        ["T_wall_init", f"{T_wall_init:.2f}", "K",         "начальная температура стенок"],
        ["u_flow",      f"{u_flow:.3f}", "м/с",            "скорость потока на входе"],
        ["dt",          f"{dt:.3f}",    "с",               "шаг по времени"],
        ["total_t",     f"{total_t:.1f}", "с",             "время моделирования"],
        ["ρ_v0",        f"{rho_v0:.3f}", "кг/м³",          "плотность пара"],
        ["c_p_v",       f"{cp_v:.1f}",   "Дж/(кг·K)",      "теплоёмкость пара"],
        ["κ_v",         f"{kappa_v:.3f}", "Вт/(м·K)",       "теплопроводность пара"],
        ["λ",           f"{latent_h:.2e}", "Дж/кг",         "скрытая теплота конденсации"],
        ["ρ_w",         f"{rho_w:.1f}",   "кг/м³",         "плотность стенки"],
        ["c_p_w",       f"{cp_w:.1f}",    "Дж/(кг·K)",     "теплоёмкость стенки"],
        ["δ_w",         f"{th_w:.3f}",    "м",             "толщина стенки"],
        ["κ_w",         f"{kappa_w:.3f}", "Вт/(м·K)",      "теплопроводность плёнки воды"],
        ["ρ_water",     f"{rho_water:.1f}","кг/м³",         "плотность воды"],
    ]
    tbl = plt.table(cellText=table_data, loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2)
    pdf.savefig(fig_params)
    plt.close(fig_params)

    # — Страница 2: формулы (LaTeX) и пояснения —
    fig_form = plt.figure(figsize=(8.27,11.69))
    plt.axis('off')
    plt.text(0.5, 0.75,
             r"$T_{i}^{n+1} = T_{i}^{n} + \Delta t \Bigl(\alpha_i\,\frac{T_{i+1}^n - 2\,T_i^n + T_{i-1}^n}{\Delta x^2} - u_i\,\frac{T_i^n - T_{i-1}^n}{\Delta x}\Bigr)$",
             ha='center', va='center', fontsize=14)
    plt.text(0.5, 0.65,
             "— явная конечно-разностная схема для диффузии + конвекции по X\n"
             r"где\quad $\alpha_i = \frac{\kappa_v}{\rho_i\,c_{p,v}}$, \quad $\Delta x, \Delta t$ — шаги по пространству и времени",
             ha='center', va='top', fontsize=10)
    plt.text(0.5, 0.45,
             r"$\dot m = \frac{\rho_{v0}\,c_{p,v}}{\lambda}\,(T_{\rm sat} - T)\,,\quad T := T_{\rm sat}$",
             ha='center', va='center', fontsize=14)
    plt.text(0.5, 0.35,
             "— модель пленочной конденсации: масса конденсата в ячейке\n"
             r"$\Delta m = \dot m\,\Delta t \,V_{\rm cell}$",
             ha='center', va='top', fontsize=10)
    pdf.savefig(fig_form)
    plt.close(fig_form)

    # — Далее: каждый график + его пояснение —
    for fig, caption in zip(figs, captions):
        pdf.savefig(fig)
        plt.close(fig)

        fig_txt = plt.figure(figsize=(8.27,11.69))
        plt.axis('off')
        plt.text(0.01, 0.99, caption, va='top', wrap=True, fontsize=10)
        pdf.savefig(fig_txt)
        plt.close(fig_txt)
