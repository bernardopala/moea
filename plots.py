import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, ConnectionStyle, Rectangle
from scipy.interpolate import make_interp_spline
from jmetal.core.quality_indicator import HyperVolume, NormalizedHyperVolume, InvertedGenerationalDistance

def draw_pareto_front():
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(right=1.1, wspace=0.5)
    ### przestrzeń zmiennych decyzyjnych
    ax[0].set_xlabel('$x_1$')
    ax[0].set_ylabel('$x_2$')
    ax[0].set_xlim(0, 1.0)
    ax[0].set_ylim(0, 1.0)
    ax[0].set_title('Przestrzeń zmiennych decyzyjnych')
    # ax[0].legend(loc='lower left')
    ax[0].grid(True, linestyle='--', alpha=0.4)

    d_n1 = np.array([0.2, 0.6])
    d_n2 = np.array([0.3, 0.8])
    d_n3 = np.array([0.6, 0.7])

    d_d1 = np.array([0.8, 0.55])
    d_d2 = np.array([0.5, 0.5])
    d_d3 = np.array([0.3, 0.3])

    nondominated_points = np.array([d_n1, d_n2, d_n3])
    dominated_points = np.array([d_d1, d_d2, d_d3])
    nondominated_color = 'green'
    dominated_color = 'orange'

    ax[0].scatter(dominated_points[:, 0], dominated_points[:, 1], color=dominated_color, s=80, label='Rozwiązania zdominowane')
    ax[0].scatter(nondominated_points[:, 0], nondominated_points[:, 1], color=nondominated_color, s=80, label='Rozwiązania niezdominowane')

    ### przestrzeń celów
    ax[1].set_xlabel('$y_1$')
    ax[1].set_ylabel('$y_2$')
    ax[1].set_xlim(0, 1.0)
    ax[1].set_ylim(0, 1.0)
    ax[1].set_title('Przestrzeń celów')
    # ax[1].legend(loc='lower left')
    ax[1].grid(True, linestyle='--', alpha=0.4)

    o_n1 = np.array([0.7, 0.25])
    o_n2 = np.array([0.4, 0.5])
    o_n3 = np.array([0.2, 0.8])

    o_d1 = np.array([0.3, 0.9])
    o_d2 = np.array([0.5, 0.7])
    o_d3 = np.array([0.8, 0.6])

    nondominated_points = np.array([o_n1, o_n2, o_n3])
    dominated_points = np.array([o_d1, o_d2, o_d3])
    nondominated_color = 'green'
    dominated_color = 'orange'

    ax[1].scatter(dominated_points[:, 0], dominated_points[:, 1], color=dominated_color, s=80, label='Rozwiązania zdominowane')
    ax[1].scatter(nondominated_points[:, 0], nondominated_points[:, 1], color=nondominated_color, s=80, label='Rozwiązania niezdominowane')

    # --- Oznaczenia ---
    for i, (p, color, label) in enumerate(zip(np.vstack([nondominated_points, dominated_points]),
                                       [nondominated_color, nondominated_color, nondominated_color, dominated_color, dominated_color, dominated_color],
                                       ['0', '0', '0', '1', '2', '1'])):
        ax[1].text(p[0]+0.02, p[1]+0.03, f"{label}", color=color, fontsize=10)


    # aproksymacja frontu splajnem
    front_x = np.array([0.125,0.2, 0.4, 0.7, 0.9])
    front_y = np.array([0.95, 0.8, 0.5, 0.25, 0.15])
    front_new = np.linspace(front_x.min(), front_x.max(), 300)

    spl = make_interp_spline(front_x, front_y, k=2)
    power_smooth = spl(front_new)
    plt.plot(front_new, power_smooth, nondominated_color)

    # linie rzutujące na oś x i y
    for point in nondominated_points:
        onto_x_point = np.array([point[0], 0.0])
        onto_y_point = np.array([0.0, point[1]])

        x_values = [[onto_x_point[0], point[0]], [point[0], onto_y_point[0]]]
        y_values = [[onto_x_point[1], point[1]], [point[1], onto_y_point[1]]]
        plt.plot(x_values, y_values, nondominated_color, linestyle='dashed', linewidth=0.8)

    for point in dominated_points:
        onto_x_point = np.array([point[0], 0.0])
        onto_y_point = np.array([0.0, point[1]])

        x_values = [[onto_x_point[0], point[0]], [point[0], onto_y_point[0]]]
        y_values = [[onto_x_point[1], point[1]], [point[1], onto_y_point[1]]]
        plt.plot(x_values, y_values, dominated_color, linestyle='dashed', linewidth=0.8)

    ### linie łączące punkty
    n1_con = ConnectionPatch(xyA=o_n1, xyB=d_n1, coordsA="data", coordsB="data",
                          axesA=ax[1], axesB=ax[0], color="darkgray", linestyle='dashed',
                          connectionstyle=ConnectionStyle.Arc3(rad=-0.2))
    n2_con = ConnectionPatch(xyA=o_n2, xyB=d_n2, coordsA="data", coordsB="data",
                          axesA=ax[1], axesB=ax[0], color="darkgray", linestyle='dashed',
                          connectionstyle=ConnectionStyle.Arc3(rad=0.2))
    n3_con = ConnectionPatch(xyA=o_n3, xyB=d_n3, coordsA="data", coordsB="data",
                          axesA=ax[1], axesB=ax[0], color="darkgray", linestyle='dashed',
                          connectionstyle=ConnectionStyle.Arc3(rad=0.2))
    ax[1].add_artist(n1_con)
    ax[1].add_artist(n2_con)
    ax[1].add_artist(n3_con)

    d1_con = ConnectionPatch(xyA=o_d1, xyB=d_d1, coordsA="data", coordsB="data",
                          axesA=ax[1], axesB=ax[0], color="darkgray", linestyle='dashed',
                          connectionstyle=ConnectionStyle.Arc3(rad=0.2))
    d2_con = ConnectionPatch(xyA=o_d2, xyB=d_d2, coordsA="data", coordsB="data",
                          axesA=ax[1], axesB=ax[0], color="darkgray", linestyle='dashed',
                          connectionstyle=ConnectionStyle.Arc3(rad=-0.2))
    d3_con = ConnectionPatch(xyA=o_d3, xyB=d_d3, coordsA="data", coordsB="data",
                          axesA=ax[1], axesB=ax[0], color="darkgray", linestyle='dashed',
                          connectionstyle=ConnectionStyle.Arc3(rad=-0.2))
    ax[1].add_artist(d1_con)
    ax[1].add_artist(d2_con)
    ax[1].add_artist(d3_con)

    from matplotlib.patches import Rectangle
    square = Rectangle(
        (0.55, 0.2),  # pozycja (x, y) w układzie współrzędnych figury (0–1)
        0.1, 0.65,    # szerokość i wysokość
        transform=fig.transFigure,
        color='darkgray',
        alpha=0.5
    )

    fig.patches.append(square)
    fig.text(0.6, 0.6, r'$\boldsymbol{y}=f(\boldsymbol{x})$', ha='center', va='center', fontsize=14)

    plt.show()


def draw_hypervolume():
    # Punkty Pareto (A, B, C)
    A = np.array([0.2, 0.8])
    B = np.array([0.4, 0.5])
    C = np.array([0.8, 0.2])
    pareto_points = np.array([A, B, C])

    # Punkt referencyjny
    W = np.array([1.0, 1.0])

    fig, ax = plt.subplots(figsize=(5, 5))

    # Rysujemy prostokąty zdominowane przez punkty Pareto
    for p in pareto_points:
        width = W[0] - p[0]
        height = W[1] - p[1]
        rect = Rectangle(p, width, height, facecolor='lightgray', alpha=0.6, edgecolor='none')
        ax.add_patch(rect)

    # Rysunek frontu Pareto
    ax.plot(pareto_points[:, 0], pareto_points[:, 1], 'o', color='green', label='Front Pareto')
    ax.plot(W[0], W[1], 'ko', label='Punkt referencyjny W')

    # Oznaczenia punktów
    labels = ['A', 'B', 'C']
    for p, label in zip(pareto_points, labels):
        ax.text(p[0] - 0.05, p[1] - 0.1, label, fontsize=12)
    ax.text(W[0] + 0.04, W[1], 'W', fontsize=12)

    # Ustawienia osi
    ax.set_xlabel('$y_1$')
    ax.set_ylabel('$y_2$')
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.3)
    ax.set_title('Hiperobjętość (HV)')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.4)

    # aproksymacja frontu splajnem
    front_x = np.array([0.1, 0.2, 0.4, 0.8, 0.9])
    front_y = np.array([1.0, 0.8, 0.5, 0.2, 0.15])
    front_new = np.linspace(front_x.min(), front_x.max(), 300)

    spl = make_interp_spline(front_x, front_y, k=2)
    power_smooth = spl(front_new)
    plt.plot(front_new, power_smooth, 'green')

    plt.show()

def draw_selected_generations():
    ref_front = np.loadtxt("resources/reference_fronts/Fonseca.pf")
    ref_x = ref_front[:, 0]
    ref_y = ref_front[:, 1]

    hv_ref_point = np.max(ref_front, axis=0) + 0.1
    hv = HyperVolume(hv_ref_point)
    igd = InvertedGenerationalDistance(ref_front)

    fun1 = np.loadtxt("results/selected_generations/FUN.NSGAII.Fonseca.1")
    fun1_x = fun1[:, 0]
    fun1_y = fun1[:, 1]
    fun1_hv = hv.compute(fun1)
    fun1_igd = igd.compute(fun1)

    fun5 = np.loadtxt("results/selected_generations/FUN.NSGAII.Fonseca.5")
    fun5_x = fun5[:, 0]
    fun5_y = fun5[:, 1]
    fun5_hv = hv.compute(fun5)
    fun5_igd = igd.compute(fun5)

    fun10 = np.loadtxt("results/selected_generations/FUN.NSGAII.Fonseca.10")
    fun10_x = fun10[:, 0]
    fun10_y = fun10[:, 1]
    fun10_hv = hv.compute(fun10)
    fun10_igd = igd.compute(fun10)

    fun50 = np.loadtxt("results/selected_generations/FUN.NSGAII.Fonseca.50")
    fun50_x = fun50[:, 0]
    fun50_y = fun50[:, 1]
    fun50_hv = hv.compute(fun50)
    fun50_igd = igd.compute(fun50)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), constrained_layout=True)
    bbox_props = dict(boxstyle='square', facecolor='w', alpha=1)
    ref_s = 2
    fun_s = 20
    ndigits = 4
    legend_h = 0.05
    legend_v = 0.2
    fontsize = 10
    # fig.suptitle("Pareto front approximation (NSGAII-Fonseca)", fontsize="x-large")
    plt.subplot(221)  # (rows-columns-index)
    plt.xlabel("$f_1{(x)}$")
    plt.ylabel("$f_2{(x)}$")
    plt.grid(True)
    plt.scatter(ref_x, ref_y, marker="o", facecolors='blue', s=ref_s)
    plt.scatter(fun1_x, fun1_y, marker="o", facecolors='red', s=fun_s)
    plt.title(f'a) Populacja początkowa (HV = {round(fun1_hv, ndigits)}, IGD = {round(fun1_igd, ndigits)})',
              fontsize=fontsize)
    # plt.text(legend_h, legend_v, f'nHV = {round(fun1_nhv, ndigits)}\nIGD = {round(fun1_igd, ndigits)}', ha='left', va='center', size=12, bbox=bbox_props)

    plt.subplot(222)
    plt.xlabel("$f_1{(x)}$")
    plt.ylabel("$f_2{(x)}$")
    plt.grid(True)
    plt.scatter(ref_x, ref_y, marker="o", facecolors='blue', s=ref_s)
    plt.scatter(fun5_x, fun5_y, marker="o", facecolors='red', s=fun_s)
    plt.title(f'b) 5. generacja populacji (HV = {round(fun5_hv, ndigits)}, IGD = {round(fun5_igd, ndigits)})',
              fontsize=fontsize)
    # plt.text(legend_h, legend_v, f'nHV = {round(fun5_nhv, ndigits)}\nIGD = {round(fun5_igd, ndigits)}', ha='left', va='center', size=12, bbox=bbox_props)

    plt.subplot(223)
    plt.xlabel("$f_1{(x)}$")
    plt.ylabel("$f_2{(x)}$")
    plt.grid(True)
    plt.scatter(ref_x, ref_y, marker="o", facecolors='blue', s=ref_s)
    plt.scatter(fun10_x, fun10_y, marker="o", facecolors='red', s=fun_s)
    plt.title(f'c) 10. generacja populacji (HV = {round(fun10_hv, ndigits)}, IGD = {round(fun10_igd, ndigits)})',
              fontsize=fontsize)
    # plt.text(legend_h, legend_v, f'nHV = {round(fun10_nhv, ndigits)}\nIGD = {round(fun10_igd, ndigits)}', ha='left', va='center', size=12, bbox=bbox_props)

    plt.subplot(224)
    plt.xlabel("$f_1{(x)}$")
    plt.ylabel("$f_2{(x)}$")
    plt.grid(True)
    plt.scatter(ref_x, ref_y, marker="o", facecolors='blue', s=ref_s)
    plt.scatter(fun50_x, fun50_y, marker="o", facecolors='red', s=fun_s)
    plt.title(f'd) 50. generacja populacji (HV = {round(fun50_hv, ndigits)}, IGD = {round(fun50_igd, ndigits)})',
              fontsize=fontsize)
    # plt.text(legend_h, legend_v, f'nHV = {round(fun50_nhv, ndigits)}\nIGD = {round(fun50_igd, ndigits)}', ha='left', va='center', size=12, bbox=bbox_props)

    plt.show()

def generate_plots_2d(problem_name, iteration):
    ref_front = np.loadtxt(f"resources/reference_fronts/{problem_name}.pf")
    ref_x = ref_front[:, 0]
    ref_y = ref_front[:, 1]

    hv_ref_point = np.max(ref_front, axis=0) + 0.1
    hv = HyperVolume(hv_ref_point)
    igd = InvertedGenerationalDistance(ref_front)

    fun1 = np.loadtxt(f"results/comparative_analysis/FUN.NSGAII.{problem_name}.{iteration}").reshape(-1, 2)
    fun1_x = fun1[:, 0]
    fun1_y = fun1[:, 1]
    fun1_hv = hv.compute(fun1)
    fun1_igd = igd.compute(fun1)

    fun2 = np.loadtxt(f"results/comparative_analysis/FUN.SPEA2.{problem_name}.{iteration}").reshape(-1, 2)
    fun2_x = fun2[:, 0]
    fun2_y = fun2[:, 1]
    fun2_hv = hv.compute(fun2)
    fun2_igd = igd.compute(fun2)

    fun3 = np.loadtxt(f"results/comparative_analysis/FUN.MOEAD.{problem_name}.{iteration}").reshape(-1, 2)
    fun3_x = fun3[:, 0]
    fun3_y = fun3[:, 1]
    fun3_hv = hv.compute(fun3)
    fun3_igd = igd.compute(fun3)

    fun4 = np.loadtxt(f"results/comparative_analysis/FUN.Epsilon-IBEA.{problem_name}.{iteration}").reshape(-1, 2)
    fun4_x = fun4[:, 0]
    fun4_y = fun4[:, 1]
    fun4_hv = hv.compute(fun4)
    fun4_igd = igd.compute(fun4)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), constrained_layout=True)
    bbox_props = dict(boxstyle='square', facecolor='w', alpha=1)
    ref_s = 2
    fun_s = 20
    ndigits = 4
    legend_h = -30
    legend_v = -30
    alpha = 0.7
    fontsize = 12

    plt.subplot(221) # (rows-columns-index)
    plt.xlabel("$f_1{(x)}$")
    plt.ylabel("$f_2{(x)}$")
    plt.grid(True)
    plt.scatter(ref_x, ref_y, marker="o", facecolors='blue', s=ref_s)
    plt.scatter(fun1_x, fun1_y, marker="o", facecolors='red', s=fun_s, alpha=alpha)
    plt.title(f'a) NSGAII (HV = {round(fun1_hv, ndigits)}, IGD = {round(fun1_igd, ndigits)})', fontsize=fontsize)

    plt.subplot(222)
    plt.xlabel("$f_1{(x)}$")
    plt.ylabel("$f_2{(x)}$")
    plt.grid(True)
    plt.scatter(ref_x, ref_y, marker="o", facecolors='blue', s=ref_s)
    plt.scatter(fun2_x, fun2_y, marker="o", facecolors='red', s=fun_s, alpha=alpha)
    plt.title(f'b) SPEA2 (HV = {round(fun2_hv, ndigits)}, IGD = {round(fun2_igd, ndigits)})', fontsize=fontsize)

    plt.subplot(223)
    plt.xlabel("$f_1{(x)}$")
    plt.ylabel("$f_2{(x)}$")
    plt.grid(True)
    plt.scatter(ref_x, ref_y, marker="o", facecolors='blue', s=ref_s)
    plt.scatter(fun3_x, fun3_y, marker="o", facecolors='red', s=fun_s, alpha=alpha)
    plt.title(f'c) MOEAD (HV = {round(fun3_hv, ndigits)}, IGD = {round(fun3_igd, ndigits)})', fontsize=fontsize)

    plt.subplot(224)
    plt.xlabel("$f_1{(x)}$")
    plt.ylabel("$f_2{(x)}$")
    plt.grid(True)
    plt.scatter(ref_x, ref_y, marker="o", facecolors='blue', s=ref_s)
    plt.scatter(fun4_x, fun4_y, marker="o", facecolors='red', s=fun_s, alpha=alpha)
    plt.title(f'd) Epsilon-IBEA (HV = {round(fun4_hv, ndigits)}, IGD = {round(fun4_igd, ndigits)})', fontsize=fontsize)

    plt.show()


def generate_plots_3d(problem_name, iteration):
    ref_front = np.loadtxt(f"resources/reference_fronts/{problem_name}.pf")
    ref_x = ref_front[:, 0]
    ref_y = ref_front[:, 1]
    ref_z = ref_front[:, 2]

    hv_ref_point = np.max(ref_front, axis=0) + 0.1
    hv = HyperVolume(hv_ref_point)
    igd = InvertedGenerationalDistance(ref_front)

    fun1 = np.loadtxt(f"results/comparative_analysis/FUN.NSGAII.{problem_name}.{iteration}").reshape(-1, 3)
    fun1_x = fun1[:, 0]
    fun1_y = fun1[:, 1]
    fun1_z = fun1[:, 2]
    fun1_hv = hv.compute(fun1)
    fun1_igd = igd.compute(fun1)

    fun2 = np.loadtxt(f"results/comparative_analysis/FUN.SPEA2.{problem_name}.{iteration}").reshape(-1, 3)
    fun2_x = fun2[:, 0]
    fun2_y = fun2[:, 1]
    fun2_z = fun2[:, 2]
    fun2_hv = hv.compute(fun2)
    fun2_igd = igd.compute(fun2)

    fun3 = np.loadtxt(f"results/comparative_analysis/FUN.MOEAD.{problem_name}.{iteration}").reshape(-1, 3)
    fun3_x = fun3[:, 0]
    fun3_y = fun3[:, 1]
    fun3_z = fun3[:, 2]
    fun3_hv = hv.compute(fun3)
    fun3_igd = igd.compute(fun3)

    fun4 = np.loadtxt(f"results/comparative_analysis/FUN.Epsilon-IBEA.{problem_name}.{iteration}").reshape(-1, 3)
    fun4_x = fun4[:, 0]
    fun4_y = fun4[:, 1]
    fun4_z = fun4[:, 2]
    fun4_hv = hv.compute(fun4)
    fun4_igd = igd.compute(fun4)

    fig = plt.figure(figsize=(12, 10))
    elev = 30
    azim = 45
    roll = 0
    ndigits = 4
    fontsize = 12
    zoom = 0.9
    alpha = 0.7
    ref_s = 2
    fun_s = 20

    ax = fig.add_subplot(221, projection='3d')
    ax.scatter3D(ref_x, ref_y, ref_z, marker="o", facecolors='blue', s=ref_s, alpha=alpha)
    ax.scatter3D(fun1_x, fun1_y, fun1_z, marker="o", facecolors='red', s=fun_s)
    ax.set_box_aspect(None, zoom=zoom)
    ax.set_xlabel('$f_1{(x)}$')
    ax.set_ylabel('$f_2{(x)}$')
    ax.set_zlabel('$f_3{(x)}$')
    ax.view_init(elev, azim, roll)
    ax.set_title(f'a) NSGAII (HV = {round(fun1_hv, ndigits)}, IGD = {round(fun1_igd, ndigits)})', fontsize=fontsize)

    ax = fig.add_subplot(222, projection='3d')
    ax.scatter3D(ref_x, ref_y, ref_z, marker="o", facecolors='blue', s=ref_s, alpha=alpha)
    ax.scatter3D(fun2_x, fun2_y, fun2_z, marker="o", facecolors='red', s=fun_s)
    ax.set_box_aspect(None, zoom=zoom)
    ax.set_xlabel('$f_1{(x)}$')
    ax.set_ylabel('$f_2{(x)}$')
    ax.set_zlabel('$f_3{(x)}$')
    ax.view_init(elev, azim, roll)
    ax.set_title(f'b) SPEA2 (HV = {round(fun2_hv, ndigits)}, IGD = {round(fun2_igd, ndigits)})', fontsize=fontsize)

    ax = fig.add_subplot(223, projection='3d')
    ax.scatter3D(ref_x, ref_y, ref_z, marker="o", facecolors='blue', s=ref_s, alpha=alpha)
    ax.scatter3D(fun3_x, fun3_y, fun3_z, marker="o", facecolors='red', s=fun_s)
    ax.set_box_aspect(None, zoom=zoom)
    ax.set_xlabel('$f_1{(x)}$')
    ax.set_ylabel('$f_2{(x)}$')
    ax.set_zlabel('$f_3{(x)}$')
    ax.view_init(elev, azim, roll)
    ax.set_title(f'c) MOEAD (HV = {round(fun3_hv, ndigits)}, IGD = {round(fun3_igd, ndigits)})', fontsize=fontsize)

    ax = fig.add_subplot(224, projection='3d')
    ax.scatter3D(ref_x, ref_y, ref_z, marker="o", facecolors='blue', s=ref_s, alpha=alpha)
    ax.scatter3D(fun4_x, fun4_y, fun4_z, marker="o", facecolors='red', s=fun_s)
    ax.set_box_aspect(None, zoom=zoom)
    ax.set_xlabel('$f_1{(x)}$')
    ax.set_ylabel('$f_2{(x)}$')
    ax.set_zlabel('$f_3{(x)}$')
    ax.view_init(elev, azim, roll)
    ax.set_title(f'd) Epsilon-IBEA (HV = {round(fun4_hv, ndigits)}, IGD = {round(fun4_igd, ndigits)})',
                 fontsize=fontsize)

    plt.show()