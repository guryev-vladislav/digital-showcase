import numpy as np
import matplotlib.pyplot as plt


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def stretched_linspace(n, alpha, bias_toward_zero=True):
    """
    Создает геометрически сгущённую линейную последовательность от 0 до 1.
    Если bias_toward_zero=True, сгущение к 0; иначе - к 1.
    """
    if bias_toward_zero:
        return np.linspace(0, 1, n) ** alpha
    else:
        return 1 - (np.linspace(0, 1, n)[::-1] ** alpha)


def bilinear_interpolation(xi_coords, eta_coords, P1, P2, P3, P4):
    """
    Выполняет билинейную интерполяцию для произвольного четырехугольника.
    P1 (левый нижний), P2 (правый нижний), P3 (правый верхний), P4 (левый верхний).
    """
    return (
            np.outer(1 - xi_coords, 1 - eta_coords)[:, :, None] * P1 +
            np.outer(xi_coords, 1 - eta_coords)[:, :, None] * P2 +
            np.outer(xi_coords, eta_coords)[:, :, None] * P3 +
            np.outer(1 - xi_coords, eta_coords)[:, :, None] * P4
    )


def run_preprocessor(LX, DX, LY, DY, NX, NY, ALPHA_STRETCH):

    P1 = np.array([LX, DX])
    P2 = np.array([LX + LY, DX])
    P3 = np.array([LX + LY, DX + DY])
    P4 = np.array([LX, DX + DY])
    P_corners_initial = {'P1': P1, 'P2': P2, 'P3': P3, 'P4': P4}

    print(f"Вершины прямоугольника:")
    for key, val in P_corners_initial.items():
        print(f"{key}: {val}")
    print("\n")

    xi = stretched_linspace(NX + 1, alpha=ALPHA_STRETCH, bias_toward_zero=True)
    eta = np.linspace(0, 1, NY + 1)

    initial_grid = bilinear_interpolation(xi, eta, P1, P2, P3, P4)

    x_coords_initial = initial_grid[:, :, 0].flatten()
    y_coords_initial = initial_grid[:, :, 1].flatten()

    print(f"Общее количество узлов: {len(x_coords_initial)}\n")
    return x_coords_initial, y_coords_initial, initial_grid, P_corners_initial, xi, eta


def run_processor(initial_grid, x_coords_initial, y_coords_initial,
                  P_corners_final, FIXED_BOUNDARY,
                  LX, DX, LY, DY, NX, NY, xi, eta):

    # Вычисляем конечную сетку с помощью билинейной интерполяции по конечным вершинам
    final_grid = bilinear_interpolation(xi, eta,
                                        P_corners_final['P1_final'],
                                        P_corners_final['P2_final'],
                                        P_corners_final['P3_final'],
                                        P_corners_final['P4_final'])

    # Вычисляем перемещения как разность конечных и начальных координат
    u_x_step = (final_grid[:, :, 0] - initial_grid[:, :, 0]).flatten()
    u_y_step = (final_grid[:, :, 1] - initial_grid[:, :, 1]).flatten()

    # Применяем граничные условия (закрепление)
    if FIXED_BOUNDARY == 'bottom':
        fixed_indices = np.where(np.isclose(y_coords_initial, DX))[0]
    elif FIXED_BOUNDARY == 'top':
        fixed_indices = np.where(np.isclose(y_coords_initial, DX + DY))[0]
    elif FIXED_BOUNDARY == 'left':
        fixed_indices = np.where(np.isclose(x_coords_initial, LX))[0]
    elif FIXED_BOUNDARY == 'right':
        fixed_indices = np.where(np.isclose(x_coords_initial, LX + LY))[0]
    else:
        print(f"Предупреждение: Неизвестное значение FIXED_BOUNDARY: {FIXED_BOUNDARY}. Закрепление не применено.")
        fixed_indices = np.array([], dtype=int)

    # Обнуляем перемещения для закрепленных узлов
    u_x_step[fixed_indices] = 0.0
    u_y_step[fixed_indices] = 0.0

    print(f"Заданные параметры деформации:") # Отладочный вывод
    print(f"  Конечные вершины: {P_corners_final}") # Отладочный вывод
    print(f"  Закрепленная граница: {FIXED_BOUNDARY}") # Отладочный вывод
    print(f"  Количество закрепленных узлов: {len(fixed_indices)}\n")

    return u_x_step, u_y_step


def run_postprocessor_and_time_loop(initial_grid,
                                    u_x_step, u_y_step,
                                    P_corners_initial, NX, NY,
                                    NUM_TIME_STEPS):

    plt.figure(figsize=(10, 8))
    plt.title(f"Деформация конечно-элементной сетки за {NUM_TIME_STEPS} шага(ов)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)

    # Отображение исходной сетки (шаг 0)
    plt.plot(initial_grid[:, 0, 0], initial_grid[:, 0, 1], 'k--', alpha=0.5, label='Исходная сетка')
    for i in range(1, NY + 1):
        plt.plot(initial_grid[:, i, 0], initial_grid[:, i, 1], 'k--', alpha=0.5)

    plt.plot(initial_grid[0, :, 0], initial_grid[0, :, 1], 'k--', alpha=0.5)
    for j in range(1, NX + 1):
        plt.plot(initial_grid[j, :, 0], initial_grid[j, :, 1], 'k--', alpha=0.5)

    # Маркеры для вершин исходного прямоугольника
    plt.plot(P_corners_initial['P1'][0], P_corners_initial['P1'][1], 'ko', markersize=8, label='P1 (Исходная)')
    plt.plot(P_corners_initial['P2'][0], P_corners_initial['P2'][1], 'ko', markersize=8)
    plt.plot(P_corners_initial['P3'][0], P_corners_initial['P3'][1], 'ko', markersize=8)
    plt.plot(P_corners_initial['P4'][0], P_corners_initial['P4'][1], 'ko', markersize=8)

    current_grid = initial_grid.copy()

    u_step_grid = np.stack((u_x_step.reshape(NX + 1, NY + 1), u_y_step.reshape(NX + 1, NY + 1)), axis=-1)

    colors = ['r-', 'b-', 'g-', 'c-', 'm-', 'y-']
    for k in range(1, NUM_TIME_STEPS + 1):
        step_deformation = u_step_grid / NUM_TIME_STEPS
        current_grid = current_grid + step_deformation

        plt.plot(current_grid[:, 0, 0], current_grid[:, 0, 1], colors[k - 1], label=f'Сетка после шага {k}')
        for i in range(1, NY + 1):
            plt.plot(current_grid[:, i, 0], current_grid[:, i, 1], colors[k - 1])

        plt.plot(current_grid[0, :, 0], current_grid[0, :, 1], colors[k - 1])
        for j in range(1, NX + 1):
            plt.plot(current_grid[j, :, 0], current_grid[j, :, 1], colors[k - 1])

        # Новые координаты вершин для текущего шага.
        P1_new = current_grid[0, 0, :]
        plt.plot(P1_new[0], P1_new[1], colors[k - 1][0] + 'o', markersize=6, label=f'P1 (Шаг {k})')

        plt.plot(current_grid[NX, 0, 0], current_grid[NX, 0, 1], colors[k - 1][0] + 'o', markersize=6)  # P2
        plt.plot(current_grid[NX, NY, 0], current_grid[NX, NY, 1], colors[k - 1][0] + 'o', markersize=6)  # P3
        plt.plot(current_grid[0, NY, 0], current_grid[0, NY, 1], colors[k - 1][0] + 'o', markersize=6)  # P4

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()


def main():
    LX = 0.0  # X-координата левой границы
    DX = 0.0  # Y-координата нижней границы
    LY = 2.0  # Длина по X
    DY = 1.0  # Высота по Y

    NX = 4  # Количество элементов по X
    NY = 2  # Количество элементов по Y

    ALPHA_STRETCH = 3.0  # Параметр сгущения сетки

    NUM_TIME_STEPS = 2  # Количество шагов по времени

    P1_final = np.array([LX, DX])  # P1 остается на месте
    P2_final = np.array([LX + LY, DX])  # P2 остается на месте
    P3_final = np.array([LX + LY + 1.0, DX + DY])  # P3 смещается вправо
    P4_final = np.array([LX + 1.0, DX + DY])  # P4 смещается вправо (на ту же величину, что P3)

    P_corners_final = {'P1_final': P1_final, 'P2_final': P2_final,
                       'P3_final': P3_final, 'P4_final': P4_final}

    FIXED_BOUNDARY = 'bottom'

    x_coords_initial, y_coords_initial, initial_grid, P_corners_initial, xi, eta = run_preprocessor(
        LX, DX, LY, DY, NX, NY, ALPHA_STRETCH
    )

    u_x_step, u_y_step = run_processor(
        initial_grid, x_coords_initial, y_coords_initial,
        P_corners_final=P_corners_final,
        FIXED_BOUNDARY=FIXED_BOUNDARY,
        LX=LX, DX=DX, LY=LY, DY=DY, NX=NX, NY=NY, xi=xi, eta=eta
    )

    run_postprocessor_and_time_loop(
        initial_grid,
        u_x_step, u_y_step,
        P_corners_initial, NX, NY, NUM_TIME_STEPS
    )


if __name__ == "__main__":
    main()