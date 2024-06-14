import numpy as np
import matplotlib.pyplot as plt
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go



class PSB_project:

    def __init__(self, N: int = 4) -> None:
        # do zastanowienia: uogólnienie na dowolną liczbę N obiektów.
        self.N = N

        # macierz układu równań różniczkowych
        self.A = None

        # hiperparametry modelu
        self.a = None
        self.b = None
        self.alpha = None

        # warunki początkowe
        self.t0 = None
        self.x0 = None

        # hiperparametry metody Rungego-Kutty
        self.n_iter = None
        self.h = None

        # wskaźniki czy podano warunki początkowe oraz hiperparametry
        self.given_ics = False
        self.given_params = False

    def set_params(self, a: np.float32, b: np.float32, alpha: np.float32) -> None:
        """Funkcja umożliwiająca przestawienie hiperparametrów modelu

        Args:
            a (np.float32): parametr "a" odpowiadający w_i(1|0)
            b (np.float32): parametr "b" odpowiadający w_i(0|1)
            alpha (np.float32): parametr "α"
        """
        self.given_params = True
        self.a = a
        self.b = b
        self.alpha = alpha
        # print(f"Zmodyfikowano hiperparametry\n... Nowe wartości to a={self.a}, b={self.b}, alpha={self.alpha}.")

    def set_iconds(self, t0: np.float32, x0: np.array = None) -> None:
        """Funkcja umożliwiająca specyfikację warunków początkowych

        Args:
            t0 (np.float): czas początkowy
            x0 (np.array | None, optional): wektor w chwili t0. Jeśli = None, x0 jest losowany jednostajnie z [0, 1]. Wartość domyślna: None.
        """
        self.given_ics = True
        self.t0 = t0
        if x0 is None:
            x0 = np.random.rand(16)
        else:
            self.x0 = x0

    def f(self, t: np.float32, x: np.array) -> np.array:
        """Funkcja obliczająca pochodną wektora zmiennych w układzie równań dx/dt = Ax.

        Args:
            t (np.float32): czas
            x (np.array): wektor wartości w czasie

        Returns:
            np.array: wartość Ax, czyli pochodna: dx/dt
        """
        return self.A @ x

    def w(self, s: int) -> np.float32:
        """Funkcja pomocnicza służąca do obliczania elementów w macierzy A

        Args:
            s (int): 0 albo 1, oznacza stan elementu układu

        Raises:
            ValueError: gdy "s" nie jest równy 0 lub 1

        Returns:
            value (float): "a" albo "b", odpowiednie hiperparametry modelu
        """
        if s not in [0, 1]:
            raise ValueError("Argument 's' musi mieć wartość 0 lub 1.")
        return (self.a if s == 1 else self.b)

    def generate_A(self) -> None:
        """Funkcja generująca pełną macierz układu równań liniowych
        """
        A = np.zeros((16, 16))
        for i in range(16):
            # "i" jest indeksem zmiennej w modelu
            # możemy ją zdekodować na "s1", "s2", "s3" i "s4"
            # używając zapisu binarnego
            bin_i = bin(i).lstrip('0b').zfill(4)
            s1, s2, s3, s4 = (int(i) for i in [*bin_i])
            # print(s1, s2, s3, s4)

            # teraz uzupełnimy macierz układu równań różniczkowych liniowych
            # odzyskujemy indeksy wszystkich potrzebnych zmiennych
            x = s1 + 2 * s2 + 4 * s3 + 8 * s4  # ...to jest zmienna, której równanie uzupełniamy
            x_s1_bar = (1 - s1) + 2 * s2 + 4 * s3 + 8 * s4  # ... a te cztery są w to równanie zaangażowane
            x_s2_bar = s1 + 2 * (1 - s2) + 4 * s3 + 8 * s4  # ...
            x_s3_bar = s1 + 2 * s2 + 4 * (1 - s3) + 8 * s4  # ...
            x_s4_bar = s1 + 2 * s2 + 4 * s3 + 8 * (1 - s4)  # ...

            # wpisujemy odpowiednie wartości w odpowiednie pola
            A[x, x_s1_bar] = self.w(s1) * (1 + self.alpha * (s2))
            A[x, x_s2_bar] = self.w(s2) * (1 + self.alpha * (s1 + s3))
            A[x, x_s3_bar] = self.w(s3) * (1 + self.alpha * (s2 + s4))
            A[x, x_s4_bar] = self.w(s4) * (1 + self.alpha * (s3))
            A[x, x] = -(self.w(1 - s1) * (1 + self.alpha * (s2)) +
                        self.w(1 - s2) * (1 + self.alpha * (s1 + s3)) +
                        self.w(1 - s3) * (1 + self.alpha * (s2 + s4)) +
                        self.w(1 - s4) * (1 + self.alpha * (s3)))

        self.A = A
        # print(f"Pomyślnie zmodyfikowano macierz układu równań.")

    def Runge_Kutta(self, n_iter: np.int16 = 1000, h: np.float32 = 0.01) -> tuple[np.array]:

        if self.given_ics == False:
            raise ValueError("Przed wywołaniem należy podać warunki początkowe do metody 'iconds'.")

        self.n_iter = n_iter
        self.h = h

        # Runge-Kutta loop
        t = np.zeros(self.n_iter)
        x = np.zeros((self.n_iter, 16))

        t[0] = self.t0
        x[0, :] = self.x0
        for i in range(self.n_iter):
            k1 = self.h * self.f(t[i], x[i, :])
            k2 = self.h * self.f(t[i] + self.h / 2, x[i, :] + k1 / 2)
            k3 = self.h * self.f(t[i] + self.h / 2, x[i, :] + k2 / 2)
            k4 = self.h * self.f(t[i] + self.h, x[i, :] + k3)

            if i == self.n_iter - 1:
                break

            t[i + 1] = t[i] + self.h
            x[i + 1, :] = x[i, :] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

            if np.abs(np.sum(x[i + 1]) - 1) > 0.001:
                raise ValueError("Prawdopodobieństwa nie sumują się do 1. Należy zmniejszyć długość kroku metody R-K.")

        return t, x

    def plot_variables(self, t: np.array, x: np.array, horizon: np.float32 = 1., save: bool = False) -> plt.Figure:
        max_t = self.t0 + horizon

        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(16):
            bin_i = bin(i).lstrip('0b').zfill(4)
            s1, s2, s3, s4 = (int(i) for i in [*bin_i])

            ax.plot(t[t <= max_t],
                    x[t <= max_t, i],
                    lw=0.6,
                    label=f"P({s1}, {s2}, {s3}, {s4})")

            ax.scatter(t[t <= max_t],
                       x[t <= max_t, i],
                       s=1.5)

        ax.grid(which="major", visible=True)
        ax.grid(which="minor", alpha=0.5, linestyle=":")
        ax.minorticks_on()

        leg = ax.legend(loc="upper right", title="Zmienne")
        for line in leg.get_lines():
            line.set_linewidth(2.0)

        ax.set_title(f"$a=${self.a}, $b$={self.a}, $\\alpha$={self.alpha}")
        ax.set_xlabel("Czas")
        ax.set_ylabel("Wartość")
        ax.set_ylim(bottom=0, top=np.max(x) + 0.05)

        fig.suptitle(f"Zmienne w modelu Isinga", fontsize="x-large", fontweight="semibold")
        if save:
            plt.savefig(f"Plots_toTeX/VAR_alpha" + f"{self.alpha}_a" + f"{self.a}_b" + f"{self.b}.png")
        return fig

    def comparative_plot_variables(self,
                                   As: list[np.float32],
                                   Bs: list[np.float32],
                                   Alphas: list[np.float32],
                                   horizon: np.float32 = 1.,
                                   save: bool = False) -> plt.figure:
        max_t = self.t0 + horizon
        subplots_nb = len(As) * len(Bs) * len(Alphas)
        if subplots_nb == 1:
            self.set_params(As[0], Bs[0], Alphas[0])
            t, x = self.Runge_Kutta()
            self.plot_variables(t, x, save=save)
            return None

        # trzeba sprytnie ustawić grid i figsize - jak?
        if subplots_nb % 2 == 0:
            nrow = subplots_nb // 2
            ncol = 2
            fig, AX = plt.subplots(nrow, ncol, figsize=(16, 6 * nrow), layout="constrained")
        elif subplots_nb == 3:
            nrow = 3
            ncol = 1
            fig, AX = plt.subplots(nrow, ncol, figsize=(8, 6 * nrow), layout="constrained")
        else:
            raise ValueError(
                "Należy podać takie listy hiperparametrów 'As', 'Bs', 'Alphas', żeby iloczyn ich długości był liczbą parzystą lub był równy 3.")

        if nrow == 1 or ncol == 1:
            plot_nb = 0
        elif nrow > 1 and ncol > 1:
            plot_row = 0
            plot_col = 0
        else:
            raise ValueError("Niepoprawna siatka do 'plt.figure'.")

        for a in As:
            for b in Bs:
                for alpha in Alphas:
                    self.set_params(a, b, alpha)
                    self.generate_A()
                    t, x = self.Runge_Kutta()

                    if nrow == 1 or ncol == 1:
                        ax = AX[plot_nb]
                    else:
                        ax = AX[plot_row, plot_col]

                    for i in range(16):
                        bin_i = bin(i).lstrip('0b').zfill(4)
                        s1, s2, s3, s4 = (int(i) for i in [*bin_i])

                        ax.plot(t[t <= max_t],
                                x[t <= max_t, i],
                                lw=0.6,
                                label=f"P({s1}, {s2}, {s3}, {s4})")

                        ax.scatter(t[t <= max_t],
                                   x[t <= max_t, i],
                                   s=1.5)

                    ax.grid(which="major", visible=True)
                    ax.grid(which="minor", alpha=0.5, linestyle=":")
                    ax.minorticks_on()

                    leg = ax.legend(loc="upper right", title="Zmienne")
                    for line in leg.get_lines():
                        line.set_linewidth(2.0)

                    ax.set_title(f"$a=${self.a}, $b$={self.b}, $\\alpha$={self.alpha}")
                    ax.set_xlabel("Czas")
                    ax.set_ylabel("Wartość")
                    ax.set_ylim(bottom=0, top=np.max(x) + 0.05)

                    if nrow == 1 or ncol == 1:
                        plot_nb += 1
                    else:
                        if plot_row + 1 < nrow:
                            plot_row += 1
                        else:
                            plot_row = 0
                            plot_col += 1
        fig.suptitle(f"Dynamika zmiennych w modelu Isinga", fontsize="xx-large", fontweight="semibold")
        if save:
            plt.savefig(
                f"Plots_toTeX/comparativeVAR_alphafrom" + f"{Alphas[0]}to{Alphas[-1]}_afrom" + f"{As[0]}to{As[-1]}_bfrom" + f"{Bs[0]}to{Bs[-1]}.png")
        return fig

    ######################################################################################
    # Poniżej: sekcja poświęcona badaniu średniej
    ######################################################################################

    def expected_value(self, x: np.array, which: int) -> tuple[np.array]:

        temp_list = []
        for i in range(16):
            bin_i = bin(i).lstrip('0b').zfill(4)
            s1, s2, s3, s4 = (int(i) for i in [*bin_i])
            if which == 1 and s1 == 1:
                temp_list.append(i)
            if which == 2 and s2 == 1:
                temp_list.append(i)
            if which == 3 and s3 == 1:
                temp_list.append(i)
            if which == 4 and s4 == 1:
                temp_list.append(i)

        # temp_list zawiera numery wszystkich zmiennych, które chcemy zidentyfikować
        return np.sum(x[:, temp_list], axis=1)

    def plot_expectations(self,
                          t: np.array,
                          x: np.array,
                          which: list = [1, 2, 3, 4],
                          horizon: np.float32 = 1.,
                          save: bool = False) -> plt.figure:

        max_t = self.t0 + horizon
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in which:
            expval = self.expected_value(x, i)

            ax.plot(t[t <= max_t],
                    expval[t <= max_t],
                    lw=0.6,
                    label=f"$E_t[s_{i}]$")

            ax.scatter(t[t <= max_t],
                       expval[t <= max_t],
                       s=1.5)
            ax.grid(which="major", visible=True)
            ax.grid(which="minor", alpha=0.5, linestyle=":")
            ax.minorticks_on()

        leg = ax.legend(loc="upper right")
        for line in leg.get_lines():
            line.set_linewidth(2.0)
        ax.set_title(f"$a=${self.a}, $b$={self.b}, $\\alpha$={self.alpha}")
        ax.set_xlabel("Czas")
        ax.set_ylabel("Wartość")
        ax.set_ylim(bottom=0, top=np.max(expval) + 0.02)

        fig.suptitle(f"Średnie w modelu Isinga", fontsize="large", fontweight="semibold")
        if save:
            plt.savefig(f"Plots_toTeX/EXP_alpha" + f"{self.alpha}_a" + f"{self.a}_b" + f"{self.b}.png")
        return fig

    def comparative_plot_expectations(self,
                                      As: list[np.float32],
                                      Bs: list[np.float32],
                                      Alphas: list[np.float32],
                                      horizon: np.float32 = 1.,
                                      save: bool = False) -> plt.figure:
        max_t = self.t0 + horizon
        subplots_nb = len(As) * len(Bs) * len(Alphas)
        if subplots_nb == 1:
            self.set_params(As[0], Bs[0], Alphas[0])
            t, x = self.Runge_Kutta()
            self.plot_expectations(t, x, save=save)
            return None

        # trzeba sprytnie ustawić grid i figsize - jak?
        if subplots_nb % 2 == 0:
            nrow = subplots_nb // 2
            ncol = 2
            fig, AX = plt.subplots(nrow, ncol, figsize=(16, 6 * nrow), layout="constrained")
        elif subplots_nb == 3:
            nrow = 3
            ncol = 1
            fig, AX = plt.subplots(nrow, ncol, figsize=(8, 6 * nrow), layout="constrained")
        else:
            raise ValueError(
                "Należy podać takie listy hiperparametrów 'As', 'Bs', 'Alphas', żeby iloczyn ich długości był liczbą parzystą lub był równy 3.")

        if nrow == 1 or ncol == 1:
            plot_nb = 0
        elif nrow > 1 and ncol > 1:
            plot_row = 0
            plot_col = 0
        else:
            raise ValueError("Niepoprawna siatka do 'plt.figure'.")

        for a in As:
            for b in Bs:
                for alpha in Alphas:
                    self.set_params(a, b, alpha)
                    self.generate_A()
                    t, x = self.Runge_Kutta()

                    if nrow == 1 or ncol == 1:
                        ax = AX[plot_nb]
                    else:
                        ax = AX[plot_row, plot_col]

                    maxval = 0
                    for i in range(1, 5):
                        expval = self.expected_value(x, i)
                        maxval = (np.max(expval) if np.max(expval) > maxval else maxval)

                        ax.plot(t[t <= max_t],
                                expval[t <= max_t],
                                lw=0.6,
                                label=f"$E_t[s_{i}]$")

                        ax.scatter(t[t <= max_t],
                                   expval[t <= max_t],
                                   s=1.5)
                        ax.grid(which="major", visible=True)
                        ax.grid(which="minor", alpha=0.5, linestyle=":")
                        ax.minorticks_on()

                    leg = ax.legend(loc="upper right")
                    for line in leg.get_lines():
                        line.set_linewidth(2.0)
                    ax.set_title(f"$a=${self.a}, $b$={self.b}, $\\alpha$={self.alpha}")
                    ax.set_xlabel("Czas")
                    ax.set_ylabel("Wartość")
                    ax.set_ylim(bottom=0, top=maxval + 0.02)

                    if nrow == 1 or ncol == 1:
                        plot_nb += 1
                    else:
                        if plot_row + 1 < nrow:
                            plot_row += 1
                        else:
                            plot_row = 0
                            plot_col += 1
        fig.suptitle(f"Dynamika średnich w modelu Isinga", fontsize="xx-large", fontweight="semibold")
        if save:
            plt.savefig(
                f"Plots_toTeX/comparativeEXP_alphafrom" + f"{Alphas[0]}to{Alphas[-1]}_afrom" + f"{As[0]}to{As[-1]}_bfrom" + f"{Bs[0]}to{Bs[-1]}.png")
        return fig


# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    dcc.Graph(id='ising-plot'),
    html.Div([
        html.Label('Parameter a'),
        dcc.Slider(id='a-slider', min=0, max=5, step=0.1, value=1.0,
                   marks={i: str(i) for i in range(6)}),
    ]),
    html.Div([
        html.Label('Parameter b'),
        dcc.Slider(id='b-slider', min=0, max=5, step=0.1, value=2.0,
                   marks={i: str(i) for i in range(6)}),
    ]),
    html.Div([
        html.Label('Parameter alpha'),
        dcc.Slider(id='alpha-slider', min=0, max=5, step=0.1, value=0.5,
                   marks={i: str(i) for i in range(3)}),
    ])
])


# Callback to update the plot based on slider values
@app.callback(
    Output('ising-plot', 'figure'),
    [Input('a-slider', 'value'),
     Input('b-slider', 'value'),
     Input('alpha-slider', 'value')]
)
def update_plot(a, b, alpha):
    t, x = solve_model(a, b, alpha)
    max_t = 5

    fig = go.Figure()

    for i in range(16):
        bin_i = bin(i).lstrip('0b').zfill(4)
        s1, s2, s3, s4 = (int(i) for i in [*bin_i])

        fig.add_trace(go.Scatter(
            x=t[t <= max_t],
            y=x[t <= max_t, i],
            mode='lines',
            line=dict(width=1.5),  # Decreased line width
            name=f"P({s1}, {s2}, {s3}, {s4})"
        ))

    fig.update_layout(
        title=dict(
            text=f"Zmienne w modelu Isinga<br><sup>a={a}, b={b}, alpha={alpha}</sup>",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Czas",
        yaxis_title="Wartość",
        yaxis=dict(range=[0, np.max(x) + 0.05]),
        legend_title_text="Zmienne",
        legend=dict(
            x=1,
            y=1,
            title=dict(text="Zmienne"),
            bordercolor="Black",
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig
t0 = 0
U = np.random.rand(15)
Usorted = np.sort(U)
x0 = np.diff(np.hstack((0, Usorted, 1)))
assert np.sum(x0) == 1.
def solve_model(a,b,alpha):
    PSB = PSB_project()
    PSB.set_iconds(t0, x0)
    PSB.set_params(a, b, alpha)
    PSB.generate_A()
    t, x = PSB.Runge_Kutta()
    return t,x

app.run_server(debug=True)

