import random
import numpy as np
import matplotlib.pyplot as plt


def onemax(x):
    return sum(x)


def leading_ones(x):
    count = 0
    for bit in x:
        if bit == 0:
            break
        count += 1
    return count


def genetic_algorithm(D, fitness, pop_size=50, mutation=0.01,
                      elite_ratio=0.1, crossover=1.0,
                      selection="roulette", seed=None):
    if D < 2 or pop_size < 2:
        raise ValueError("D a veľkosť populácie musia byť aspoň 2.")
    if not (0 <= mutation <= 1 and 0 <= crossover <= 1
            and 0 <= elite_ratio < 1):
        raise ValueError("Neplatné pravdepodobnosti alebo pomer elitizmu.")
    if selection not in ("roulette", "rank"):
        raise ValueError("Selekcia musí byť roulette alebo rank.")

    rng = random.Random(seed)
    max_evals = 100 * D
    pop_size = min(pop_size, max_evals)
    population = [[rng.randint(0, 1) for _ in range(D)]
                  for _ in range(pop_size)]
    history = []
    best = 0

    def evaluate(individual):
        nonlocal best
        score = fitness(individual)
        best = max(best, score)
        history.append(best)
        return score

    scores = [evaluate(x) for x in population]
    elite_count = int(pop_size * elite_ratio)

    while len(history) < max_evals:
        order = sorted(range(pop_size), key=lambda i: scores[i], reverse=True)
        new_population = [population[i][:] for i in order[:elite_count]]
        new_scores = [scores[i] for i in order[:elite_count]]

        if selection == "roulette":
            weights = [s + 1 for s in scores]
        else:
            weights = [0.0] * pop_size
            ascending = sorted(range(pop_size), key=lambda i: scores[i])
            start = 0
            while start < pop_size:
                end = start + 1
                while end < pop_size and scores[ascending[end]] == scores[ascending[start]]:
                    end += 1
                for i in ascending[start:end]:
                    weights[i] = (start + 1 + end) / 2
                start = end

        while len(new_population) < pop_size and len(history) < max_evals:
            p1, p2 = rng.choices(population, weights=weights, k=2)
            if rng.random() < crossover:
                point = rng.randint(1, D - 1)
                children = [p2[:point] + p1[point:],
                            p1[:point] + p2[point:]]
            else:
                children = [p1[:], p2[:]]

            for child in children:
                if len(new_population) == pop_size or len(history) == max_evals:
                    break
                for i in range(D):
                    if rng.random() < mutation:
                        child[i] = 1 - child[i]
                new_scores.append(evaluate(child))
                new_population.append(child)

        population, scores = new_population, new_scores

    return best, history


SETTINGS = {
    "Zaklad": {},
    "Populacia 30": {"pop_size": 30},
    "Populacia 100": {"pop_size": 100},
    "Elitizmus 20 %": {"elite_ratio": 0.2},
    "Mutacia 0.5 %": {"mutation": 0.005},
    "Krizenie 80 %": {"crossover": 0.8},
    "Poradova selekcia": {"selection": "rank"},
}


def main():
    print("Základ: populácia 50, elitizmus 10 %, mutácia 1 %, "
          "kríženie 100 %, ruletová selekcia, jednobodové kríženie.")
    print("Štatistiky sú z najlepších konečných výsledkov 10 behov.")
    print("Std = populačná smerodajná odchýlka (ddof=0).")

    for name, fitness in {"OneMax": onemax, "Leading Ones": leading_ones}.items():
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(name + " – priemerná konvergencia z 10 behov")

        for ax, D in zip(axes, [10, 30, 100]):
            comparison = []
            print(f"\n{name}, {D}D, limit {100 * D} ohodnotení na beh")
            print(f"{'Nastavenie':<22} {'Best':>5} {'Worst':>5} "
                  f"{'Mean':>7} {'Median':>7} {'Std':>7}")

            for label, parameters in SETTINGS.items():
                results, histories = [], []
                for run in range(10):
                    # Rovnaký zoznam seedov pre všetky nastavenia.
                    result, history = genetic_algorithm(
                        D, fitness, seed=1000 + run, **parameters)
                    results.append(result)
                    histories.append(history)

                avg_history = np.mean(histories, axis=0)
                ax.plot(np.arange(1, 100 * D + 1), avg_history, label=label)
                mean = np.mean(results)
                comparison.append((mean, np.mean(avg_history), label))
                print(f"{label:<22} {max(results):>5} {min(results):>5} "
                      f"{mean:>7.2f} {np.median(results):>7.2f} "
                      f"{np.std(results):>7.2f}")

            winner = max(comparison, key=lambda item: (item[0], item[1]))
            print(f"Najlepšie z testovaných: {winner[2]} (priemer {winner[0]:.2f}/{D}).")
            ax.set_title(f"{D}D")
            ax.set_xlabel("Počet ohodnotení účelovej funkcie")
            ax.set_ylabel("Priemer najlepšieho nájdeného skóre")
            ax.set_ylim(0, D + 1)
            ax.grid(True)

        axes[-1].legend(fontsize=8)
        fig.tight_layout()

    print("\nPorovnanie je orientačné pre tieto nastavenia a 10 seedov; "
          "nejde o dôkaz globálne najlepších parametrov.")
    plt.show()


if __name__ == "__main__":
    main()
