"""
coupled_latin_family.py

產生「耦合」拉丁陣群的工具：

- 對任意 n，想要 r 個 n×n 拉丁陣，使得：
  對群中的任兩個陣 A, B，任意欄位組合 (cA, cB)，
  在所有 row 中 A[row,cA] == B[row,cB] 的次數最多 1 次。

- 若 n 為質數冪且 r <= n-1：
    用有限域定理 L_a[x,y] = a*x + y（GF(n)），
    直接給出零違規的耦合拉丁陣群（理論保證）。

- 其他情況：
    用 GA 在「拉丁方空間」上搜尋，拉丁性由構造保證，
    loss 衡量「耦合違規」(col,row 規則)。
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any
import time
import random
import numpy as np
import json
from abc import ABC, abstractmethod

try:
    import galois  # 若要支援 n = p^k, k>1，需 pip install galois
except ImportError:
    galois = None


# ================================================================
#  基本數論工具：判斷質數 / 質數冪
# ================================================================

def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def prime_factorization(n: int) -> Dict[int, int]:
    """回傳 {質因數: 次方}"""
    factors: Dict[int, int] = {}
    d = 2
    x = n
    while d * d <= x:
        while x % d == 0:
            factors[d] = factors.get(d, 0) + 1
            x //= d
        d = 3 if d == 2 else d + 2
    if x > 1:
        factors[x] = factors.get(x, 0) + 1
    return factors


def is_prime_power(n: int) -> Tuple[bool, int | None, int | None]:
    """判斷 n 是否為質數冪 n = p^k，回傳 (True,p,k) 或 (False,None,None)"""
    if n <= 1:
        return (False, None, None)
    fac = prime_factorization(n)
    if len(fac) != 1:
        return (False, None, None)
    (p, k), = fac.items()
    return (True, p, k)


# ================================================================
#  有限域構造：對質數冪 n, 產生 r 個完美耦合拉丁陣
# ================================================================

def generate_field_coupled_family(n: int, r: int, one_based: bool = True) -> List[np.ndarray]:
    """
    對 n = p^k (質數冪) 且 r <= n-1：
      產生 r 個 n×n 拉丁陣 L_a[x,y] = a*x + y（在 GF(n) 中運算），
      理論上任兩個 L_a, L_b 的 col/row 耦合違規數皆為 0。
    回傳陣的值為 0..n-1（若 one_based=True 則 +1）。
    """
    ok, p, k = is_prime_power(n)
    if not ok:
        raise ValueError(f"n={n} 不是質數冪，無法使用有限域構造。")
    if r < 1:
        raise ValueError("r 必須 >= 1")
    if r > n - 1:
        raise ValueError(f"質數冪構造僅能產生最多 n-1 個拉丁陣 (n={n})。")

    # 質數階：直接用 Z_n 作為 GF(n)
    if k == 1:
        squares: List[np.ndarray] = []
        a_values = list(range(1, n))[:r]
        for a in a_values:
            # L_a[x,y] = a*x + y (mod n)
            L = np.fromfunction(lambda i, j, a=a: (a * i + j) % n, (n, n), dtype=int)
            if one_based:
                L = L + 1
            squares.append(L)
        return squares

    # 非質數但為質數冪：用 galois.GF(n)
    if galois is None:
        raise ImportError(
            f"n={n} 是質數冪，但尚未安裝 'galois' 套件。請先: pip install galois"
        )

    GF = galois.GF(n)
    elems = list(GF.elements)  # 長度 n
    if len(elems) != n:
        raise RuntimeError("galois.GF(n) 回傳元素數量與 n 不符，請檢查環境。")

    elem_to_int = {elem: idx for idx, elem in enumerate(elems)}
    nonzero = elems[1:]  # 通常第 0 個是 0
    if r > len(nonzero):
        raise ValueError(f"r={r} 超過可用非零元素數量 {len(nonzero)}。")

    a_values = nonzero[:r]
    squares: List[np.ndarray] = []

    for a in a_values:
        L = np.zeros((n, n), dtype=int)
        for ix, x in enumerate(elems):
            for iy, y in enumerate(elems):
                v = a * x + y
                idx = elem_to_int[v]
                L[ix, iy] = idx
        if one_based:
            L = L + 1
        squares.append(L)

    return squares


# ================================================================
#  耦合違規統計（對拉丁陣群）
# ================================================================
def colpair_distance_penalty(colA: np.ndarray, colB: np.ndarray) -> int:
    """
    對單一欄位對 (cA, cB) 計算距離加權的違規：
    找出所有 row 使得 colA[row] == colB[row]，
    若只有 0 或 1 筆 → 0
    若 >=2 筆，對所有成對 row (ri, rj) 加總 (n - |rj-ri|)
    """
    n = len(colA)
    rows = [r for r in range(n) if colA[r] == colB[r]]
    k = len(rows)
    if k <= 1:
        return 0

    penalty = 0
    # rows 順序已依 row index 遞增，可不排序；穩妥起見也可 rows.sort()
    # rows.sort()
    for i in range(k):
        for j in range(i + 1, k):
            d = rows[j] - rows[i]          # d >= 1
            penalty += (n - d)             # d 越小，懲罰越大
    return penalty

def coupling_t_list_for_family(squares: List[np.ndarray]) -> List[int]:
    """
    對一群拉丁陣 squares (長度 r, 每個 n×n)：
    收集所有 pair-condition 的 t 值：

      t_{(i,j),(cA,cB)} = max(0, matches-1)
      matches = # {row | squares[i][row,cA] == squares[j][row,cB] }

    回傳所有 t>0 的列表。
    """
    if not squares:
        return []

    r = len(squares)
    n = squares[0].shape[0]
    t_list: List[int] = []

    for idxA in range(r):
        A = squares[idxA]
        for idxB in range(idxA + 1, r):
            B = squares[idxB]
            for cA in range(n):
                colA = A[:, cA]
                for cB in range(n):
                    colB = B[:, cB]
                    penalty = colpair_distance_penalty(colA, colB)
                    if penalty > 0:
                        t_list.append(penalty)
                    # matches = int(np.sum(colA == colB))
                    # t = max(0, matches - 1)
                    # if t > 0:
                    #     t_list.append(t)

    return t_list


def collect_coupling_violations(squares: List[np.ndarray]) -> List[Dict[str, Any]]:
    """
    收集所有耦合違規的詳細資訊。
    
    回傳格式：
    [
        {
            "square_A": int,     # 左陣索引
            "square_B": int,     # 右陣索引
            "col_A": int,        # 左陣欄位
            "col_B": int,        # 右陣欄位
            "matching_rows": [   # 所有相同的 row
                {
                    "row": int,
                    "value_A": int,
                    "value_B": int
                },
                ...
            ],
            "penalty": int       # 該 col-pair 的違規懲罰值
        },
        ...
    ]
    """
    if not squares:
        return []

    r = len(squares)
    n = squares[0].shape[0]
    violations: List[Dict[str, Any]] = []

    for idxA in range(r):
        A = squares[idxA]
        for idxB in range(idxA + 1, r):
            B = squares[idxB]
            for cA in range(n):
                colA = A[:, cA]
                for cB in range(n):
                    colB = B[:, cB]
                    
                    # 找出所有相同的 row
                    matching_rows = []
                    for row in range(n):
                        if colA[row] == colB[row]:
                            matching_rows.append({
                                "row": int(row),
                                "value_A": int(colA[row]),
                                "value_B": int(colB[row])
                            })
                    
                    # 計算懲罰值
                    penalty = colpair_distance_penalty(colA, colB)
                    
                    # 只記錄有違規的（penalty > 0 代表 matches >= 2）
                    if penalty > 0:
                        violations.append({
                            "square_A": int(idxA),
                            "square_B": int(idxB),
                            "col_A": int(cA),
                            "col_B": int(cB),
                            "matching_rows": matching_rows,
                            "penalty": int(penalty)
                        })

    return violations


# ================================================================
#  Generation Handler 基底類別
# ================================================================

class BaseGenerationHandler(ABC):
    """
    自定義 handler，用於在每次 generation 更新時接收資訊。
    子類別應實作 update 方法，並可設置屬性來儲存 score 和 violations。
    """
    @abstractmethod
    def update(
        self,
        generation: int,
        best_loss: int,
        best_squares: List[np.ndarray],
        violations: List[Dict[str, Any]]
    ) -> None:
        """
        每次 generation 更新時被調用。
        
        參數：
          generation: 當前世代數
          best_loss: 當前最佳 loss 值
          best_squares: 當前最佳拉丁陣列表
          violations: 當前最佳解的違規詳情列表
        """
        pass


class DefaultGenerationHandler(BaseGenerationHandler):
    """
    預設的 handler 實作，將資訊儲存在屬性中。
    """
    def __init__(self):
        self.generation = 0
        self.score = float('inf')
        self.violations: List[Dict[str, Any]] = []
        self.best_squares: List[np.ndarray] = []
    
    def update(
        self,
        generation: int,
        best_loss: int,
        best_squares: List[np.ndarray],
        violations: List[Dict[str, Any]]
    ) -> None:
        self.generation = generation
        self.score = best_loss
        self.violations = violations
        self.best_squares = best_squares


# ================================================================
#  Loss 策略基底與兩種實作 (lexico / power)
# ================================================================

class BaseCouplingLoss(ABC):
    @abstractmethod
    def compute_family(self, squares: List[np.ndarray]) -> int:
        """回傳整個拉丁陣群的耦合違規損失值"""
        pass


class LexicographicCouplingLoss(BaseCouplingLoss):
    """
    loss = A*M + B*K + S

    M = 所有 square-pair, col-pair 中 t 的最大值
    K = t > 0 的 (square-pair, col-pair) 數量
    S = 所有 t 的總和
    t = max(0, matches-1)

    權重 A,B 設計成三層 lexicographic：
      先最小化 M，再最小化 K，最後才是 S。
    """
    def __init__(self, n: int, r: int):
        self.n = n
        self.r = r
        num_pairs = r * (r - 1) // 2
        S_max = num_pairs * n * n * (n - 1)
        self.B = S_max + 1
        self.A = (n * n + 1) * self.B

    def compute_family(self, squares: List[np.ndarray]) -> int:
        t_list = coupling_t_list_for_family(squares)
        if not t_list:
            return 0
        M = max(t_list)
        K = len(t_list)
        S = sum(t_list)
        loss = self.A * M + self.B * K + S
        return loss


class PowerCouplingLoss(BaseCouplingLoss):
    """
    loss = Σ t^p
    p >= 2 時，單一 col-pair 上的集中違規會被放大懲罰。
    """
    def __init__(self, power: int = 2):
        assert power >= 1
        self.p = power

    def compute_family(self, squares: List[np.ndarray]) -> int:
        t_list = coupling_t_list_for_family(squares)
        loss = sum(t ** self.p for t in t_list)
        return loss


# ================================================================
#  GA：在「拉丁方空間」上搜尋耦合拉丁陣群
# ================================================================

class LatinFamilyGA:
    """
    對任意 n，產生 r 個 n×n 拉丁方陣的 GA：

    - 固定 base cyclic Latin square L0[i,j] = (i+j) mod n
      （對任意 n 都是合法 Latin 方陣）
    - 每個 square 用三個 permutation 表示：
        row_perm, col_perm, sym_perm
    - 個體 = [perm_0, perm_1, ..., perm_{3r-1}]
      其中第 k 個 square 對應 (perm[3k], perm[3k+1], perm[3k+2])
    - 解碼時用這三個 perm 作用在 L0 上，就一定得到合法拉丁方。
    """

    def __init__(
        self,
        n: int,
        r: int,
        loss_mode: str = "lexico",
        loss_power: int = 2,
        pop_size: int = 80,
        max_generations: int = 10_000,
        timeout_seconds: int = 300,
        seed: int = 123,
        # === 新增：SA / Tabu 開關與參數 ===
        use_sa_acceptance: bool = False,
        sa_initial_T: float = 1.0,
        sa_cooling_rate: float = 0.99,

        use_tabu: bool = False,
        tabu_tenure: int = 50,
        # === 新增：Generation Handler ===
        generation_handler: BaseGenerationHandler | None = None
    ):
        if r < 1:
            raise ValueError("r 必須 >= 1")
        self.n = n
        self.r = r
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.timeout_seconds = timeout_seconds
        self.seed = seed

        # base cyclic Latin square
        self.L0 = np.fromfunction(lambda i, j: (i + j) % n, (n, n), dtype=int)

        self.use_sa_acceptance = use_sa_acceptance
        self.sa_initial_T = sa_initial_T
        self.sa_cooling_rate = sa_cooling_rate

        self.use_tabu = use_tabu
        self.tabu_tenure = tabu_tenure
        self._tabu_list = []  # 存 hash

        # generation handler
        self.generation_handler = generation_handler

        # loss strategy
        if loss_mode == "lexico":
            self.loss_strategy: BaseCouplingLoss = LexicographicCouplingLoss(n, r)
        elif loss_mode == "power":
            self.loss_strategy = PowerCouplingLoss(power=loss_power)
        else:
            raise ValueError(f"未知的 loss_mode: {loss_mode}")

    # --- Latin 方建構 ---

    def build_square(self, row_perm: np.ndarray, col_perm: np.ndarray, sym_perm: np.ndarray) -> np.ndarray:
        sq = self.L0[row_perm][:, col_perm]
        return np.vectorize(lambda v: sym_perm[v])(sq)

    # --- GA 個體編碼 ---

    def random_perm(self) -> np.ndarray:
        p = list(range(self.n))
        random.shuffle(p)
        return np.array(p, dtype=int)

    def random_individual(self) -> List[np.ndarray]:
        # 3 permutations per square, total 3r perms
        return [self.random_perm() for _ in range(3 * self.r)]

    def decode_individual(self, ind: List[np.ndarray]) -> List[np.ndarray]:
        squares: List[np.ndarray] = []
        for k in range(self.r):
            row_perm = ind[3 * k]
            col_perm = ind[3 * k + 1]
            sym_perm = ind[3 * k + 2]
            squares.append(self.build_square(row_perm, col_perm, sym_perm))
        return squares

    # --- GA operators ---

    def fitness(self, ind: List[np.ndarray]) -> int:
        squares = self.decode_individual(ind)
        return self.loss_strategy.compute_family(squares)

    def mutate_perm(self, p: np.ndarray, rate: float = 0.3) -> np.ndarray:
        p = p.copy()
        if random.random() < rate:
            # 1~3 次 random swap
            for _ in range(random.randint(1, 3)):
                i, j = random.sample(range(self.n), 2)
                p[i], p[j] = p[j], p[i]
        return p

    def mutate(self, ind: List[np.ndarray], rate: float = 0.3) -> List[np.ndarray]:
        return [self.mutate_perm(p, rate) for p in ind]

    def crossover_perm(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """
        Order crossover (OX) for permutations。
        """
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[a:b] = p1[a:b]
        fill = [x for x in p2 if x not in child[a:b]]
        j = 0
        for i in range(size):
            if child[i] is None:
                child[i] = fill[j]
                j += 1
        return np.array(child, dtype=int)

    def crossover(self, ind1: List[np.ndarray], ind2: List[np.ndarray]) -> List[np.ndarray]:
        return [self.crossover_perm(p1, p2) for p1, p2 in zip(ind1, ind2)]

    def tournament_select(self, pop: List[List[np.ndarray]], fitnesses: List[int], k: int = 4) -> List[np.ndarray]:
        best_idx = None
        for _ in range(k):
            idx = random.randrange(len(pop))
            if best_idx is None or fitnesses[idx] < fitnesses[best_idx]:
                best_idx = idx
        return pop[best_idx]

    def _hash_individual(self, ind):
        """
        把一個個體（list of np.ndarray permutations）轉成 hashable 形式。
        這裡用每個 permutation 的 tuple 串起來。
        """
        return tuple(tuple(p.tolist()) for p in ind)
    # --- run GA ---

    def run(self, mutation_rate: float = 0.4, crossover_rate: float = 0.9
            ) -> Tuple[int, List[np.ndarray], List[int]]:
        """
        執行 GA，回傳：
          best_loss, best_squares(值 0..n-1), loss_curve(list)
        """
        import math

        random.seed(self.seed)
        np.random.seed(self.seed)

        pop = [self.random_individual() for _ in range(self.pop_size)]
        fits = [self.fitness(ind) for ind in pop]

        best_loss = min(fits)
        best_ind = pop[int(np.argmin(fits))]
        curve = [best_loss]

        # SA 初始溫度
        T = self.sa_initial_T

        # 清空 tabu list
        self._tabu_list = []

        # 初始 generation (gen=0) 的 handler 調用
        if self.generation_handler is not None:
            best_squares_initial = self.decode_individual(best_ind)
            violations_initial = collect_coupling_violations(best_squares_initial)
            self.generation_handler.update(
                generation=0,
                best_loss=best_loss,
                best_squares=best_squares_initial,
                violations=violations_initial
            )

        start = time.time()
        for gen in range(1, self.max_generations + 1):
            if time.time() - start > self.timeout_seconds:
                print(f"[INFO] GA timeout {self.timeout_seconds}s at generation {gen}")
                break

            new_pop: List[List[np.ndarray]] = []
            new_fits: List[int] = []

            while len(new_pop) < self.pop_size:
                # 選 parent
                p1 = self.tournament_select(pop, fits, k=4)
                p2 = self.tournament_select(pop, fits, k=4)
                f_p1 = self.fitness(p1)  # 當作 baseline

                # crossover / mutate 產生 child
                if random.random() < crossover_rate:
                    child = self.crossover(p1, p2)
                else:
                    child = [p.copy() for p in p1]
                child = self.mutate(child, rate=mutation_rate)
                f_child = self.fitness(child)

                # ---- Tabu 檢查 ----
                accept_child = True
                if self.use_tabu:
                    h = self._hash_individual(child)
                    if h in self._tabu_list and f_child >= best_loss:
                        # 在 tabu 且不是 global best 改善 → 拒絕
                        accept_child = False

                if accept_child:
                    # ---- SA acceptance ----
                    if self.use_sa_acceptance and T > 1e-8:
                        delta = f_child - f_p1
                        if delta <= 0:
                            # 比 parent 好 → 一定接受
                            pass
                        else:
                            prob = math.exp(-delta / T)
                            if random.random() > prob:
                                # 不接受 child → 用 p1 代替
                                child = [p.copy() for p in p1]
                                f_child = f_p1
                    else:
                        # 不開 SA → 若 child 比 parent 更爛，也直接接受（純 GA）
                        pass
                else:
                    # 被 tabu 擋住 → 退回 parent
                    child = [p.copy() for p in p1]
                    f_child = f_p1

                # 更新 tabu list（只記錄真正放進族群的 child）
                if self.use_tabu:
                    h_child = self._hash_individual(child)
                    self._tabu_list.append(h_child)
                    if len(self._tabu_list) > self.tabu_tenure:
                        self._tabu_list.pop(0)

                new_pop.append(child)
                new_fits.append(f_child)

            # 更新族群
            pop = new_pop
            fits = new_fits

            # 更新 global best
            gen_best = min(fits)
            if gen_best < best_loss:
                best_loss = gen_best
                best_ind = pop[int(np.argmin(fits))]
            curve.append(best_loss)

            # 調用 generation handler（每次 generation 都更新）
            if self.generation_handler is not None:
                best_squares_current = self.decode_individual(best_ind)
                violations = collect_coupling_violations(best_squares_current)
                self.generation_handler.update(
                    generation=gen,
                    best_loss=best_loss,
                    best_squares=best_squares_current,
                    violations=violations
                )

            # 更新 SA 溫度
            if self.use_sa_acceptance:
                T *= self.sa_cooling_rate

            if best_loss == 0:
                print(f"[INFO] GA found loss=0 at generation {gen}")
                break

        best_squares = self.decode_individual(best_ind)
        return best_loss, best_squares, curve


# ================================================================
#  對外主入口：generate_coupled_latin_family
# ================================================================

def generate_coupled_latin_family(
        n: int,
        r: int,
        method: str = "auto",     # "auto" / "field" / "ga"
        ga_loss_mode: str = "power",
        ga_loss_power: int = 2,
        ga_params: Dict[str, Any] | None = None,
        one_based: bool = True,
        generation_handler: BaseGenerationHandler | None = None,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    產生 r 個 n×n 的「耦合」拉丁陣群。

    - method="auto":
        若 n 為質數冪且 r <= n-1 -> 用有限域定理 (method="field")
        否則 -> 用 GA 搜尋 (method="ga")
    - method="field":
        強制使用有限域構造（若 n 不是質數冪會丟錯）
    - method="ga":
        強制用 GA 搜尋（任意 n,r 皆可，品質取決於 GA 參數與時間）

    回傳：
      squares: List[np.ndarray]，每個 n×n，值為 1..n（若 one_based=True）
      info: dict，包含 method、best_loss、loss_curve、violations 等資訊
    """
    if r < 1:
        raise ValueError("r 必須 >= 1")

    ok_pp, p, k = is_prime_power(n)

    # --- field branch ---
    if method == "field" or (method == "auto" and ok_pp and r <= n - 1):
        squares = generate_field_coupled_family(n, r, one_based=one_based)
        violations = collect_coupling_violations(squares)
        info = {
            "method": "field",
            "prime": p,
            "power": k,
            "best_loss": 0,
            "loss_curve": [0],
            "violations": violations,
            "n": n,
            "r": r,
        }
        return squares, info

    # --- GA branch ---
    if ga_params is None:
        ga_params = {}
    
    # 如果 ga_params 中有 generation_handler，優先使用；否則使用傳入的參數
    handler = ga_params.pop("generation_handler", generation_handler)
    
    ga = LatinFamilyGA(
        n=n,
        r=r,
        loss_mode=ga_loss_mode,
        loss_power=ga_loss_power,
        generation_handler=handler,
        **ga_params,
    )
    best_loss, best_squares_zero_based, curve = ga.run()
    if one_based:
        squares = [S + 1 for S in best_squares_zero_based]
    else:
        squares = best_squares_zero_based

    violations = collect_coupling_violations(squares)
    info = {
        "method": "ga",
        "best_loss": best_loss,
        "loss_curve": curve,
        "violations": violations,
        "n": n,
        "r": r,
        "ga_loss_mode": ga_loss_mode,
        "ga_loss_power": ga_loss_power,
    }
    return squares, info


def save_results_to_json(
    squares: List[np.ndarray],
    info: Dict[str, Any],
    filename: str = "coupled_latin_results.json"
) -> None:
    """
    將結果儲存成 JSON 格式。
    
    參數：
      squares: 拉丁陣列表
      info: 相關資訊（來自 generate_coupled_latin_family）
      filename: 輸出檔名
    """
    # 將 numpy array 轉成 list
    squares_list = [sq.tolist() for sq in squares]
    
    output = {
        "squares": squares_list,
        "n": info.get("n"),
        "r": info.get("r"),
        "method": info.get("method"),
        "best_loss": info.get("best_loss"),
        "loss_curve": info.get("loss_curve"),
        "violations": info.get("violations", []),
    }
    
    # 如果是 field method，加入質數資訊
    if info.get("method") == "field":
        output["prime"] = info.get("prime")
        output["power"] = info.get("power")
    
    # 如果是 GA method，加入 GA 參數
    if info.get("method") == "ga":
        output["ga_loss_mode"] = info.get("ga_loss_mode")
        output["ga_loss_power"] = info.get("ga_loss_power")
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] 結果已儲存至 {filename}")


# ================================================================
#  簡單示範
# ================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example 1: n = 5 (質數)，r = 3 -> 用 finite field 構造
    squares_5, info_5 = generate_coupled_latin_family(
        n=5,
        r=3,
        method="auto",   # 會自動走 field
    )
    print("Example n=5, r=3, method =", info_5["method"])
    print("best_loss =", info_5["best_loss"])
    print("violations count =", len(info_5["violations"]))
    for idx, sq in enumerate(squares_5):
        print(f"\nSquare #{idx}:")
        print(sq)
    
    # 儲存 Example 1 的結果
    save_results_to_json(squares_5, info_5, "example_n5_r3.json")

    # Example 2: n = 6, r = 2 -> 用 GA 搜尋（使用 handler）
    handler = DefaultGenerationHandler()
    squares_6, info_6 = generate_coupled_latin_family(
        n=6,
        r=2,
        method="ga",
        ga_loss_mode="lexico",
        ga_loss_power=2,
        generation_handler=handler,  # 傳入 handler
        ga_params={
            "pop_size": 80,
            "max_generations": 5000,
            "timeout_seconds": 600,  # 你可以改 300 試長一點
            "seed": 123,
        },
    )
    print("\nExample n=6, r=2, method =", info_6["method"])
    print("best_loss =", info_6["best_loss"])
    print("violations count =", len(info_6["violations"]))
    
    # 從 handler 讀取最新狀態
    print(f"\nHandler 狀態：")
    print(f"  當前 generation: {handler.generation}")
    print(f"  當前 score: {handler.score}")
    print(f"  違規數量: {len(handler.violations)}")
    
    for idx, sq in enumerate(squares_6):
        print(f"\nSquare #{idx}:")
        print(sq)
    
    # 顯示違規詳情（如果有的話）
    if info_6["violations"]:
        print("\n違規詳情：")
        for v in info_6["violations"][:5]:  # 只顯示前5個
            print(f"  Square {v['square_A']} col {v['col_A']} vs Square {v['square_B']} col {v['col_B']}:")
            print(f"    相同的rows: {[m['row'] for m in v['matching_rows']]}, penalty={v['penalty']}")
    
    # 儲存 Example 2 的結果
    save_results_to_json(squares_6, info_6, "example_n6_r2.json")
