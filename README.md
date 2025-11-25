# 耦合拉丁方陣族 (coupled_latin_family.py)

這個腳本提供兩種方法產生「耦合拉丁方陣族」(coupled Latin squares family)：
1. **有限體解析構造**：適用 `n` 為質數冪且 `r ≤ n-1`。
2. **遺傳演算法 (GA) 搜尋**：適用任意 `n`、`r`，以耦合違規的懲罰函數為目標最小化。

檔案位置：`LatinSquareStudio/coupled_latin_family.py`

## 主要功能
- `generate_coupled_latin_family(n, r, method="auto", ...)`：依條件自動選擇解析構造或 GA。
- `generate_field_coupled_family`：在 GF(n) 下直接生成 `r` 個拉丁方陣，欄/列對之間滿足耦合約束。
- `LatinFamilyGA`：以排列編碼的 GA 搜尋最佳方陣族，可選擇 lexicographic 或 power 懲罰。
- `collect_coupling_violations`：列出違反耦合約束的欄對及細節。
- `save_results_to_json`：將產生的方陣與附帶資訊輸出成 JSON（UTF-8）。

## 安裝需求
- Python 3.8+
- `numpy`
- 可選：`galois`（當 `n` 為質數冪且使用解析構造時需要）
- 範例繪圖：`matplotlib`（僅在 `__main__` 範例使用）

安裝示例：
```bash
pip install numpy galois matplotlib
```

## 基本使用
```python
from coupled_latin_family import generate_coupled_latin_family, save_results_to_json

# 例 1：質數 n=5，r=3，自動走有限體構造
squares_5, info_5 = generate_coupled_latin_family(
    n=5,
    r=3,
    method="auto",    # 對質數冪且 r≤n-1 會自動選解析
)
save_results_to_json(squares_5, info_5, "example_n5_r3.json")

# 例 2：n=6，r=2，強制用 GA
squares_6, info_6 = generate_coupled_latin_family(
    n=6,
    r=2,
    method="ga",
    ga_loss_mode="lexico",   # 或 "power"
    ga_loss_power=2,
    ga_params={
        "pop_size": 80,
        "max_generations": 5000,
        "timeout_seconds": 600,
        "seed": 123,
    },
)
save_results_to_json(squares_6, info_6, "example_n6_r2.json")
```

## 參數說明
- `n`：階數；`r`：要生成的方陣數量。
- `method`：`"auto"`（預設，質數冪且 `r ≤ n-1` 用解析，否則 GA）、`"field"`（強制解析）、`"ga"`（強制 GA）。
- `one_based`：`True` 時輸出元素為 1..n；`False` 時為 0..n-1。
- GA 相關：
  - `ga_loss_mode`：`"lexico"`（優先降低最大違規，再減少違規筆數，再降總懲罰）或 `"power"`（`Σ t^p`，強化大違規）。
  - `ga_loss_power`：`"power"` 模式的次方 `p`（預設 2）。
  - `ga_params`：傳給 `LatinFamilyGA` 的其他參數，如 `pop_size`、`max_generations`、`timeout_seconds`、`seed`、`use_sa_acceptance`、`sa_initial_T`、`sa_cooling_rate`、`use_tabu`、`tabu_tenure`、`generation_handler` 等。

## 輸出資訊 (`info` 字典)
- 通用：`method`、`best_loss`、`loss_curve`、`violations`、`n`、`r`。
- 解析法：`prime`、`power`。
- GA：`ga_loss_mode`、`ga_loss_power`。
- `violations` 內容包含欄對索引、匹配的列及懲罰值，方便檢查耦合違規。

## Generation Handler
- 實作 `BaseGenerationHandler` 的 `update(generation, best_loss, best_squares, violations)`，可在 GA 過程中取得即時最佳解。
- 範例 `DefaultGenerationHandler` 會把最新世代、分數、違規與最佳方陣存成屬性，方便外部監看或繪圖。

## 注意事項
- 解析構造僅適用 `n = p^k` 且 `r ≤ n-1`；否則請改用 GA。
- 若使用 `galois`，請確保已安裝且 `n` 合法；否則會丟出 ImportError 或 ValueError。
- GA 搜尋可能受隨機種子、族群大小與時間限制影響；可調整 `pop_size`、`max_generations`、`timeout_seconds` 以求更好解。
