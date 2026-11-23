# Minimalne rozwiązanie P2

Samodzielne **modele + czyste obrazy → scores.csv**. `run.py`, `core.py` i cienki `optimizer.py` korzystają z oryginalnych `data.py` i `model.py`; jedna
konfiguracja Hydra. Brak importów z eksperymentów, jobów, shardów, watcherów,
etykiet backdoorów lub stałych ścieżek klastra. Poprzednie wyniki nie są potrzebne.

## Uruchomienie

Ze środowiska przygotowanego przez `uv` dla tego `pyproject.toml`:

```bash
cd p2_backdoors
uv run python run.py \
  models_dir=/absolute/path/test/models \
  clean_dir=/absolute/path/example/clean \
  output_dir=/absolute/path/new-output \
  device=cuda:0
```

`models_dir`: co najmniej dwa pliki `<id>.pt`, oryginalne state_dict ResNet18 z
CosineLinear. Cały podany zbiór stanowi populację transferu i normalizacji rang;
nie mieszać referencji z testem. Nie ma zaszytej liczby 200 modeli.
`clean_dir`: `labels.csv` z kolumnami `id,label` oraz `images/<id>.png`,
20 obrazów RGB 64×64 na każdą klasę. Kolejność CSV jest istotna dla splitu.
Model sam normalizuje kanały; nie normalizować obrazów wcześniej.

Wynik: `output_dir/scores.csv`, kolumny `id,c00,...,c19`, większy wynik = większe
podejrzenie. Folder wynikowy musi być nowy. Zawiera też zapisane łaty, liczniki,
surowe normy oraz wyniki obu gałęzi. Brak automatycznego wznawiania lub usuwania
łat. Sekwencyjny runner jest prosty, ale pełny zbiór jest kosztowny: dla 200 modeli
76 000 inwersji source-specific, 4 000 globalnych i około 304 mln obrazów w obu
transferach. Nie należy utożsamiać jego czasu z wcześniejszym biegiem wielo-GPU.

## Gdzie jest algorytm?

1. **Model i A/B — oryginalne `model.py` i `data.py`.** Oryginalny ResNet18, konwolucja wejściowa
   3×3, bez maxpool, `logits = 16*cos(feature, weight)`, FP32 i zamrożone wagi/BN.
   Seed0 dzieli każdą klasę na A10 i B10. B służy wyłącznie optymalizacji, A transferowi.
2. **Optymalizacja NC-style — `optimizer.py:invert` → `third_party/k_arm/Balanced_Scanner.py`, wywołania w `run.py`.**
   Optymalizujemy pikselowy wzorzec i miękką maskę przez tanh:
   `x'=(1-mask)*x+mask*pattern`, `CE(model(x'), target)+cost*sum(abs(mask))`.
   Adam lr0.1, betas(0.5,0.9), dokładnie100kroków. Cost startuje od0; po dwóch
   sukcesach ustawiany na0.001, co5 kolejnych sukcesów rośnie×1.5, porażek maleje÷1.5.
   Zachowujemy najmniejszą kwalifikującą maskę **sprzed kroku Adam**, z ASR B≥0.99.
   Nie kwalifikujemy stanu po setnym kroku, jeśli nie został zmierzony.
   To stałobudżetowa adaptacja NC/K-Arm, **nie oryginalny bandit K-Arm**.
3. **Gałąź globalna — `run.py`, `core.py:global_score`.** Dla każdego targetu
   optymalizacja na190 obrazach B spoza targetu; wszystkie20ramion mają wspólną
   losową inicjalizację, lecz niezależny Adam i cost. `raw=-L1(mask)/4096`.
   `z=(raw-median_20(raw))/(1.4826*MAD_20(raw))`, osobno dla każdego modelu.
4. **Stały bonus — `core.py:global_score`.** Każda zamrożona globalna łata jest
   stosowana do wszystkich modeli na tych samych190 obrazach A. Liczymy
   `G=own_ASR-mean_other_ASR`, **bez korekty clean w tej gałęzi**.
   `winner=z+1*(G>0.8)`; ścisły próg, +1 w jednostkach skalowanego MAD,
   nie prawdopodobieństwa. Pozostałe wartości nie są zerowane ani karane.
5. **Targeted/source-specific poisoning — druga pętla `run.py`.**
   Wszystkie20×19kierunków source→target, po100kroków na10obrazach B danej source.
   Każda para ma seed `(base+sha256(model_id)[:8]+source*1000003+target*10007)%2**32`.
   Nowe łaty nie są przesuwane ani reoptymalizowane podczas transferu.
   Na A10 tej samej source: `delta_r=patched_target_rate_r-clean_target_rate_r`.
   `S_pair=delta_own-mean_other(delta_r)`, `S_target=max_19sources(S_pair)`.
   `core.py:specificity_numerators` zachowuje licznik całkowity
   `M*own_delta_count-sum_all_delta_count`, mianownik `(M-1)*10`.
   **Bez kary offsource, dodatkowego MAD, reverse-gate lub filtra czystych modeli.**
   Ta gałąź szuka sygnału ograniczonego do źródła; nie dowodzi typu ataku.
6. **Fuzja — `core.py:fuse`.** Równa średnia globalnych midranków `winner` i
   `S_target`, liczonych po wszystkich M×20 parach. Remisy source są dokładne,
   oparte na licznikach. Wynik = `(rank_winner/N+rank_source/N)/2`.
   Nie jest prawdopodobieństwem; poprzedni diagnostyczny próg2 tutaj nie obowiązuje.

Brak udanej łaty lub zerowy MAD powoduje czytelny błąd zamiast fałszywego wyniku
„czystego”. ASR0.99 to wybór optymalizacji, nie wymóg zadania ani dowód backdoora.
Globalne i source-specific ramiona uruchamiane są kolejno zamiast round-robin:
zamrożony model bez losowych warstw i niezależne stany ramion dają tę samą naukę.

## Sprawdzenie i ograniczenia

- Nowy rdzeń odtworzył dokładnie MAD i bonus wszystkich4000par oraz source
  numerators z200zapisanych macierzy. Finalny CSV bitowo zgodny z kandydatem:
  `f6207fb07714057034e2a7520f00937c210e3c40089f48f40f661e61570da6fb`.
- Cztery różnicowe testy CPU adaptera względem poprzedniej implementacji:
  batch 10/190, dwa targety, 100 kroków, ASR 0.99; maski i wzorce bitowo zgodne.
- Model: identyczne logity i gradienty wejścia przy tym samym state_dict.
- Syntetyczny smoke test od dwóch state_dict i PNG do CSV naCPU przeszedł
  (2klasy,5kroków,ASR0 tylko na potrzeby testu przepływu; nie wynik naukowy).
- Szczegóły w `results/minimal-solution-verification.json`.
- Nie wykonano ponownego pełnego skanu200 tym folderem. Zgodność zapisanych ocen
  nie jest dowodem identyczności każdego nowego przebiegu GPU między wersjami
  bibliotek/urządzeniami. Konfiguracja produkcyjna pozostaje100kroków/ASR0.99.
- Użytkownik zgłosił **0.957** dla kandydata (wcześniej0.934), część public/private
  niepotwierdzona. Referencyjne0.982906 z3pozytywów/117negatywów to inny pomiar.
  Wynik zgłoszony nie izoluje wykrywania source-specific ataków ani czystych modeli.

Optymalizator importowany jest z zachowanych lokalnych plików
`third_party/k_arm/Balanced_Scanner.py` i `K_Arm_Scanner.py`; ich treść nie została
zmieniona. Zachowano `third_party/k_arm/LICENSE` (MIT, PurduePAML). Adapter dodaje
folder vendora do ścieżki importu, ponieważ vendor używa absolutnego importu pliku
sąsiedniego. Nie wymaga instalacji całego repozytorium ani `karm_detector.py`.
Zależność `tqdm` jest jawnie wymieniona w `pyproject.toml`.

`BalancedScanner` używa stałego przydziału; odziedziczona implementacja
`K_Arm_Scanner` dostarcza inicjalizację/tanh i parametry. Nie wywołujemy skanera
banditowego. Wymagane przez interfejs pola zgodności w Hydra nie włączają bandita.

## Układ plików i zachowane szablony

`run.py:main` wywołuje cztery etapy: `prepare_inputs`, `invert_models`,
`collect_transfer_counts`, `save_results`. `core.py` zawiera wyłącznie formuły ocen. `optimizer.py` przekazuje batch i inicjalizację
do zachowanego `BalancedScanner`; nie kopiuje jego pętli optymalizacji.
Adapter `prepare_inputs` przekazuje oryginalnemu `data.py` ścieżkę/seed i odtwarza
poprzednie ustawienia modułu. Wagi ładuje oryginalne `model.load`; model jest
następnie zamrażany przez runner. Nie ma kopii architektury ani loadera obrazów.

Oryginalne `data.py` i `model.py` pozostają w katalogu głównym. Moduły
`evaluate.py`, `scores.py`, `detectors.py`, `run_detector.py`, `make_shadows.py`
i `triggers.py` przeniesiono do `old/`; zmieniono tylko ich wzajemne importy.
`shadows/labels.csv` pozostaje w dotychczasowej lokalizacji. Historyczne skrypty
uruchamiaj **z katalogu `p2_backdoors`** (nie przez `python old/plik.py`):

```bash
uv run python -m old.run_detector
uv run python -m old.scores
uv run python -m old.make_shadows
```

To osobne polecenia; ostatnie generuje/trenuje modele shadow. Ustawienia historyczne
pozostają w modułach. Ścieżki `data/example`, `data/test`, `shadows` i `results`
są nadal względne wobec katalogu P2. Dane i wagi pozostają lokalne, ignorowane
przez Git. Biblioteki pomocnicze importuj jako `old.evaluate`, `old.scores` itd.
Zależności torch/torchvision/Pillow są w
`pyproject.toml`. Dawny `uv.lock` pozostaje lokalnie ignorowany: dotyczy poprzedniego
manifestu, więc nie przedstawiamy go jako aktualnego lockfile. `uv run` rozwiąże
zależności obecnego manifestu; nie regenerowano środowiska podczas refaktoru.
