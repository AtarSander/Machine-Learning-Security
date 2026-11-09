# Modele w końcowym rozwiązaniu

1. **Przy przewidywaniu wyniku dla nowego obrazu używamy trzech modeli: ResNeta-18, małego CNN oraz modelu jakości opartego na drzewach.** Dodatkowo podczas treningu CNN korzystamy z wcześniej wytrenowanej regresji logistycznej do dobierania trudnych przykładów. Łącznie w kodzie występują więc cztery wytrenowane modele, ale tylko trzy są potrzebne do predykcji. Podział rozwiązania na dwa etapy opisuje kolejność obliczeń, a nie liczbę modeli.

2. **Model pierwszy: ResNet-18 — mierzy wpływ wygładzenia prostokąta na klasyfikację obrazu.**

   To dostarczona z zadaniem sieć klasyfikująca obraz do jednej z 200 klas Tiny ImageNet. Uruchamiamy ją na oryginalnym obrazie, a potem na obrazach, w których wygładzono wnętrze kolejnych kandydatów na łatkę. Zmiany wyników klasyfikacji dają sześć cech każdego prostokąta: dla dwóch sił wygładzenia mierzymy zmianę marginesu klasy, zmianę jej prawdopodobieństwa i odległość między rozkładami prawdopodobieństw.

   Wagi ResNeta pozostają zamrożone. **Nie ma funkcji straty ResNeta w naszym treningu**, ponieważ go nie uczymy. Repozytorium zawiera architekturę i gotowe wagi, ale nie potwierdza funkcji straty użytej do pierwotnego wytrenowania dostarczonych wag.

   - Architektura: [model/resnet.py](model/resnet.py), klasy `ResNet18TIN` i `BasicBlock`.
   - Wagi: `model/resnet18_tin.pt`.
   - Uruchamianie i obliczanie cech: [src/candidates.py](src/candidates.py), `CandidateScorer.infer()` i `CandidateScorer.extract()`.
   - Opcjonalnie [src/candidate_openvino.py](src/candidate_openvino.py) wykonuje tę samą sieć przez OpenVINO. Nie jest to dodatkowy model.

3. **Model drugi: mały CNN — wybiera prostokąt najlepiej pasujący do łatki.**

   Dla każdego z 265 kandydatów CNN dostaje wycinek 40×40 z ośmioma kanałami: sześcioma kanałami reszt po wygładzeniu, maską ocenianego prostokąta i maską części wycinka znajdującej się wewnątrz obrazu. Dostaje też dziesięć liczb: sześć cech z ResNeta oraz cztery opisujące rozmiar i proporcje prostokąta.

   Sieć ma dwie warstwy splotowe: 8→16 i 16→32 kanały, z ReLU i zmniejszaniem rozdzielczości przez średnią. Dalej tworzy wektor 64 liczb, łączy go z dziesięcioma cechami i przez warstwy w pełni połączone wylicza jedną ocenę. Wygrywa prostokąt z najwyższą oceną. Ta ocena służy do porównywania kandydatów; nie jest prawdopodobieństwem obecności łatki.

   **Funkcja straty: ważona strata rankingowa dla par kandydatów.** IoU oznacza pole przecięcia prostokąta z prawdziwą łatką podzielone przez pole ich sumy. W parze pierwszy kandydat ma większe IoU niż drugi. Chcemy, aby CNN przyznał mu wyższą ocenę:

   ```text
   strata pary = (IoU_lepszego − IoU_gorszego)
                 × log(1 + exp(ocena_gorszego − ocena_lepszego))
   ```

   Wynik jest uśredniany po parach. Większa różnica jakości kandydatów daje większą wagę błędu. Pary o równym IoU nie trafiają do tej straty. Sieć uczymy optymalizatorem AdamW z learning rate 0.0001 i weight decay 0.0001. Wybór zapisanej wersji sieci odbywa się według średniego IoU wybranych prostokątów na obrazach walidacyjnych z łatką.

   - Architektura: [src/model.py](src/model.py), `CropScorer`.
   - Strata: ten sam plik, `pair_loss()`; `softplus` realizuje powyższy logarytm w sposób stabilny numerycznie.
   - Przygotowanie wycinka: `native_crop()` w tym samym pliku.
   - Dobieranie par i trening: [src/training.py](src/training.py), `pair_schedule_variable()` i `train()`.
   - Końcowe wagi: `outputs/more_sources_v1/selected.pt`.

4. **Model trzeci: regresor jakości — pomaga zdecydować, czy na obrazie w ogóle jest łatka.**

   To `HistGradientBoostingRegressor` z biblioteki scikit-learn: model złożony z drzew decyzyjnych, a nie sieć neuronowa. Ma 200 iteracji boostingowych, maksymalną głębokość drzewa 3 i learning rate 0.05. Otrzymuje 54 cechy prostokąta: 44 statystyki reszt po wygładzeniu, sześć cech z ResNeta i cztery cechy geometrii.

   Model został wcześniej nauczony przewidywania IoU prostokąta z prawdziwą łatką. Dla obrazów bez łatki docelowe IoU wynosiło zero. **Funkcja straty: błąd kwadratowy (`loss="squared_error"`)** między przewidywaną jakością a rzeczywistym IoU. Podczas końcowego treningu CNN ten model był już zamrożony.

   Przy predykcji obcinamy jego wyniki do przedziału [0, 1] i bierzemy maksimum ze wszystkich 265 prostokątów. Jeśli maksimum jest mniejsze niż **0.20670875367837696**, zwracamy brak łatki. W przeciwnym przypadku zwracamy prostokąt wybrany przez CNN. Prostokąt z najwyższą oceną jakości może być inny niż zwycięzca CNN.

   - Implementacja modelu i straty: biblioteka scikit-learn, klasa `sklearn.ensemble.HistGradientBoostingRegressor`.
   - Zapisany model: `outputs/candidate_quality_v1/quality.joblib`.
   - Wczytanie i ocena kandydatów: [src/inference.py](src/inference.py), `predict()`.
   - Decyzja prostokąt/brak łatki: ten sam plik, `submission()`.
   - Po uproszczeniu repozytorium kod ponownego trenowania tego regresora został usunięty razem z eksperymentami. Pozostały wytrenowany model i kod używający go do predykcji.

5. **Pomocniczy model treningowy: regresja logistyczna — wskazuje trudne błędne kandydatury.**

   To `StandardScaler` połączony z `LogisticRegression`. Korzysta z 50 cech: 44 statystyk reszt i sześciu odpowiedzi ResNeta. Jego wysokie oceny dla prostokątów o małym IoU pomagają znaleźć trudne przykłady tła lub wabików do porównania z dobrym prostokątem podczas treningu CNN.

   **Funkcja straty przy jego wcześniejszym uczeniu: binarna entropia krzyżowa, czyli log loss, z regularyzacją L2 (`C=1`).** W końcowej procedurze model jest tylko wczytywany. Nie aktualizujemy jego wag i nie używamy go do predykcji testowych.

   - Implementacja: biblioteka scikit-learn, `StandardScaler` i `LogisticRegression`.
   - Zapisany model: `outputs/ablation_full/reference/scorers.joblib`, element `fit_combined`.
   - Użycie: [src/training.py](src/training.py), `train()` uzupełnia `attack_score`, a `pair_schedule_variable()` korzysta z niego przy wyborze trudnych par.
   - Kod jego pierwotnego trenowania także został usunięty przy porządkowaniu eksperymentów.

6. **Próg odrzucenia i siatka prostokątów nie są dodatkowymi modelami.**

   Siatka 265 prostokątów jest zdefiniowana w [src/data.py](src/data.py), przez `SIZES`, `grid_boxes()` i `GRID`. Próg dobieramy na oznaczonych danych kalibracyjnych, wybierając wartość maksymalizującą wynik zadania: IoU dla obrazu z łatką, 1 za poprawne wskazanie braku łatki i 0 za błędną decyzję. Odpowiadają za to `thresholds()` i `calibrate()` w [src/calibration.py](src/calibration.py). Nie jest to uczenie kolejnej sieci ani dodatkowa funkcja straty CNN.
