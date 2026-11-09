# Lokalizacja łatki

Implementacja ma dwa etapy: obliczenie cech 265 prostokątów na obrazie 64×64, a następnie wybór najlepszego przez CNN. Osobny model jakości decyduje, czy zwrócić prostokąt, czy brak łatki. Szczegóły są w [opisie rozwiązania](opis_rozwiazania.md).

Kod obejmuje końcowe rozwiązanie:

- `data.py`, `candidates.py`, `batched_residuals.py` — obrazy, prostokąty, reszty po wygładzaniu i reakcje zamrożonego ResNeta;
- `model.py` — CNN, wycinki z maskami i funkcja straty;
- `training.py` — dobór kandydatów i trening na 3072 oryginalnych obrazach, z wyborem modelu na osobnych 128 obrazach;
- `calibration.py`, `inference.py` — próg odrzucenia, predykcje i zapis CSV;
- `candidate_openvino.py` — opcjonalne przyspieszenie ResNeta przez OpenVINO.

Polecenia uruchamiaj z `p1_student`. Instalacja: `uv sync` lub `pip install -r requirements.txt`. Przykłady używają istniejącego `.venv`.

## Gotowe zgłoszenie

`submission_more_sources.csv` pozostaje końcowym zgłoszeniem. Odtworzenie go z zapisanych wyników, bez ponownego liczenia modeli:

```bash
.venv/bin/python -m src export --output submission.csv
```

Polecenie sprawdza sumy kontrolne wyników i kolejność identyfikatorów. Nie nadpisuje istniejącego CSV.

Pełne obliczenia na obrazach, z wznawialnym zapisem cech:

```bash
.venv/bin/python -m src predict --output submission.csv
```

Domyślnie działa PyTorch na CPU. Opcja `--backend openvino-gpu-fp16` wymaga osobno zainstalowanego OpenVINO i sterownika GPU Intela. Precyzja backendu może wpływać na wynik; dokładne odtworzenie istniejącego zgłoszenia zapewnia `export`.

## Trening i nowe zgłoszenie

Trening zachowuje końcową procedurę: start z oryginalnej wersji CNN po 500 aktualizacjach, stałą normalizację, pary kandydatów z ośmiu różnych obrazów na aktualizację i 4000 dodatkowych aktualizacji. Potrzebuje zachowanych podziałów, cech i modeli w `outputs/`; nie jest treningiem od zera. Model jakości pozostaje zamrożony.

```bash
.venv/bin/python -m src train --output outputs/training
.venv/bin/python -m src score --checkpoint outputs/training/selected.pt --cache outputs/rescue_v1/validation_base.npz --images train/images --output outputs/calibration_scores.npz
.venv/bin/python -m src calibrate --scores outputs/calibration_scores.npz --output outputs/calibration.json
.venv/bin/python -m src score --checkpoint outputs/training/selected.pt --output outputs/test_scores.npz
.venv/bin/python -m src submit --scores outputs/test_scores.npz --calibration outputs/calibration.json --output submission_retrained.csv
```

Ponowne obliczenie 32 kandydatów dla każdego z 2688 dodatkowych obrazów treningowych: `python -m src acquire --output outputs/acquired`. Następnie podaj `train --records outputs/acquired/records`. Zapisane cechy pozostałych 384 obrazów i 128 obrazów walidacyjnych są nadal wymagane.

## Weryfikacja

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/ruff check src tests
```

Testy sprawdzają cechy, geometrię, dobór par, gradienty, kalibrację i decyzję o braku łatki. Gdy lokalne dane są dostępne, sprawdzają też zgodność CNN z zapisanymi ocenami i odtworzenie wszystkich 5000 wierszy zgłoszenia.

Dane, modele, zapisane wyniki eksperymentów i wcześniejsze opisy pozostały na dysku. Usunięto ich generatory raportów oraz kod dawnych metod.
