# Ostateczne rozwiązanie

1. **Cała implementacja jest podzielona na dwa etapy: wybór miejsca ataku i decyzję, czy atak w ogóle występuje.**

   W pierwszym etapie sieć CNN wskazuje prostokąt, który według niej najlepiej pasuje do łatki atakującej. W drugim osobny model ocenia, czy mamy wystarczające podstawy, żeby zgłosić atak na tym obrazie. Jeśli tak, zapisujemy prostokąt wybrany przez CNN. Jeśli nie, zapisujemy `NONE`, czyli „brak ataku”.

   Ten podział jest potrzebny, ponieważ samo wybranie najlepszego prostokąta nie oznacza, że obraz zawiera atak. Nawet na czystym obrazie jeden z ocenianych prostokątów otrzyma najwyższą ocenę. Potrzebujemy więc osobnego mechanizmu pozwalającego odrzucić wszystkie propozycje.

2. **Pierwszy etap odpowiada na pytanie: „Gdyby na obrazie był atak, gdzie najprawdopodobniej się znajduje?”.**

   CNN porównuje 265 wcześniej ustalonych prostokątów i wybiera jeden z nich. Nie rysuje prostokąta od zera i nie wyznacza dowolnych współrzędnych. Sprawdza różne położenia oraz rozmiary możliwej łatki i każdemu przypisuje ocenę. Najwyższa ocena oznacza najlepszą propozycję lokalizacji, a nie prawdopodobieństwo obecności ataku.

   Prostokąty powstają przez przesuwanie po obrazie okien o rozmiarach 14×28, 28×14, 16×24, 24×16, 20×20 i 24×24 piksele. Przesuwamy je co 8 pikseli, uwzględniając również pozycje przy krawędziach. Dla obrazu 64×64 daje to 265 propozycji. Wszystkie spełniają ograniczenia pola 8–15% obrazu i proporcji od 1:2 do 2:1.

   Sprawdzane prostokąty mogą się nakładać, bo są różnymi propozycjami położenia tej samej łatki. Ostatecznie zwracamy tylko jeden. Ograniczenie braku nakładania się prawdziwych łatek i wabików nie dotyczy tych alternatywnych propozycji.

3. **Żeby porównać prostokąty, zbieramy informacje o wyglądzie obrazu i o wpływie każdego obszaru na klasyfikator.**

   Sam nietypowy wygląd obszaru nie wystarcza: zarówno atak, jak i wabik mogą wyglądać jak szum. Korzystamy więc z dwóch rodzajów informacji. Pierwszy opisuje drobne zmiany widoczne w obrazie. Drugi opisuje, jak zmienia się odpowiedź dostarczonego klasyfikatora ResNet, gdy wygładzimy dany obszar.

   Te informacje przygotowujemy przed oceną prostokątów przez CNN. Odpowiedzi ResNetu zapisujemy, żeby nie obliczać ich ponownie podczas każdego treningu lub tworzenia kolejnego zgłoszenia.

4. **Drobne zmiany w obrazie pokazujemy przez porównanie go z wersją rozmytą.**

   Rozmywamy obraz filtrem Gaussa na dwa sposoby, z parametrami 0,7 i 1,2. Następnie odejmujemy każdą rozmytą wersję od oryginału. Otrzymujemy dwie mapy różnic, każdą z trzema kanałami koloru, czyli łącznie sześć kanałów.

   W tych mapach pozostają drobne szczegóły, które wygładzanie usunęło. Mogą to być ślady perturbacji, ale również zwykłe krawędzie i tekstury. Dlatego same mapy różnic nie są jeszcze informacją, gdzie znajduje się atak — model musi nauczyć się je wykorzystywać.

5. **Wpływ prostokąta na klasyfikację mierzymy za pomocą zamrożonego ResNetu.**

   Najpierw sprawdzamy odpowiedź ResNetu na oryginalny obraz. Następnie zastępujemy piksele wewnątrz ocenianego prostokąta ich rozmytą wersją i ponownie uruchamiamy klasyfikator. Sprawdzamy zmianę prawdopodobieństwa pierwotnie wybranej klasy, zmianę jej przewagi nad innymi klasami oraz zmianę całego zestawu prawdopodobieństw.

   Powtarzamy to dla obu sposobów rozmycia, uzyskując sześć liczb dla każdego prostokąta. ResNet nie jest douczany. Nie potrzebujemy też prawdziwej klasy obrazu testowego: obserwujemy zmianę jego własnej odpowiedzi. Wygładzenie obszaru nie gwarantuje usunięcia ataku, ale dostarcza dodatkowej informacji o tym obszarze.

6. **CNN łączy te informacje i ocenia każdą propozycję lokalizacji.**

   Wokół środka każdego prostokąta pobieramy wycinek 40×40 pikseli z sześciu map różnic. Dodajemy dwa oznaczenia: które piksele wycinka należą do ocenianego prostokąta i które należą do rzeczywistego obrazu. Fragmenty wystające poza obraz wypełniamy zerami. Wycinków nie rozciągamy ani nie zmniejszamy.

   CNN analizuje te osiem kanałów w dwóch blokach konwolucyjnych, z 16 i 32 kanałami. Wynik łączymy z sześcioma liczbami uzyskanymi z ResNetu oraz szerokością, wysokością, powierzchnią i proporcjami prostokąta. Ostatnia część sieci oblicza jego ocenę. Tak powstaje ranking 265 propozycji i wybór najlepszego prostokąta.

7. **Drugi etap odpowiada na pytanie: „Czy zgłosić atak, czy odrzucić ten obraz?”.**

   Osobny model oparty na drzewach decyzyjnych przewiduje jakość każdego z 265 prostokątów. Jego zadaniem jest oszacowanie, jak dobrze prostokąt pokrywa prawdziwą łatkę. Na obrazach bez ataku poprawna jakość każdego prostokąta wynosi zero.

   Model korzysta z 54 liczb: 44 statystyk map różnic, sześciu odpowiedzi ResNetu i czterech informacji o rozmiarze oraz proporcjach prostokąta. Bierzemy najwyższą przewidywaną jakość spośród całej siatki. Jeżeli jest niższa niż **0,20670875367837696**, zwracamy `NONE`. Jeżeli osiąga ten próg, zwracamy prostokąt wybrany w pierwszym etapie przez CNN.

   Ważny szczegół: drugi model sprawdza najwyższą ocenę jakości w całej siatce. Nie musi ona należeć do tego samego prostokąta, który wybrał CNN. Jego rola w końcowym rozwiązaniu to zaakceptowanie lub odrzucenie obrazu, a nie zmiana lokalizacji wybranej w pierwszym etapie.

8. **Dwa etapy były uczone do różnych zadań.**

   CNN uczył się wybierać lepszy prostokąt spośród kilku możliwości. Pokazywaliśmy mu pary z tego samego obrazu: jeden prostokąt lepiej pokrywał prawdziwą łatkę, drugi gorzej. Model był karany, jeśli wyżej oceniał gorszy prostokąt.

   Jakość pokrycia mierzymy przez IoU: pole wspólnej części przewidywanego i prawdziwego prostokąta dzielimy przez pole zajmowane łącznie przez oba. Wynik 1 oznacza idealne pokrycie, a 0 — brak pokrycia. Im większa różnica IoU w parze, tym większą wagę ma błędna kolejność ocen.

   Drugi model uczył się bezpośrednio przewidywać IoU na podstawie 54 opisujących prostokąt liczb. Był trenowany na 512 oryginalnych obrazach, również tych bez ataku. Nie douczaliśmy go podczas ostatniego rozszerzenia treningu CNN.

9. **Dane zostały podzielone tak, żeby obrazy używane do oceny nie aktualizowały wag CNN.**

   Z 5000 oznaczonych obrazów 3200 przeznaczyliśmy do treningu, 800 do oceny rozwiązania i doboru progu odrzucenia, a 1000 pozostawiono do wcześniejszych sprawdzeń. Z puli 3200 odłożyliśmy jeszcze 128 obrazów do wyboru najlepszej wersji CNN. Pozostałe 3072 tworzyły pulę końcowego treningu. Oddzielne 5000 obrazów testowych służyło wyłącznie do przygotowania zgłoszenia; nie znaliśmy ich poprawnych odpowiedzi.

   **W tym rozwiązaniu nie użyliśmy danych syntetycznych.** Podział zadania to 80% obrazów z atakiem, 10% z samymi wabikami i 10% czystych. `NONE` jest poprawną odpowiedzią zarówno dla czystych obrazów, jak i dla obrazów zawierających wyłącznie wabiki.

10. **Końcowy CNN powstał przez douczenie wcześniejszego modelu na większej liczbie obrazów.**

    Początkowy model korzystał z puli 384 oryginalnych obrazów. Jego zapisaną wersję po 500 krokach treningu wybrano na podstawie wyników na odłożonych 128 obrazach. Od tej wersji rozpoczęliśmy końcowe douczanie, zachowując budowę sieci i wcześniejszy sposób przeliczania jej danych wejściowych.

    Dodaliśmy 2688 oryginalnych obrazów. Dla każdego zapisaliśmy po 32 różne prostokąty: dobrze pokrywające atak, częściowo go pokrywające, obejmujące wabiki i położone w innych miejscach. Brakujące grupy uzupełnialiśmy innymi propozycjami. Znane położenie ataku służyło tylko do przygotowania przykładów treningowych. W teście zawsze sprawdzamy wszystkie 265 prostokątów.

    Dla wcześniejszych obrazów mieliśmy pełne zestawy po 265 prostokątów. Losowaliśmy jednak obrazy równomiernie, żeby większa liczba zapisanych propozycji nie powodowała, że stare obrazy zdominują trening.

11. **Jeden krok treningu CNN obejmuje osiem obrazów i 32 oceny prostokątów.**

    Losujemy osiem różnych obrazów z atakiem. Z każdego wybieramy dwie pary prostokątów, a w każdej parze jeden prostokąt lepiej pokrywa łatkę niż drugi. Model ocenia więc **8 × 2 × 2 = 32 prostokąty** i na podstawie błędów poprawia swoje wagi.

    Porównujemy zarówno podobne położenia granic, jak i dobry prostokąt z błędnym regionem obejmującym tło lub wabik. Obrazy czyste i z samymi wabikami nie dają par o różnym IoU, ponieważ wszystkie ich prostokąty mają poprawną wartość zero. Dlatego w aktualizowaniu CNN uczestniczyły faktycznie **2453 obrazy z atakiem**, w tym 2146 nowych. Obrazy bez ataku były uwzględnione przy uczeniu osobnego modelu jakości i dobieraniu progu odrzucenia.

12. **Douczanie trwało 4000 dodatkowych kroków, z regularnym sprawdzaniem wyników.**

    Użyliśmy optymalizatora AdamW z szybkością uczenia 0,0001 i współczynnikiem ograniczającym wzrost wag 0,0001. Nie obracaliśmy obrazów ani nie stosowaliśmy odbić, ponieważ zapisane odpowiedzi ResNetu dotyczyły oryginalnego ułożenia pikseli. Łącznie były to 32 000 losowań obrazów i 128 000 ocen prostokątów; te liczby obejmują powtórzenia.

    Co 250 kroków ocenialiśmy model na tych samych 128 odłożonych obrazach, za każdym razem na pełnej siatce 265 prostokątów. Model początkowy również mógł zostać wybrany, gdyby douczanie nie pomogło. Najlepsza była wersja po 4000 dodatkowych krokach: średnie IoU na obrazach z atakiem wzrosło z **0,4559 do 0,4758**. Poprawiło się 27 lokalizacji, a pogorszyło 20.

13. **Po wyborze CNN sprawdziliśmy całe rozwiązanie na 800 obrazach i dopasowaliśmy próg odrzucenia.**

    Wybrany model porównaliśmy z poprzednim na tych samych obrazach. Przy ocenie progów podzieliliśmy je na pięć grup. Za każdym razem próg dobieraliśmy na czterech grupach i mierzyliśmy wynik na piątej, niewykorzystanej do jego doboru. Wszystkie prostokąty jednego obrazu pozostawały razem. Po porównaniu modeli końcowy próg zgłoszenia ustaliliśmy na wszystkich 800 obrazach. Nie używaliśmy odpowiedzi dla zbioru testowego.

    Średnie IoU wybranego prostokąta wzrosło z **0,4852 do 0,4926**. Przy tym samym, pierwotnym progu wynik punktowy wzrósł z **0,4876 do 0,4927**. Przy osobno dobieranych progach, ocenianych w opisanym podziale na pięć grup, wynik wzrósł z **0,4917 do 0,4968**.

    Poprawa wystąpiła w czterech z pięciu grup, ale jest mała. Oszacowany przedział niepewności dla różnicy wyników wynosi od **−0,00539 do +0,01532** i obejmuje zero. Nie możemy więc uznać poprawy za pewną ani obiecać lepszego wyniku na platformie. Te dane były też wcześniej analizowane podczas rozwoju rozwiązania, więc nie są całkowicie nowym testem.

14. **Większy trening pomógł średnio, ale nie we wszystkich przypadkach.**

    Poprawiła się lokalizacja ataków CW, MI-FGSM i ukierunkowanego PGD. Największy wzrost wystąpił dla łatek przy krawędziach: IoU wzrosło z **0,5235 do 0,5583** na 168 obrazach.

    Dla nieukierunkowanego PGD IoU spadło z **0,5268 do 0,5129** na 152 obrazach. Dla 479 ataków poza krawędziami wynik nieznacznie spadł z **0,4718 do 0,4696**. Wszystkie środki łatek atakujących w tej kalibracji znajdowały się w dolnej połowie obrazu, więc te wyniki nie potwierdzają skuteczności w górnej połowie.

    Pozostają też ograniczenia samej metody: CNN wybiera ze stałej siatki, a nie poprawia granic dowolnie. Oznaczenie pikseli poza obrazem może przekazywać mu informację o bliskości krawędzi i sprzyjać wykorzystywaniu typowych położeń łatek.

15. **Najdroższe obliczenia wykonujemy raz i zapisujemy ich wyniki.**

    Ocena wpływu wszystkich prostokątów na ResNet wymaga 530 zmodyfikowanych wersji jednego obrazu: 265 prostokątów razy dwa rozmycia. Zapisanie odpowiedzi umożliwia późniejsze uczenie kolejnych CNN i tworzenie zgłoszeń bez powtarzania tej pracy.

    Dla nowych obrazów treningowych wystarczyły po 32 prostokąty, czyli po 64 zmodyfikowane wersje obrazu. Łącznie wykonaliśmy 172 032 takie oceny oraz 2688 ocen obrazów oryginalnych. Obliczenie i zapisanie nowych cech zajęło około **196 sekund**. Używaliśmy Intel GPU przez OpenVINO i zaakceptowanej wcześniej precyzji FP16, czyli obliczeń na liczbach zapisanych z mniejszą dokładnością. Sam trening z okresową oceną na 128 obrazach trwał około **117 sekund**, bez przygotowania danych.

16. **Gotowe zgłoszenie zawiera wynik obu etapów dla każdego obrazu testowego.**

    [submission_more_sources.csv](submission_more_sources.csv) zawiera **5000 odpowiedzi, w tym 1211 NONE**, czyli 24,22%.

    Zapisujemy całkowite współrzędne prostokąta albo `(0,0,0,0)`. Prawa i dolna współrzędna oznaczają koniec prostokąta: na przykład zakres od 10 do 20 obejmuje piksele od 10 do 19. Sprawdziliśmy wszystkie identyfikatory, ich kolejność, poprawność współrzędnych i zgodność każdego wiersza z ocenami modeli. Plik przygotowano przed terminem konkursowym.
