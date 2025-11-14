# Algorytmy ewolucyjnej optymalizacji wielokryterialnej
### Przykład wykorzystania platformy GitHub i notatników Jupyter do stworzenia nowoczesnej formy publikacji naukowej w duchu Open Science

---
Artykuł prezentuje analizę porównawczą wybranych algorytmów ewolucyjnej optymalizacji wielokryterialnej. Do badań 
wykorzystano algorytmy: NSGA- II, SPEA2, MOEA/D oraz IBEA. Skuteczność ich funkcjonowania została zbadana 
z wykorzystaniem znanych funkcji testowych (tzw. banchmarków): 2-kryterialnych Kursawe i Binh2 oraz 3-kryterialnych 
DTLZ2 i DTLZ7. Eksperymenty numeryczne i wnioski poprzedzono syntetycznie ujętym wstępem teoretycznym. Obliczenia
wykonano z wykorzystaniem biblioteki jMetalPy 1.7.0 (https://github.com/jMetal/jMetalPy).

---
W prezentowanej pracy przyjęto formę „żywego” artykułu w postaci Jupyter Notebooka opublikowanego na platformie GitHub, 
aby maksymalnie zwiększyć rzetelność i przejrzystość całego procesu badawczego. Taka forma integruje narrację naukową 
z kodem, danymi oraz wynikami, dzięki czemu każdy etap analizy jest jawny i możliwy do odtworzenia. Notebook zawiera 
klasyczną strukturę publikacji naukowej — tytuł, autorstwo, wprowadzenie, część teoretyczną, eksperymenty oraz wnioski 
— ale rozszerza ją o pełną dokumentację obliczeń, która w tradycyjnych artykułach jest z konieczności jedynie 
streszczana.

Wszystkie eksperymenty zostały wykonane w osobnych notebookach, a ich wyniki są dostępne w repozytorium. Umożliwia to 
niezależną weryfikację procedur, parametrów, implementacji oraz sposobu przetwarzania danych. Repozytorium obejmuje 
także skrypty generujące wykresy i tabele, co pozwala sprawdzić nie tylko rezultaty, ale również sam proces analizy. 
Dzięki temu użytkownik ma pełny wgląd w logikę badania, może powtórzyć je krok po kroku, prześledzić wpływ 
poszczególnych decyzji obliczeniowych oraz zidentyfikować ewentualne błędy lub niejasności.

Publiczny charakter repozytorium otwiera przestrzeń do dyskusji naukowej i krytycznej oceny, co w praktyce wzmacnia 
obiektywizm i podnosi wartość pracy. Zamiast dostarczać jedynie końcowy zestaw wyników, notatnik ujawnia całą ścieżkę 
prowadzącą do wniosków. Taka otwarta forma sprzyja zarówno transparentności, jak i replikowalności: każdy zainteresowany
może uruchomić te same eksperymenty, na tych samych danych i w identycznych warunkach obliczeniowych, a także testować 
własne warianty algorytmów. Sam autor ma możliwość wprowadzenia poprawek do publikacji, co zostaje odnotowane w historii
commitów do repozytorium - jednocześnie cytujący mogą powoływać się na artykuł z uwzględnieniem stanu i wersji w danym 
punkcie czasu.

W przeciwieństwie do statycznych publikacji — nawet w renomowanych czasopismach — podejście notebookowe eliminuje 
niejasności dotyczące implementacji, ukrytych założeń, czy przetwarzania danych. Ogranicza to ryzyko pomyłek 
interpretacyjnych oraz ułatwia późniejsze rozszerzenia, poprawki i dalszy rozwój badania. „Żywy” artykuł nie jest 
zamrożonym dokumentem, lecz pełnym środowiskiem badawczym, które można analizować, uruchamiać, nadpisywać i rozwijać, 
co stanowi praktyczne urzeczywistnienie zasad otwartej i odpowiedzialnej nauki.

---
#### Struktura repozytorium
Notatniki:
* /moea.ipnyb – główny artykuł
* /comparative_analysis.ipnyb - notebook z eksperymentami zw. z analizą porównawczą algorytmów na wybranych probelmach (wyniki lądują w folderze \results\comparative_analysis\)
* /fonseca_selected_generation.ipnyb - notebook z eksprymentami prezentującymi przykładowy proces ewolucyjnej optymalizacji wielokryterialnej (wyniki lądują w folderze \results\selected_generations\)

Moduły pomocnicze:
* /plots.py – moduł pomocniczy do prezentacji wykresów na bazie wyników z podfolderów znajdujących się w folderze \results\
* /evaluation.py – moduł pomocniczy do obróbki wyników analizy porównawczej (bazuje na wynikach z folderu \results\comparative_analysis\)

Foldery:
* /results/ – wygenerowane pliki z rozwiązaniami problemów
* /resoruces/ - pliki pomocnicze potrzebne do wykonania eksperymentów z wykorzystaniem biblioteki jMetalPy

---
### Multi-objective evolutionary optimization algorithms - a tutorial. An illustration of combining GitHub and Jupyter Notebooks to create a modern scientific publication in the spirit of Open Science
