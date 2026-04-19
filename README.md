# ADOM_2026L_Głębokie_CNN-VGG
Repozytorium przeznaczone na projekt w ramach zajęć akademickich "Analiza danych obrazowych i multimedialnych", temat 7 - Głębokie CNN - VGG (2014).

## Cel
<!-- Stworzenie środowiska umożliwiającego przeprowadzenie eksperymentów dydatkycznych związanych z głęboką siecią CNN - VGG16. -->

Analiza architektury VGG16 w porównaniu z AlexNet, ze szczególnym uwzględnieniem wpływu głębokości sieci, dropout, fine-tuningu oraz odporności na zakłócenia obrazu.

## Instalacja
W trakcie

## Uruchomienie
W trakcie

## Zadanie projektowe (Temat 7 - Głębokie CNN – VGG (2014))

**Artykuł:** https://arxiv.org/pdf/1409.1556

**Repozytorium implementacji:** https://docs.pytorch.org/vision/stable/models/vgg.html

### Opis wykorzystywanej techniki

Architektura VGG stanowi rozwinięcie idei głębokich sieci konwolucyjnych poprzez systematyczne zwiększenie liczby warstw przy zachowaniu prostych filtrów o rozmiarze 3×3. Zastosowanie wielu kolejnych małych filtrów pozwala uzyskać dużą efektywną szerokość pola recepcyjnego przy jednoczesnym ograniczeniu liczby parametrów w pojedynczej warstwie. Model charakteryzuje się regularną, powtarzalną strukturą bloków konwolucyjnych, co czyni go architekturą
przejrzystą i łatwą do analizy. W porównaniu do AlexNet, VGG znacząco zwiększa głębokość
sieci, co przekłada się na lepszą jakość reprezentacji cech wizualnych. Jednocześnie wiąże się to
ze znacznym wzrostem liczby parametrów oraz kosztu obliczeniowego. VGG stał się jednym z
najczęściej wykorzystywanych modeli bazowych (backbone) w wielu zadaniach wizji komputerowej. Analiza tej architektury pozwala zrozumieć wpływ głębokości sieci na jakość reprezentacji
oraz ograniczenia wynikające z rosnącej złożoności modelu.

### Wymagane eksperymenty

- VGG16 vs AlexNet
- Dropout włączony/wyłączony
- Analiza feature maps
- Pomiar pamięci RAM
- Rozszerzenie GPU: pełny fine-tuning + analiza warstw głębokich

## Notatki

- augmentacja danych: losowe przycinanie, obracanie, odwracanie, zmiana jasności/kontrastu
- odporność na zakłócenia: dodanie szumu, rozmycie, zmiana kolorów
- analiza kiedy VGG radzi sobie lepiej niż AlexNet, a kiedy gorzej
- czym się różni VGG od AlexNet, jakie są zalety i wady obu architektur
- nie trzeba opowiadać o warstwach splotowych w sieci (to już będzie w przypadku AlexNeta)
- prezentacja:
    - 20 min omawianie metody, augmentacji
    - 10 min omawianie eksperymentów - słabe punkty metody, kiedy działa lepiej a kiedy gorzej itd.

## Plan eksperymentów

### 1. VGG16 vs AlexNet

Porównać architektury:
- głębokość
- liczba parametrów
- skuteczność
- odporność na dane

### 


