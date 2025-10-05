# Примеры команд для запуска пайплайна

Ниже собраны типичные команды, которые помогут быстро проверить
работоспособность основных частей проекта. Перед стартом убедитесь, что
структура данных соответствует значениям по умолчанию (см. `--data-root`):

```
<data-root>/
├── slides/          # исходные WSI (.svs, .ndpi)
├── annotations/     # аннотации в JSON (опционально)
└── cache/           # кэш и временные файлы (создаётся автоматически)
```

`train_mil_cv` при первом запуске сам создаст каталог `patch_csvs/` и заполнит
его метаданными патчей.

## 1. Базовые запуски MIL / GNN

Базовая иерархическая MIL с сильными аугментациями и нормализацией окраски:

```bash
python -m train_mil_cv --data-root /path/to/data --output runs/hiermil_strong \
  --model hiermil --encoder vit_b16 --aug strong --stain --epochs 20 \
  --k-folds 5 --device cuda
```

Сравнение ABMIL без аугментаций и с базовыми аугментациями:

```bash
python -m train_mil_cv --data-root /path/to/data --output runs/abmil_none \
  --model abmil --encoder resnet50 --aug none --epochs 15

python -m train_mil_cv --data-root /path/to/data --output runs/abmil_basic \
  --model abmil --encoder resnet50 --aug basic --epochs 15 --no-class-weights
```

Графовая голова GAT поверх мешка патчей (kNN-граф строится автоматически):

```bash
python -m train_mil_cv --data-root /path/to/data --output runs/gat_knn \
  --model gat --encoder convnext_base --aug basic --k-folds 3 \
  --no-oversample --device cuda
```

## 2. Варианты экспериментов для сравнения

Сравнение режимов без/с аугментациями и веса классов можно автоматизировать,
запуская несколько конфигураций подряд. Пример полного набора:

```bash
python -m train_mil_cv --data-root /path/to/data --output runs/abmil_weighted \
  --model abmil --encoder resnet50 --aug basic --epochs 15

python -m train_mil_cv --data-root /path/to/data --output runs/abmil_unweighted \
  --model abmil --encoder resnet50 --aug basic --epochs 15 --no-class-weights

python -m train_mil_cv --data-root /path/to/data --output runs/abmil_noaug \
  --model abmil --encoder resnet50 --aug none --epochs 15

python -m train_mil_cv --data-root /path/to/data --output runs/abmil_augstrong \
  --model abmil --encoder resnet50 --aug strong --stain --epochs 15
```

## 3. Постобработка и сравнение результатов

После завершения нескольких прогонов агрегируйте метрики и постройте
сравнительный график:

```bash
python -m compare_experiments runs/hiermil_strong runs/abmil_none \
  runs/abmil_basic runs/gat_knn --metric f1_macro \
  --output runs/compare_f1_macro.png
```

В каждой папке запуска автоматически сохраняются:

- `summary.csv` — сводные метрики по фолдам;
- `plots/loss_curve.png` и `plots/metrics.png` — кривые обучения;
- `attention/*.csv` — веса внимания MIL для последующей визуализации;
- `explainer/*.png` — карты важности GNN при выборе `--model gcn|gat`.

## 4. Дополнительный патч-классификатор

Для быстрой оценки патчей в отрыве от MIL можно запустить вспомогательный
скрипт кросс-валидации линейной головы:

```bash
python -m train_patchclf_cv
```

Скрипт ожидает CSV-файлы патчей в `data/wss1_v2/out/train` (путь можно
отредактировать вверху файла) и сохраняет кривые потерь в `data/images/`.

---

Полезные флаги:

- `--verbose debug` — расширенный логинг через loguru;
- `--no-mixed-precision` — отключить AMP, если нужно запускать только на CPU;
- `--no-oversample` — выключить балансировку классов на уровне мешков.
