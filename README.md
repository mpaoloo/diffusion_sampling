# Методы генерации из диффузионных моделей

Baseline DDIM на CIFAR-10 и план сравнения методов сэмплирования.

## Отчёт и результаты

[Отчёт, 6 страниц](report/report.pdf) · [Текст отчёта](report/report.md)

Использована предобученная модель `google/ddpm-cifar10-32`: DDIM, 50 шагов,
η = 0, float32, батч 16. Измерения выполнены на Apple M1 с 16 ГиБ памяти, MPS.

- Выборка: 1000 сгенерированных и 1000 test-изображений CIFAR-10.
  Реальная выборка содержит по 100 изображений каждого класса.
- FID: **57,440** (Inception-v3-compat, 2048 признаков).
- NFE: **50** на изображение.
- Время денойзинга: **1545,82 с**, в среднем **1,546 с** на изображение.
- Автоматические проверки: **7 тестов пройдены**.

FID на выборках 1000/1000 используется как предварительная оценка и не сопоставляется
напрямую с опубликованными результатами на 50 000 изображений. На следующем этапе
планируется сравнить DDIM, DPM-Solver++ 2M и Heun при одинаковом NFE и подобрать расписание шума.

## Запуск

Требуется Python 3.12. Команды выполняются из корня репозитория.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python download_assets.py
.venv/bin/python -m pytest -q test_contract.py
.venv/bin/python run.py sanity --device mps
.venv/bin/python run.py baseline --device mps
.venv/bin/python run.py evaluate --device mps
.venv/bin/python make_figures.py
```

Для CPU используется `--device cpu`. Параметры эксперимента заданы в `config.json`.
Завершённые результаты сохраняются; прерванная генерация продолжается с последнего батча.

## Файлы проекта

- `run.py` — проверка модели, генерация и расчёт FID, NFE и времени.
- `download_assets.py` — загрузка весов и данных с проверкой контрольных сумм.
- `test_contract.py` — проверки параметризаций, шага DDIM, NFE и сохранения результатов.
- `make_figures.py` — построение графиков и сеток изображений.
- `results/baseline_metrics.json`, `results/fid_metrics.json` — итоговые метрики.
- `results/batch_metrics.csv`, `results/timing_repeats.csv`, `results/logs/` — измерения и логи.
- `results/initial_noise.npy`, `results/samples.npy` — начальные шумы и полученные изображения.
- `results/reference_indices.json`, `results/features_*` — реальная выборка и признаки Inception.
- `results/figures/` — примеры изображений и графики.

## Данные и веса

Модель: [google/ddpm-cifar10-32](https://huggingface.co/google/ddpm-cifar10-32),
лицензия Apache 2.0, ревизия `267b167dc01f0e4e61923ea244e8b988f84deb80`.

Данные: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), test-часть из
[uoft-cs/cifar10](https://huggingface.co/datasets/uoft-cs/cifar10), ревизия
`0b2714987fa478483af9968de7c934580d0bb9a2`. В карточке зеркала лицензия указана как `unknown`.

Веса Inception: [torch-fidelity](https://github.com/toshas/torch-fidelity/releases/tag/v0.2.0).
URL и SHA256 сохранены в `results/assets.json`. Данные и веса загружаются в `cache/`,
который не включён в репозиторий. Дообучение модели не проводилось.
