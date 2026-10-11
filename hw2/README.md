# Reproduction instructions / Порядок воспроизведения

[English](#english) · [Русский](#русский)

## English

The main file is [colab.ipynb](colab.ipynb). It contains the training program, the unchanged author architecture, data validation, and the code for all of our figures. No separate Python files are required to run it.

### Analyze the saved measurements

Open the notebook and run all cells with `RUN_TRAINING = False`. The [results_snapshot.zip](results_snapshot.zip) archive contains the analysis data. In Colab, it is downloaded from this repository; when running locally, the copy next to the notebook is used. The archive and individual file checksums are verified before the data are read.

Local analysis requires Python, NumPy, Matplotlib, IPython, and a Jupyter environment. Training in this study runs only in Colab on Tesla T4. The notebook’s saved outputs come from sequential execution of the analysis; the training section is not presented as a new locally completed experiment.

The current published snapshot contains all ten completed preliminary networks and 51,480 of 78,000 updates of the first random-half final network. The ten-network mean scores and selected random indices passed independent checks. Final accuracies are not yet available. Missing results are not replaced with zeros, artificial curves, or results from the earlier small experiment.

### Train in Google Colab

1. Open the [notebook in Colab](https://colab.research.google.com/github/Mixlazer/Efficient-NN-HW-2/blob/main/hw2/colab.ipynb) and select a **Tesla T4** accelerator.
2. For a new experiment, set `RUN_TRAINING = True` and leave `RESTORE_ARCHIVE = None`. To resume, first upload a verified checkpoint archive through the Files panel, then enter its path in `RESTORE_ARCHIVE`.
3. Run the preparation, computation checks, and launch cells. A duplicate process is not started if the program is already running.
4. During training, periodically run the **“Сохранить состояние на компьютер”** cell (“Save state to the computer”). Keep the downloaded archive on permanent storage. Colab session storage may be cleared when a session ends; without a backup, the current training state cannot be restored.
5. After all ten preliminary and eight final networks finish, rerun the analysis section. It will independently verify the results and plot the final comparison.

The program saves weights, accumulated updates, and network statistics every 1,950 steps. Completed networks are identified by `complete.json` or `metrics.json` and are not trained again. Their scores and measurements are sufficient to skip them; the restore archive includes weights only for the unfinished network. Training runs in a separate process. The command `experiment_process.terminate()` stops it; restarting resumes from the latest checkpoint.

### Exact settings

A **training step** is one weight update using a batch of images. An **epoch** is one pass through a dataset. **EL2N** is the distance between class probabilities and the correct-answer vector, which contains one for the correct class and zeros elsewhere. We average ten **distances**. The random half is selected again for each repetition; the half selected by mean score is shared by all four repetitions. Equal scores retain their original order.

| Setting | Value |
|---|---|
| Architecture | Authors’ ResNet-18 for small images, 32-bit floating-point arithmetic |
| Data | CIFAR-10: 50,000 training images, 10,000 test images |
| Preliminary scoring | 10 networks × 7,800 steps on the full training set |
| Final methods | Random 25,000 images; 25,000 images with the highest mean EL2N scores |
| Repetitions | 4 per method, numbered 10–13; paired initial weights |
| Duration | 78,000 steps per final network |
| Batch size | 128 images |
| Initial learning rate | 0.1 |
| Momentum | Nesterov rule, coefficient 0.9 |
| Weight decay | 0.0005 |
| Learning-rate decay | Divide by five at steps 23,400, 46,800, and 62,400 |
| Image processing | Normalization, four-pixel reflection padding, a shared crop offset within each batch, independent flips |
| Evaluation | Once after the last step; no best checkpoint is selected using the test set |

Under the authors’ counting convention, 390 steps correspond to almost one epoch of the full dataset. Reducing the dataset does not reduce the step budget: 78,000 steps give approximately 400 passes through half of the data. The entire experiment comprises 702,000 planned weight updates.

### Verifiable evidence

- `score_*/history.jsonl`, `complete.json`, `scores.npy`: history, 7,800 steps, and 50,000 scores for each preliminary network.
- `sorted_to_original_indices.npy`: the mapping between sorted and original images, checked against the labels in the standard CIFAR-10 archive.
- `evaluate_*/`: selected indices, probabilities for ten classes on 10,000 test images, and measured accuracy appear as runs finish.
- `analysis.json`: independent verification results. The `complete` field becomes true only when all ten preliminary and eight final trainings are present.
- `file_manifest.json` and `provenance.json`: published-data checksums and provenance.
- `checkpoint_import.json`: the documented transfer of the first preliminary network’s state after 3,900 steps from the previous experiment.

The sample standard deviation describes variation between repetitions and is calculated with a denominator one smaller than their number. A percentile is a value below which a specified proportion of results lies. The 16th and 84th percentiles, calculated with linear interpolation, describe the spread of four repetitions; they are not a confidence interval.

### Differences and limitations

The authors’ historical environment is replaced with modern Python 3.13.15, JAX 0.11.1, Flax 0.12.10, and NumPy 2.1.3. PyTorch 2.11.0 and Torchvision 0.26.0 are used to read CIFAR-10; training itself uses JAX. Deterministic accelerator operations are enabled. Exact checkpoint continuation was verified on T4.

The author architecture, normalization, image transformations, and weight-update formula from the historical `flax.optim.Momentum` are preserved. The ordering of the standard archive differs from TensorFlow Datasets, as do the random generators for shuffling and transformations. This is not a bitwise repetition of the authors’ historical run.

There is no final training on the full dataset. The comparison can therefore show an advantage of EL2N over a random half, but cannot establish accuracy preservation relative to the full dataset. Other datasets, label errors, transfer between architectures, and the paper’s other studies are not reproduced here.

Original Figure 1 from the [paper](https://arxiv.org/abs/2107.07075v2) is used with attribution. Our own figures are generated in the notebook from the published measurements; preliminary-stage plots are not presented as evidence of a final improvement. The author architecture is distributed under [Apache 2.0](author_LICENSE.txt).

## Русский

Основной файл — [colab.ipynb](colab.ipynb). Он содержит программу обучения, неизменённую архитектуру авторов, проверку данных и код всех собственных графиков. Отдельные файлы Python для запуска не нужны.

### Анализ сохранённых измерений

Откройте блокнот и выполните все ячейки с `RUN_TRAINING = False`. Архив [results_snapshot.zip](results_snapshot.zip) содержит данные для анализа. В Colab он загружается из этого репозитория; рядом с локальным блокнотом используется местная копия. Перед чтением проверяется контрольная сумма архива и каждого файла.

Для локального анализа нужны Python, NumPy, Matplotlib, IPython и среда Jupyter. Обучение в этой работе выполняется только в Colab на Tesla T4. Сохранённые выходные данные блокнота получены при последовательном выполнении анализа; раздел обучения не выдаётся за заново выполненный локально опыт.

В текущем опубликованном снимке завершены все десять предварительных сетей; первая итоговая сеть на случайной половине выполнила 51 480 из 78 000 шагов. Средние оценки десяти сетей и индексы случайного отбора прошли независимую проверку. Итоговых точностей пока нет. Отсутствующие результаты не заменяются нулями, искусственными кривыми или результатами прежнего небольшого опыта.

### Обучение в Google Colab

1. Откройте [блокнот в Colab](https://colab.research.google.com/github/Mixlazer/Efficient-NN-HW-2/blob/main/hw2/colab.ipynb) и выберите ускоритель **Tesla T4**.
2. Для нового опыта установите `RUN_TRAINING = True`, оставив `RESTORE_ARCHIVE = None`. Для продолжения сначала загрузите проверенный архив сохранения через панель «Файлы», затем укажите его путь в `RESTORE_ARCHIVE`.
3. Выполните подготовку, проверку вычислений и запуск. Повторный процесс не запускается, если программа уже работает.
4. Во время работы периодически выполняйте ячейку **«Сохранить состояние на компьютер»**. Храните скачанный архив на постоянном диске. Память сессии Colab может быть очищена при разрыве соединения; без копии текущее обучение не восстановить.
5. После завершения десяти предварительных и восьми итоговых сетей повторите раздел анализа. Он самостоятельно проверит результаты и построит итоговое сравнение.

Программа сохраняет веса, накопленные изменения и статистики сети каждые 1 950 шагов. Завершённые сети определяются по `complete.json` или `metrics.json` и не обучаются заново. Для их пропуска достаточно оценок и измерений; архив восстановления содержит веса только незавершённой сети. Обучение идёт в отдельном процессе. Команда `experiment_process.terminate()` останавливает его; повторный запуск продолжает с последнего сохранения.

### Точные условия

**Шаг обучения** — одно обновление весов по порции изображений. **Эпоха** — один проход по выборке. **EL2N** — расстояние между вероятностями классов и вектором правильного ответа: единица у нужного класса, нули у остальных. Среднее берётся по десяти **расстояниям**. Случайная половина выбирается заново в каждом повторении; половина по средней оценке одинакова для четырёх повторений. При равенстве оценок сохраняется исходный порядок.

| Настройка | Значение |
|---|---|
| Архитектура | Авторская ResNet-18 для маленьких изображений, 32-разрядные числа с плавающей точкой |
| Данные | CIFAR-10: 50 000 обучающих, 10 000 проверочных |
| Предварительное оценивание | 10 сетей × 7 800 шагов на всех данных |
| Итоговые варианты | Случайные 25 000; 25 000 с наибольшей средней оценкой EL2N |
| Повторения | 4 на способ, номера 10–13; начальные веса попарно одинаковы |
| Длительность | 78 000 шагов на итоговую сеть |
| Порция | 128 изображений |
| Начальный размер обновления | 0,1 |
| Накопление прошлых обновлений | Правило Нестерова, коэффициент 0,9 |
| Штраф за большие веса | 0,0005 |
| Снижение размера обновления | В пять раз на шагах 23 400, 46 800 и 62 400 |
| Обработка изображений | Нормализация, зеркальное дополнение на 4 пикселя, общий сдвиг обрезки порции, независимые отражения |
| Проверка | Один раз после последнего шага; лучший результат по проверочной выборке не выбирается |

В авторском отсчёте 390 шагов соответствуют почти одной эпохе полного набора. После сокращения выборки число шагов не уменьшается: 78 000 шагов дают примерно 400 проходов по половине данных. Всего запланировано 702 000 обновлений весов.

### Проверяемые свидетельства

- `score_*/history.jsonl`, `complete.json`, `scores.npy`: история, 7 800 шагов и 50 000 оценок каждой предварительной сети.
- `sorted_to_original_indices.npy`: соответствие упорядоченных и исходных изображений; проверяется по меткам стандартного архива CIFAR-10.
- `evaluate_*/`: по мере завершения появляются индексы, вероятности десяти классов для 10 000 проверочных изображений и измеренная точность.
- `analysis.json`: результаты независимых проверок. Поле `complete` станет истинным только при наличии всех десяти предварительных и восьми итоговых обучений.
- `file_manifest.json` и `provenance.json`: контрольные суммы опубликованных данных и их происхождение.
- `checkpoint_import.json`: документированный перенос состояния первой предварительной сети после 3 900 шагов из прежнего опыта.

Выборочное стандартное отклонение характеризует разброс между повторениями и вычисляется с делителем на единицу меньше их количества. Процентиль — значение, ниже которого находится указанная доля результатов. Используются 16-й и 84-й процентили с линейной интерполяцией, как способ показать разброс четырёх повторений; это не доверительный интервал.

### Отличия и ограничения

Историческая среда авторов заменена современными Python 3.13.15, JAX 0.11.1, Flax 0.12.10 и NumPy 2.1.3. Для чтения CIFAR-10 используются PyTorch 2.11.0 и Torchvision 0.26.0; само обучение выполняется в JAX. Включён фиксированный порядок операций ускорителя. Проверка точного продолжения после сохранения прошла на T4.

Сохранены авторская архитектура, нормализация, преобразования изображений и формула обновления весов исторического `flax.optim.Momentum`. Отличаются порядок стандартного архива вместо TensorFlow Datasets и генераторы перемешивания и преобразований. Это не побитовое повторение исторического запуска авторов.

Отсутствует итоговое обучение на всех данных. Поэтому сравнение может показать преимущество EL2N перед случайной половиной, но не сохранение точности относительно полного набора. Другие наборы, ошибки разметки, перенос между архитектурами и остальные исследования статьи здесь не повторяются.

Оригинальный рисунок 1 [статьи](https://arxiv.org/abs/2107.07075v2) используется с указанием источника. Собственные графики строятся в блокноте из опубликованных измерений; графики предварительного этапа не выдаются за доказательство итогового выигрыша. Архитектура авторов распространяется на условиях [Apache 2.0](author_LICENSE.txt).
