# open-xai-materials

Открытые материалы курса **ExplainableAI: Интерпретация моделей от ML до LLM**.

Здесь лежат ноутбуки домашних заданий и туториалов в том виде, в каком их проходят
студенты. Решения и ответы сюда не попадают.

## Ноутбуки

| Блок | | Тема | | |
| --- | --- | --- | --- | --- |
| Модуль 02 | ДЗ 1 | Линейные модели: веса, регуляризация и что они прогнозируют | [RU](https://colab.research.google.com/github/SadSabrina/open-xai-materials/blob/main/homeworks/HW1_linear_models_weights_and_predictions.ipynb) | [EN](https://colab.research.google.com/github/SadSabrina/open-xai-materials/blob/main/homeworks/HW1_linear_models_weights_and_predictions_en.ipynb) |
| Модуль 02 | Туториал | GLM и GAM: веса в правильной шкале | [RU](https://colab.research.google.com/github/SadSabrina/open-xai-materials/blob/main/homeworks/TUTORIAL_GLM_weights_on_the_right_scale.ipynb) | [EN](https://colab.research.google.com/github/SadSabrina/open-xai-materials/blob/main/homeworks/TUTORIAL_GLM_weights_on_the_right_scale_en.ipynb) |
| Модуль 04 | ДЗ 2 | Нелинейные модели: важность признаков от дерева до бустингов | [RU](https://colab.research.google.com/github/SadSabrina/open-xai-materials/blob/main/homeworks/HW2_forest_and_boosting_importances.ipynb) | [EN](https://colab.research.google.com/github/SadSabrina/open-xai-materials/blob/main/homeworks/HW2_forest_and_boosting_importances_en.ipynb) |

| Модуль 08 | ДЗ 3 | Взаимодействие признаков: карты, SHAP interaction values, H-статистика | [RU](https://colab.research.google.com/github/SadSabrina/open-xai-materials/blob/main/homeworks/HW3_feature_interactions.ipynb) | — |
| Модуль 09 | ДЗ 4 | CNN: карты активаций и выученные признаки | [RU](https://colab.research.google.com/github/SadSabrina/open-xai-materials/blob/main/homeworks/HW4_cnn_learned_features.ipynb) | — |
| Модуль 10 | ДЗ 5 | Vanilla Gradients и Gradient × Input | [RU](https://colab.research.google.com/github/SadSabrina/open-xai-materials/blob/main/homeworks/HW5_vanilla_gradients.ipynb) | — |
| Модуль 11 | ДЗ 6 | CAM: карты активации класса | [RU](https://colab.research.google.com/github/SadSabrina/open-xai-materials/blob/main/homeworks/HW6_cam.ipynb) | — |
| Модуль 11 | ДЗ 7 | Guided Backpropagation | [RU](https://colab.research.google.com/github/SadSabrina/open-xai-materials/blob/main/homeworks/HW7_guided_backprop.ipynb) | — |
| Модуль 11 | ДЗ 8 | Grad-CAM и контрфактические карты | [RU](https://colab.research.google.com/github/SadSabrina/open-xai-materials/blob/main/homeworks/HW8_gradcam.ipynb) | — |
| Модуль 12 | ДЗ 9 | Integrated Gradients: аксиомы на практике | [RU](https://colab.research.google.com/github/SadSabrina/open-xai-materials/blob/main/homeworks/HW9_integrated_gradients.ipynb) | — |
| Модуль 12 | ДЗ 10 | DeepLIFT: отклонения активаций от эталона | [RU](https://colab.research.google.com/github/SadSabrina/open-xai-materials/blob/main/homeworks/HW10_deeplift.ipynb) | — |
| Модуль 12 | ДЗ 11 | Layer-wise Relevance Propagation | [RU](https://colab.research.google.com/github/SadSabrina/open-xai-materials/blob/main/homeworks/HW11_lrp.ipynb) | — |

Ссылки открывают ноутбук прямо в Google Colab. Данные ноутбуки тянут по сети, а
недостающие библиотеки ставят сами — готовить ничего заранее не нужно.

## Три репозитория: кто за что отвечает

Правила приняты 05.08.2026. Контент курса живёт в трёх местах, и путать их дорого:
правка, сделанная не в том репозитории, теряется при следующей синхронизации.

```
                 забрали → исправили → вылили обратно
   Степик  ←──────────────────────────────────────────→  open-xai-materials-private
   (русский, боевой)                                     (русский, мастерская)
                                                                  │
                                                                  │ согласованное
                                                                  │ и помеченное готовым
                                                                  ↓
   open-xai-materials                                      open_xai_platform
   (ноутбуки, публичные)  ←── ссылки ───                   (сайт: RU + перевод на EN)
```

### `related/open-xai-materials-private` — мастерская русского контента

Здесь ведётся вся работа над текстом уроков. Порядок работы всегда один:

1. **забрали** — выгрузили блок со Степика (`stepik_scripts/export_content.py`, затем
   `export_module.py`), сделали неприкосновенный снимок в `stepik_snapshots/<дата>/`;
2. **исправили** — вычитка, новые уроки, перестройка структуры, задачи, спойлеры, картинки;
3. **согласовали** — разбор и решения фиксируются в `Curriculum/` репозитория open-xai-org;
4. **вылили обратно** — `push_module.py` отправляет исправленное на Степик.

Всё, что здесь лежит, кроме снимков в `stepik_snapshots/`, **синхронизировано со Степиком**.
Русский язык — единственный. Английского здесь нет и быть не должно.

Сюда же относятся задачи: вычитка, математика, структура блока, перерисовка картинок,
удаление и перестановка шагов на Степике.

### `related/open-xai-materials` — публичные ноутбуки

Открытый репозиторий с домашними заданиями и туториалами: только то, что **проверено и
работает**. Правило простое: **если ноутбук лежит здесь, ссылки на него и со Степика, и с
сайта ведут именно сюда** — не в личный Colab и не на Google Drive.

Ноутбук попадает сюда, когда прогнан целиком, числа в ответах тренажёра сверены с выводом
ячеек, данные доступны, а решения из него убраны. Английские версии живут рядом, с суффиксом
`_en`.

### `open_xai_platform` — сайт

Берёт **согласованный русский контент** из materials-private, переводит на английский и
выкладывает на [open-xai-platform.web.app](https://open-xai-platform.web.app). Здесь живут
контент сайта (`web/content/`), платформа, скрипты сборки и деплоя.

Сюда относятся задачи: импорт готовых модулей, перевод на английский, `check_translation.py`,
нумерация модулей на сайте, вёрстка, деплой.

**Ключевое следствие:** блок не импортируется на сайт, пока он не вычитан и не согласован
по-русски. Сначала Степик и materials-private, только потом сайт и английский.

---

## Что где лежит

- `homeworks/` — ноутбуки, по одному файлу на задание; английские версии с суффиксом `_en`;
- `data/` — наборы данных, которых нет в открытом доступе в другом месте.

Английские версии ДЗ 3–11 пока не готовы — в таблице у них прочерк.
