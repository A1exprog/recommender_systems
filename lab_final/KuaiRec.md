# KuaiRec — контекст проекта рекомендательной системы

## Цель

Необходимо разработать рекомендательную систему на датасете **KuaiRec**.

Основные требования к данным:
- train interactions;
- test/validation interactions;
- user features;
- item features.

Датасет уже скачан полностью.

## Имеющиеся файлы

```text
big_matrix.csv                     ~1.08 GB
small_matrix.csv                   ~406 MB
item_categories.csv                ~113 KB
item_daily_features.csv            ~86 MB
kuairec_caption_category.csv       ~1.96 MB
social_network.csv                 ~7 KB
user_features.csv                  ~744 KB
user_features_raw.csv              ~1.54 MB
video_raw_categories_multi.csv     ~1.72 MB
```

---

## 1. `big_matrix.csv`

**Основной датасет взаимодействий user-item.**

Примерный масштаб:
- 7176 пользователей;
- 10 728 видео;
- ~12.5 млн взаимодействий.

Основные поля:

```text
user_id
video_id
play_duration
video_duration
time
date
timestamp
watch_ratio
```

Главный сигнал:

```text
watch_ratio = play_duration / video_duration
```

`watch_ratio > 1` допустим, например при повторном просмотре.

Можно использовать:
- `watch_ratio` как непрерывный implicit-feedback signal;
- либо определить positive interaction через threshold, например `watch_ratio >= 0.8`.

Для основной модели именно `big_matrix.csv` следует использовать как источник interactions.

---

## 2. `small_matrix.csv`

Почти полностью наблюдаемая user-item матрица:

- ~1411 пользователей;
- ~3327 видео;
- ~4.7 млн interactions;
- заполненность около 99%.

Это важная особенность KuaiRec.

Не следует бездумно считать:

```text
big_matrix = train
small_matrix = test
```

Это разные режимы сбора данных.

Для обычного RecSys pipeline лучше сделать temporal:

```text
big_matrix
    ├── train
    ├── validation
    └── test
```

А `small_matrix` использовать отдельно как дополнительный evaluation benchmark.

---

## 3. `user_features.csv`

Основные признаки пользователей.

Примеры:

```text
user_id
user_active_degree
is_lowactive_period
is_live_streamer
is_video_author

follow_user_num
follow_user_num_range

fans_user_num
fans_user_num_range

friend_user_num
friend_user_num_range

register_days
register_days_range

onehot_feat0
...
onehot_feat17
```

Содержит:
- активность пользователя;
- социальную активность;
- является ли пользователь автором/стримером;
- возраст аккаунта;
- анонимизированные categorical features.

Это основной источник **user features**.

---

## 4. `user_features_raw.csv`

Расширенная/raw-версия признаков пользователей.

Необходимо отдельно посмотреть:

```python
df = pd.read_csv("user_features_raw.csv")

print(df.shape)
print(df.columns.tolist())
display(df.head())
```

После этого определить точный смысл колонок и решить, нужны ли они вместо/в дополнение к `user_features.csv`.

---

## 5. `item_categories.csv`

Основные категориальные признаки видео.

Структура примерно:

```text
video_id | feat
```

где `feat` содержит список категорий/тегов:

```text
video_id = 123
feat = [4, 11, 27]
```

Одно видео может иметь несколько категорий.

Это простой и удобный источник **item features**.

---

## 6. `item_daily_features.csv`

Динамические признаки видео по дням.

Логика:

```text
(video_id, date) -> statistics
```

Содержит различные показатели активности/популярности видео:
- просмотры;
- лайки;
- комментарии;
- shares;
- другие агрегированные показатели.

Позволяет учитывать изменение популярности видео во времени.

### Важно: data leakage

При temporal split нельзя использовать признаки из будущего.

Если interaction произошел в момент `t`, item features должны соответствовать времени `<= t`.

---

## 7. `kuairec_caption_category.csv`

Расширенные content-based признаки видео.

Может содержать:

```text
video_id
manual_cover_text
caption
topic_tag

first_level_category_id
first_level_category_name

second_level_category_id
second_level_category_name

third_level_category_id
third_level_category_name
```

Позволяет использовать:
- caption;
- текст обложки;
- topics/tags;
- иерархические категории.

Можно строить текстовые item embeddings:

```text
caption
+ cover text
+ categories
      ↓
Text Encoder
      ↓
Item Embedding
```

Полезно для hybrid/content-based recommendation и cold start.

---

## 8. `video_raw_categories_multi.csv`

Расширенные raw-категории видео с поддержкой нескольких категорий.

Нужно отдельно исследовать реальные колонки:

```python
df = pd.read_csv("video_raw_categories_multi.csv")

print(df.shape)
print(df.columns.tolist())
display(df.head())
```

После этого решить, использовать ли его вместо или совместно с `item_categories.csv` / `kuairec_caption_category.csv`.

---

## 9. `social_network.csv`

Социальный граф пользователей.

Примерная структура:

```text
user_id | friend_list
```

где `friend_list` представляет список связанных пользователей.

Может использоваться для:
- social recommendation;
- graph-based recommendation;
- GNN.

Для первого baseline этот файл можно не использовать.

---

# Соответствие требованиям проекта

| Требование | KuaiRec |
|---|---|
| Interactions | `big_matrix.csv` |
| Train | temporal split `big_matrix.csv` |
| Validation | temporal split `big_matrix.csv` |
| Test | temporal split `big_matrix.csv` |
| Дополнительный evaluation | `small_matrix.csv` |
| User features | `user_features.csv` |
| Extended user features | `user_features_raw.csv` |
| Item features | `item_categories.csv` |
| Rich item features | `kuairec_caption_category.csv` |
| Extended categories | `video_raw_categories_multi.csv` |
| Temporal item features | `item_daily_features.csv` |
| Social graph | `social_network.csv` |

---

# Рекомендуемый первый pipeline

Не использовать сразу все файлы.

Начальный эксперимент:

```text
                    big_matrix.csv
                          │
                   temporal split
                          │
              ┌───────────┼───────────┐
              ↓           ↓           ↓
            train        val         test


user_features.csv ───────────────┐
                                 │
                                 ├── RecSys model
                                 │
item_categories.csv ─────────────┘
```

`small_matrix.csv` оставить для дополнительного evaluation.

Первую модель имеет смысл строить на:

```text
interactions:
    user_id
    video_id
    watch_ratio
    timestamp

user features:
    user_features.csv

item features:
    item_categories.csv
```

После получения baseline постепенно добавлять:

```text
Baseline
   ↓
+ user features
   ↓
+ item categories
   ↓
+ caption/text embeddings
   ↓
+ temporal item features
   ↓
+ social graph
```

Это позволит провести **ablation study** и понять реальный вклад каждого типа признаков.

Основные RecSys-метрики:

```text
Recall@K
NDCG@K
HitRate@K
Precision@K
```

Особое внимание уделить:
1. корректному temporal train/validation/test split;
2. отсутствию data leakage;
3. определению positive/negative interactions через `watch_ratio`;
4. обработке cold-start пользователей/items;
5. сравнению baseline с моделями, использующими side features.