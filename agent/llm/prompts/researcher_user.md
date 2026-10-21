# Задача
$task_brief

Фрагмент описания задачи (task.md):
$task_excerpt

# Профиль данных
$profile

# Анализ ошибок текущего лучшего
$errors

# Текущая лучшая конфигурация ($best_id, score=$best_score)
$best_config

# Журнал
$journal

# Меню
Конфигурация:
- preprocess.cat_encoding: native | ordinal | freq | target (target — OOF target encoding вместо категорий)
- preprocess.missing_flags: true/false — флаги пропусков
- preprocess.rare_min_count: целое N — категории реже N в train объединяются в "__rare__"
- features.builtin: freq_enc (частоты категорий на train+test), target_enc (OOF target encoding в дополнение к категориям), group_stats (mean/std/отклонение числовых признаков внутри категорий; features.group_by и features.group_cols — необязательно, иначе выбираются по важности)
- features.drop: список исходных колонок или fe_XXX
- target.transform: none | log1p (только регрессия; метрика всегда считается в исходной шкале)
- postprocess.class_weights_opt: true/false — множители вероятностей классов под macro F1 (классификация)
- postprocess.clip_to_train_range: true/false — клиппинг регрессии в диапазон train
Модели (model.name: основные параметры и значения по умолчанию):
$models
Зарегистрированные признаки Инженера (включаются через features.custom):
$features

# Состояние
Итерация $iteration из $n_iters, после неё осталось $remaining.
$switch_note

Предложи $n_hyp гипотез в формате JSON.
