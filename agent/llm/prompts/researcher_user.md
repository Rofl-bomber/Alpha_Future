# Задача
$task_brief

Фрагмент описания задачи (task.md):
$task_excerpt

# Профиль данных
$profile

# Текущая лучшая конфигурация ($best_id, score=$best_score)
$best_config

# Журнал
$journal

# Фаза: $phase_title
$phase_goal
Разрешённые типы гипотез (kind): $phase_kinds
Разрешённые ключи патча: $phase_keys
$phase_menu
Зарегистрированные признаки Инженера (включаются через features.custom):
$features

# Состояние
Итерация фазы $phase_iter из $phase_max. Порог принятия: средняя дельта по фолдам > $noise_k·SE и улучшение минимум на $min_folds из $n_folds фолдов.
$switch_note

Предложи $n_hyp гипотез в формате JSON.
