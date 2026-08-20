# FP8.md — FP8-хранение замороженной базы (QLoRA-стиль) для SFT

Дата: 2026-08-12, обновлено 2026-08-14. Статус: **реализовано, протестировано
(23 CPU-теста), полный прогон v9 завершён** (4908 шагов / 7.3 ч на 4 GPU,
8753 non-padding ток/с — §6). Замечания аудита по коду закрыты; числа §5 — по
итогам честного ре-бенчмарка.

Смежные документы:
- [PERF_PLAN.md](PERF_PLAN.md) — **активный план ускорения** и опорные факты по
  железу/памяти/fill (§8 этого файла сведён к ссылке на него);
- [FP8_REVIEW.md](FP8_REVIEW.md) — ревизия 2026-08-14: замеры стоимости
  dequant, блокер W8A8, разбор устаревших утверждений;
- `trash/FP8_CHECK.md` — архив: аудит реализации, все фазы плана исправлений
  закрыты;
- `trash/ANALYSYS.md` — архив: журнал расследования v7 → v8 (packing).

---

## 1. Мотивация и гипотеза

Исходная точка: v8 (bf16, LoRA r=128, packing, grad checkpointing, чанк 4096)
даёт ~3700 полезных ток/с на 2× H100-NVL-94GB. Узкое место
([PERF_PLAN.md](PERF_PLAN.md) §2.4) — gradient checkpointing: он обязателен в
bf16 (без него активации GatedDeltaNet не влезают: ~38 GiB уже при 2048
токенах), но рекомпутация forward стоит ~31% времени шага.

**Гипотеза (Г6).** База полностью заморожена (учатся только LoRA-адаптеры),
значит её веса — константы forward'а. Если держать их в fp8 (≈½ памяти bf16),
освобождается ~30 GiB/карту, и gradient checkpointing можно выключить,
устранив рекомпутацию. Ожидаемый выигрыш: +20–50% tok/s.

Официальный источник fp8-весов: чекпоинт **`Qwen/Qwen3.5-27B-FP8`** —
квантизация сделана самим вендором (не наша самодеятельность), качество
эквивалентно bf16 по публичным бенчмаркам Qwen.

### Почему это должно работать с точки зрения качества

- Адаптеры обучаются end-to-end поверх квантизованного forward'а и
  компенсируют фиксированное смещение базы (механизм QLoRA: даже NF4/int4
  показывают паритет с bf16, а E4M3 block-128 существенно точнее).
- Замороженным весам градиенты не нужны — quantization-aware должен быть
  только forward; dgrad по входу считается точно в bf16.
- Активации НЕ квантуются (все вычисления bf16) — градиентный сигнал чище,
  чем при «настоящем» W8A8-обучении.

---

## 2. Что представляет собой FP8-чекпоинт (ФАКТ, прочитано из config.json)

- `quant_method: "fp8"` — нативный формат HF (finegrained fp8),
  **блочная 128×128** квантизация весов E4M3, `activation_scheme: dynamic`.
- В bf16 оставлены (`modules_to_not_convert`): `lm_head`, `embed_tokens`,
  `conv1d`/`in_proj_a`/`in_proj_b` слоёв GatedDeltaNet, visual-башня, `mtp.fc`.
- Квантизованы (становятся `FP8Linear`): все MLP (gate/up/down) всех 64 слоёв,
  q/k/v/o_proj 16 full-attention слоёв, `in_proj_qkv`/`in_proj_z`/`out_proj`
  48 linear-attention слоёв. Итого **400 квантизованных Linear** — среди них
  все 7 LoRA-таргетов v8.
- Размер: 28.8 GiB (веса + scales) против ~55.6 GiB bf16.
- Веса лежат в HF-кэше: `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B-FP8`.

---

## 3. Почему нельзя обучать «как есть» (ФАКТ по коду transformers 5.9)

1. `FineGrainedFP8HfQuantizer.is_trainable = False`
   (`transformers/quantizers/quantizer_finegrained_fp8.py:142`).
2. Forward `FP8Linear` вызывает Triton-ядра (DeepGEMM/finegrained-fp8)
   **без autograd-обёртки** (`transformers/integrations/finegrained_fp8.py:276`,
   `w8a8_fp8_matmul`) — backward через него умирает.
3. `Trainer.__init__ → validate_quantization_for_training` явно отказывается
   обучать модели с «необучаемой» квантизацией (ValueError).

Отсюда дизайн: **не трогать W8A8-путь HF, заменить forward квантизованных
слоёв на свой** — dequant в bf16 внутри autograd. Это и есть QLoRA-подход:
хранение квантизованное, вычисления полные.

---

## 4. Реализация

### 4.1. `ruadapt/training/core/fp8.py` (новый модуль)

> Аудит нашёл в этом модуле баг с памятью (мёртвый `save_for_backward`,
> пиннил 8.25 GiB) и две скрытые мины (алиасинг `mul_`, обнуление
> субнормальных блоков в удалённой `quantize_fp8_block`). Всё исправлено —
> см. `trash/FP8_CHECK.md` §3 и §6.
>
> **Известное ограничение (Д6, принято).** `_make_dequant_forward` создаёт
> замыкание на инстанс ⇒ пропатченная модель не сериализуется
> (`pickle`/`torch.save(model)` падают; `deepcopy` работает). Безвредно в
> текущем режиме — сохраняются только PEFT-адаптеры. При необходимости
> лечится модульным callable вместо замыкания
> ([PERF_PLAN.md](PERF_PLAN.md) §5.1).

| Компонент | Назначение |
|---|---|
| `dequantize_fp8_block(w, scale, block, dtype)` | Обратная операция; bf16-арифметика (exact fp8→bf16 каст + scale, ≤0.2% доп. ошибки от fp32→bf16 скейлов), fp32-путь для точных тестов. `to(dtype, copy=True)` — источник не мутирует даже при совпадении dtype |
| `_DequantLinearFunction` (autograd.Function) | forward: dequant → `F.linear`; backward: **повторный dequant из fp8**, градиент по входу `dy @ w`. `saved_tensors` пуст: вход не сохраняется назад не нужен, вес ре-деквантится (экономия ~8 GiB активаций — Д1) |
| `patch_fp8_linear_for_training(mod)` | Инстанс-патч forward'а одного FP8Linear (идемпотентен, маркер `_ruadapt_fp8_dequant`). Raise на слой с bias (молча необучаем — Д10) |
| `apply_fp8_storage(model)` | Патчит все 400 квантизованных FP8Linear + **замораживает fp8-параметры и scales** (HF оставляет их requires_grad=True) |
| `set_dequant_compiled(bool)` | Включает `torch.compile(dequant)` — fused cast+scale в одном проходе по памяти |
| `verify_fp8_storage(model)` | Fail-fast: нет непропатченного квантизованного FP8Linear (иначе backward умрёт в Triton-ядре), нет обучаемого fp8-параметра; возвращает сводку GiB по группам fp8_frozen / bf16_frozen / trainable |

**Ключевой дизайн-момент (почему autograd.Function, а не просто
`F.linear(x, dequant(w))`).** Обычный `F.linear` заставляет autograd сохранить
де-квантированный bf16-вес **каждого** слоя до backward (он нужен для dgrad по
входу). Это +45 GiB резидентной памяти — вся экономия fp8 исчезает, и прогон
падает с OOM (измерено: 91.6 GiB занято, аллокация 340 MiB не прошла).
`_DequantLinearFunction` сохраняет для backward только input, а вес
ре-деквантит из fp8 в момент backward — цена: один дополнительный dequant-проход
на слой за шаг.

**Почему dequant в bf16, а не fp32.** Промежуточный fp32-тензор удваивает
трафик памяти (измерено −8%, 2948→3181 ток/с при переходе). Точность:
fp8→bf16 каст точен (мантисса 3⊂8 бит), округление скейла fp32→bf16 добавляет
≤0.2% относительной ошибки — пренебрежимо на фоне собственной ошибки fp8 (~3–6%).

### 4.2. Конфиг (`ruadapt/training/config/schema.py`)

```jsonc
"model": {
    "model_name_or_path": "Qwen/Qwen3.5-27B-FP8",
    "text_only": true,           // как в v8
    "fp8_storage": true,         // включить FP8-хранение + патчи
    "fp8_compile_dequant": true  // torch.compile для dequant
}
"training": {
    "gradient_checkpointing": false  // ради этого всё и затевалось
}
```

### 4.3. Загрузка (`ruadapt/training/core/model.py`)

- `_build_fp8_quantization_config`: чекпоинт отдаёт `quantization_config`
  **dict'ом** (не объектом) — чтение через `_qc_get` (dict/объект); require
  `quant_method == "fp8"`, иначе raise (fail-fast).
- **Ловушка text-only загрузки.** Исключения в чекпоинте записаны с префиксом
  `model.language_model.*`, а у `Qwen3_5ForCausalLM` имена без него;
  `should_convert_module` матчит паттерны от начала имени ⇒ без переписывания
  `in_proj_a/in_proj_b` (nn.Linear) были бы заменены на FP8Linear и загрузка
  весов сломалась. `_rewrite_modules_to_not_convert` переписывает имена
  (проверено на реальных 259 именах чекпоинта: исключения остаются,
  все 7 LoRA-таргетов конвертируются).
- Порядок операций: from_pretrained → `apply_fp8_storage` → verify → печать
  сводки памяти — строго **до PEFT**, чтобы адаптеры обернули уже пропатченные
  слои.

### 4.4. Точки отказа, закрытые в `ruadapt/training/train.py`

1. **Блокировка Trainer'а**: `validate_quantization_for_training` падает на
   fp8 (`is_trainable=False`). Обход — выставляем
   `model._hf_peft_config_loaded = True` в fp8-ветке (PEFT действительно
   загружен; это тот самый флаг, который проверяет transformers; PEFT 0.19 сам
   его не выставляет). Дополнительно require `lora.peft=true`: замороженная
   fp8-база без адаптеров обучать нечего.
2. **Повторный `verify_fp8_storage` после PEFT** — адаптеры оборачивают
   пропатченные слои, `base_layer.forward` обязан остаться патченным
   (PEFT вызывает `base_layer(x)`).
3. **Warning при `fp8_storage + gradient_checkpointing`**: с ckpt dequant
   исполняется 3 раза за шаг (forward + рекомпутация + backward) вместо 2 —
   комбинация не имеет смысла (ни по памяти, ни по скорости).

### 4.5. Совместимость со стеком v8 (проверено smoke-прогоном)

- **PEFT 0.19**: адаптеры кастятся к `base_layer.weight.dtype` (fp8), затем
  `cast_adapter_dtype` апкастит fp8→fp32 (`UPCAST_DTYPES` включает
  float8_e4m3fn) ⇒ итоговый dtype адаптеров **fp32 — ровно как в v8**;
  lora-ветка forward'а считает в dtype адаптеров, результат кастится обратно —
  численно тот же режим, что над bf16-базой.
- **Liger**: инстанс-патч MLP (`LigerQwen3MoeSwiGLUMLP`) вызывает
  `gate_proj/up_proj/down_proj` как модули → попадает в пропатченные forward'ы;
  `lce_forward` использует `lm_head` (не квантизован). Проверено в рантайме:
  `[liger-check] base forward: lce_forward`, MLP — LigerQwen3MoeSwiGLUMLP.
- **FLA/GatedDeltaNet**: `in_proj_qkv/in_proj_z/out_proj` квантизованы и
  пропатчены, `conv1d/in_proj_a/in_proj_b` остаются bf16 nn.Linear — backward
  через всё это работает (smoke прошёл, градиенты конечные).
- **DDP/fused AdamW**: обучаемые параметры только fp32-адаптеры (637.5M,
  2.32%) — синхронизация и оптимизатор как в v8.
- **Сохранение**: `trainer.save_model`/`SavePeftModelCallback` пишут только
  адаптеры — fp8-база не сохраняется, чекпоинты того же формата, что v8.

### 4.6. Попутно: ранговая подготовка данных

В smoke оба ранка параллельно готовили идентичные датасеты (в SFT-фабрике,
в отличие от CPT `ensure_packed`, защиты не было — бесполезный CPU×N и гонка
за кэш-файлом). Добавлен `main_process_first` в
`ruadapt/training/core/distributed.py`: ранк 0 готовит и наполняет кэш,
остальные ждут `dist.barrier()` и читают из кэша. Обёрнута сборка
train/eval-датасетов в `train.py`.

---

## 5. Измерения

### 5.1. Честный ре-бенчмарк (2026-08-13, 2× GPU 2–3)

Проведён после фикса Д1 (убран мёртвый `save_for_backward`). Все прогоны на
**одних данных** (train, 380 934 сэмпла), steady-state окно шагов 20→60
(60 шагов в каждом прогоне), метрика — **non-padding ток/с суммарно на 2 GPU**
(по кумулятивным `train_runtime`/`num_input_tokens_seen` из per-step логов
Trainer; без attention_mask метрика `non_padding` деградирует в
`input_ids != pad_token_id`, [PERF_PLAN.md](PERF_PLAN.md) §2.6).

| Прогон | чанк | с/шаг | **non-pad ток/с** | computed ток/с | пик, GiB/карту |
|---|---|---|---|---|---|
| v8: bf16, ckpt ON | 4096 | 6.39 | 3695 | 3843 | 69.6 |
| v8: bf16, ckpt ON | 2048 | 3.38 | 3332 | 3637 | — |
| v9: fp8, ckpt OFF | 2048 | 5.73 | 3902 | 4291 | 73.6 |
| v9: fp8, ckpt OFF | 3072 | 7.90 | 4418 | 4665 | 89.3 |
| v9: fp8, ckpt OFF, 4 GPU (бенч) | 3072 | 5.31 | 8856 | 9256 | без OOM |
| **v9: fp8, ckpt OFF, 4 GPU (полный прогон)** | 3072 | 5.34 | **8753** | — | без OOM |

Fill по фактической подготовке: 2048 → 0.9246 (40 282 чанка), 3072 → 0.9491
(26 162), 4096 → 0.9599; `dropped_long=0` всюду.

Читается так:

1. **Якорь воспроизведён**: v8@4096 = 3695 против 3664 в аудите (шум ~1%).
2. **Механизм подтверждён like-for-like**: при равном чанке 2048 v9 даёт
   **+17.1%** (3902 против 3332); на микробатче 0.96 против 1.13 с — выключение
   ckpt ускоряет одинаковую работу на ~15%.
3. **Фикс Д1 высвободил больше заявленного**: пик v9@2048 — 73.6 GiB против
   83.8 до фикса (−10.2 GiB при паспортных 4.25–8.25). Именно это сделало
   возможным чанк 3072 без ckpt.
4. **Итог к v8: +19.6%** (4418 против 3695, non-padding, одни данные) — нижняя
   граница гипотезы Г6 (+20–50%), которую аудит считал недостижимой (+6.1%).
5. **Почему не 4096**: при 3072 занято 89.3 из ~94 GiB; активации GatedDeltaNet
   без ckpt растут ~линейно с длиной, 4096 потребовал бы ~103 GiB — OOM.
   3072 — фактический максимум; сверху остаётся ~4.5 GiB запаса, формы чанков
   унифицированы (фрагментация форм отсутствует).
6. **Почему не +50%**: v8 уже работает у измеренного compute-потолка
   bf16-гибрида ([PERF_PLAN.md](PERF_PLAN.md) §2.3); память перестала быть
   ограничением, дальнейший рост — только через смену режима вычислений
   ([PERF_PLAN.md](PERF_PLAN.md) §3.3).
7. **Бенч подтверждён полным прогоном**: 8753 non-pad ток/с в окне 20→60 и 8752
   в окне 100→1000 против 8856 в коротком бенче (−1.2%, в пределах шума и
   стоимости периодического eval).

Численная стабильность: loss/grad-динамика во всех бенч-прогонах здоровая
(loss 0.88→0.44 за 60 шагов у v9), NaN нет. Сравнение loss-кривых v8/v9
напрямую некорректно (разные веса базы) — только по eval-метрикам полного
прогона (§6).

### 5.2. Первоначальные smoke-прогоны (2026-08-12, историческая справка)

> ⚠️ Головной итог этих прогонов «+15–20% к v8» был **некорректен** (сравнение
> computed-токенов fp8 с non-padding-токенами v8, разные данные и чанки;
> like-for-like давал +6.1% — `trash/FP8_CHECK.md` §4). Таблица сохранена как
> источник данных по лестнице dequant-оптимизаций; актуальные числа — в §5.1.

Метрика: computed ток/с суммарно на 2 GPU; val-данные как train
(478 сэмплов, fill 0.9177), 20 шагов.

| Конфиг | ток/с (2 GPU) | пик, GiB/карту |
|---|---|---|
| **v8 эталон**: bf16, ckpt ON, чанк 4096 | 3634 | ~62 |
| fp8, ckpt ON, чанк 2048 | 2019 | 45.2 |
| fp8, ckpt OFF, чанк 2048, dequant fp32 | 2948 | 82.8 |
| fp8, ckpt OFF, чанк 2048, dequant bf16 in-place | 3181 | 82.4 |
| fp8, ckpt OFF, чанк 2048, dequant torch.compile | ~4350 (5.65 с/шаг) | 83.8 |

Отсюда: выключение ckpt при равном чанке даёт +46%; bf16-арифметика dequant
ещё +8%, torch.compile ещё +37%. Итоговый профиль v9 — bf16 in-place + compile.

---

## 6. Качество: результат полного прогона (закрыто 2026-08-14)

Полный прогон `sft_v9_fp8_4gpu` завершён: 4908 шагов (3 эпохи), 7.28 ч,
loss 0.806 → 0.150. Источник — `trainer_state.json` в
`/workdir/data/models/Qwen3.5-27B-FP8_..._packed_4gpu_v9fp8/checkpoint-4908/`.

| Шаг | 500 | 1000 | **1500** | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 4908 |
|---|---|---|---|---|---|---|---|---|---|---|
| eval_loss | 0.5461 | 0.5248 | **0.5171** | 0.5322 | 0.5241 | 0.5189 | 0.5821 | 0.5882 | 0.5870 | 0.5876 |

Выводы:

1. **fp8-хранение не мешает обучению**: eval_loss нормально спадает, NaN нет,
   скорость стабильна все 7 часов. Ожидание §1 (QLoRA-механизм; официальный
   fp8-чекпоинт эквивалентен bf16 у Qwen) не опровергнуто.
2. **Проблема не в fp8, а в расписании**: минимум на шаге 1500 (≈1 эпоха),
   далее +0.07 и плато ⇒ 3 эпохи переобучают. При `save_steps=500,
   save_total_limit=2` сохраняется только хвост ⇒ **лучший чекпоинт физически
   отсутствует**. Решение отложено, см. [PERF_PLAN.md](PERF_PLAN.md) §3.1.
3. **Оговорка остаётся**: loss-кривая смещена относительно v8 (другие веса
   базы) — сравнивать версии только по eval-метрикам, не по абсолютному loss.
   Прямое сравнение eval v8 vs v9 не проводилось.

## 7. Merge для сервинга

> Переписано по итогам аудита (`trash/FP8_CHECK.md` §5): прежняя версия
> переизобретала нативный dequant-путь transformers.

PEFT `merge()` делает `weight += delta` in-place — в fp8 это невозможно.
Есть два пути; **фактически применён второй**.

### 7.1. Нативный dequant-на-загрузке (описан, не применялся)

Ручная ре-квантизация не нужна: в transformers есть
`FineGrainedFP8Config(dequantize=True)` (`quantizer_finegrained_fp8.py:158`,
`Fp8Dequantize` в `integrations/finegrained_fp8.py:933`). Процедура:

1. Загрузить базу с де-квантизацией на лету **через мультимодальный класс**:
   `Qwen3_5ForConditionalGeneration.from_pretrained("Qwen/Qwen3.5-27B-FP8",
   quantization_config=FineGrainedFP8Config(dequantize=True))` — получается
   bf16-модель, точность проверена (max rel diff 0.0 к ручной де-квантизации).

   > ⚠️ **Ловушка**: тот же вызов через `Qwen3_5ForCausalLM` (text-only)
   > **молча даёт сломанные веса** — ключи чекпоинта несут префикс
   > `model.language_model.`, конвертерная цепочка text-класса не связывает их
   > со scale-тензорами (в LOAD REPORT они видны как UNEXPECTED), и веса
   > остаются без скейлов (замер: rel diff ~4600×). Для merge использовать
   > только мультимодальный класс загрузки.
2. Применить и смержить адаптеры штатными средствами PEFT.
3. Для сервинга: либо bf16-модель, либо ре-квантизация трансформерсовским
   `Fp8Quantize` (`integrations/finegrained_fp8.py:865`) — он корректнее
   удалённой локальной копии (клампит перед кастом, обрабатывает 3D-формы
   экспертов).

### 7.2. Фактически применённый путь: merge адаптера в bf16-базу

Проще и не требует нового кода:

1. в `adapter_config.json` `base_model_name_or_path` подменён с
   `Qwen/Qwen3.5-27B-FP8` на `Qwen/Qwen3.5-27B`;
2. обычный `scripts/merge_lora.py` смержил адаптер в **bf16-базу**.

Проверено на диске: `checkpoint-4908/adapter_config.json` содержит FP8-базу, а
корневой `adapter_config.json` — bf16; merged-модель — bf16
`Qwen3_5ForConditionalGeneration` (~53 GB, без `quantization_config`).
Результаты лежат в `/workdir/data/models/` как `..._v9fp8_merged` и
`..._v9fp8_apc{1.5,2.0,2.5}_merged`.

> ⚠️ **Непроверенное допущение о качестве.** Адаптер обучался компенсировать
> forward *квантованной* базы, а смержен в *неквантованную*. Разница баз —
> это собственная ошибка fp8 (~3–6% на вес). Ожидание безвредности разумное
> (смещение мало и в основном не выучено адаптером), но **не измерено**.
> Дешёвая проверка — parity-тест в [PERF_PLAN.md](PERF_PLAN.md) §3.5.

По ключам путь корректен: адаптеры v9 того же формата, что v8 (fp32, те же
модули), и `text_only`-фикс ключей применяется как обычно
(`python -m ruadapt.utils.adapter --verify`).

---

## 8. Ограничения и продолжения

Активный план ускорения вынесен в **[PERF_PLAN.md](PERF_PLAN.md)** (там же
опорные факты по железу, памяти и fill). Кратко, что относится к fp8-пути:

1. **W8A8 FP8-compute** — единственный крупный резерв (+25–33% ток/с за счёт
   ускорения GEMM, а не за счёт устранения dequant). **Заблокирован**: блочный
   рецепт `(1x128, 128x128)` требует cublasLt ≥ 12.9, стек на cu128
   ([PERF_PLAN.md](PERF_PLAN.md) §3.3).
2. **Стоимость dequant мала**: компилированный dequant — 25 мс/проход, ~3.7%
   времени шага; накладные расходы всего fp8-пути против bf16 `F.linear` —
   +7.0%. Устранение dequant даёт ≤4% и рычагом не является
   ([FP8_REVIEW.md](FP8_REVIEW.md) §3.1).
3. **Селективный gradient checkpointing** (чекпоинт только 48 GDN-слоёв) —
   шанс поднять чанк с 3072 к 4096 (fill +1.1%); замеры не проводились,
   приоритет низкий ([PERF_PLAN.md](PERF_PLAN.md) §3.4).
4. `ddp_find_unused_parameters: false` — микро-оптимизация (+1–3%), совместима
   с v9, в конфигах пока не выставлена ([PERF_PLAN.md](PERF_PLAN.md) §3.2).
5. FP8-путь не менял семантику eval: eval идёт непакованным
   (DynamicPadCollator) — как в v8.

---

## 9. Как запустить

```bash
# Тесты (CPU):
pytest tests/training/test_fp8_storage.py -v          # 23 теста

# Тест численного паритета (GPU, 27B, ~5 мин, отключён по умолчанию):
RUN_FP8_PARITY=1 pytest tests/training/test_fp8_parity.py -v

# Smoke (2 GPU):
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=29601 \
  -m ruadapt.training.train --config configs/smoke/sft_v9_fp8.json

# Полный прогон (2 GPU):
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 \
  -m ruadapt.training.train --config configs/sft_v9_fp8_2gpu.json

# Полный прогон (4 GPU, рабочий конфиг; факт полного прогона: 8753 non-pad ток/с, ×3.9 к v7):
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  -m ruadapt.training.train --config configs/sft_v9_fp8_4gpu.json
```

Нюансы запуска:
- повторный smoke с тем же `output_dir` возобновится из сохранённого
  состояния (`get_last_checkpoint`) — для чистого замера нужен новый dir;
- torch.compile dequant греется ~1–2 минуты (7–8 уникальных форм весов),
  устойчивый режим с шага ~10–15;
- дефолт проекта — bf16/v8; fp8 включается явно `fp8_storage` и
  `fp8_compile_dequant` в `model`-секции конфига.

## 10. Состав изменений

| Файл | Изменение |
|---|---|
| `ruadapt/training/core/fp8.py` | Новый: quant/dequant, autograd.Function, патчи, verify, compile-переключатель |
| `ruadapt/training/config/schema.py` | `ModelConfig.fp8_storage`, `ModelConfig.fp8_compile_dequant` |
| `ruadapt/training/core/model.py` | fp8-ветка загрузки, rewrite исключений, патч+verify до PEFT |
| `ruadapt/training/train.py` | `_hf_peft_config_loaded`, verify после PEFT, warning ckpt+fp8, `main_process_first` для датасетов |
| `ruadapt/training/core/distributed.py` | `main_process_first` |
| `tests/training/test_fp8_storage.py` | Новый: 23 CPU-теста |
| `configs/smoke/sft_v9_fp8.json` | Smoke: val-данные, 20 шагов, ckpt off, чанк 2048, compile on |
| `configs/smoke/sft_v9_fp8_ckpton_diag.json` | Диагностика: то же с ckpt on |
| `configs/sft_v9_fp8_2gpu.json` | Полный прогон: чанк 3072, accum 8 (~47k полезных ток/шаг как v8), `ddp_timeout=7200` |
| `configs/sft_v9_fp8_4gpu.json` | Рабочий конфиг полного прогона: чанк 3072, accum 4 |
| `tests/training/test_fp8_parity.py` | Новый: численный паритет fp8-storage vs `dequantize=True` (GPU, gate `RUN_FP8_PARITY=1`) |

Исправления по итогам аудита (`trash/FP8_CHECK.md`, фазы 1–4):
убран мёртвый `save_for_backward` (Д1, −10.2 GiB в замере), алиасинг `mul_`
закрыт `to(dtype, copy=True)` (Д3), bias закрыт fail-fast'ом (Д10),
`quantize_fp8_block` удалён (Д4/Д5), печать сводки памяти перенесена за PEFT
(Д8), честный ре-бенчмарк (§5.1) и чанк 3072 в полном конфиге (Д9).
Открытым остаётся только Д6 (не пиклится пропатченный модуль) — принят как
известное ограничение (§4.1).
