# План исправлений и унификации дообучения RuAdapt

Активная задача — единая, понятная реализация подготовки batches для CPT и SFT
с четырьмя явно заданными режимами. Все четыре режима должны поддерживать
Qwen3 и Qwen3.5 с `sdpa`, `flash_attention_2` и `flash_attention_3` на совместимом
оборудовании. Поддержка должна включать правильные forward, loss, backward,
микробатчи и возобновление обучения, а не только успешную загрузку модели.

План фиксирует реализацию, её проверки и оставшуюся аппаратную приёмку. Текущее поведение и ограничения кода
описаны в [документации packing и attention](../docs/PACKING_AND_ATTENTION.md).
Ниже отдельно сохранены результаты уже выполненных исправлений.
Следующая самостоятельная задача — [документация всего проекта](#документация-проекта),
включая справочник конфигурации, основные модули и концепции.

## Обязательные режимы

| ID | Задача и режим | Семантика | Текущее состояние |
| --- | --- | --- | --- |
| R1 | CPT с packing без изоляции | Тексты образуют поток, разрезаемый на блоки. Внутри блока разрешён контекст через границы исходных документов | Реализован общий интерфейс, явный новый stream сохраняет хвост; legacy путь сохранён |
| R2 | SFT без packing | Один диалог — один dataset sample. Несколько samples могут образовывать обычный padded batch | Реализован общий конфиг, padded batches сохранены |
| R3 | CPT с packing и изоляцией | Каждый документ или явно выделенный фрагмент независим; labels не создают предсказаний через его границу | Реализован через общий packer/collator |
| R4 | SFT с packing и изоляцией | Каждый диалог независим; сохраняются маски ролей и прочая разметка labels | Общий путь с CPT; Qwen3/Qwen3.5, логический B > 1, SDPA/FA2 проверены |

R1 использует обычный causal attention модели. Он не требует блочной маски,
сброса позиций на границах исходных документов или сброса GatedDeltaNet между
ними. Поддержка FA2/FA3 в этом режиме не означает включения document isolation.
Между отдельными блоками не переносится cache или recurrent state.

R2 не ограничивает `per_device_train_batch_size` единицей: «один пример — один
sample» относится к dataset, а не к числу примеров в batch. Padding не участвует
в loss. Диалог остаётся единой контекстной последовательностью.

R3 и R4 используют один общий механизм document packing. Отличается подготовка
примеров и исходных labels. Преобразование stream CPT в isolated CPT выполняется
только явным выбором режима: это изменение контекста обучения.

## Матрица поддержки

«Три FA» в требованиях означают три значения `attn_implementation`: SDPA,
FlashAttention 2 и FlashAttention 3. SDPA — интерфейс выбора kernels, а не версия
FlashAttention; его поддержка обязательна, включая изолированные режимы.

| Режим | Qwen3 SDPA | Qwen3 FA2 | Qwen3 FA3 | Qwen3.5 SDPA | Qwen3.5 FA2 | Qwen3.5 FA3 |
| --- | --- | --- | --- | --- | --- | --- |
| R1 CPT stream | Проверено | Проверено | Hardware skip | Проверено | Проверено | Hardware skip |
| R2 SFT без packing | Проверено | Проверено | Hardware skip | Проверено | Проверено | Hardware skip |
| R3 CPT isolated | Проверено | Проверено | Hardware skip | Проверено | Проверено | Hardware skip |
| R4 SFT isolated | Проверено | Проверено | Hardware skip | Проверено | Проверено | Hardware skip |

Это 24 обязательных сочетания. Проверены 16 сочетаний SDPA/FA2 на RTX 4090
с реальным loss/backward для full fine-tuning и LoRA. Оставшиеся 8 сочетаний FA3
реализованы через штатный HF varlen interface, но требуют проверки на Hopper.
Статус матрицы не подтверждает production размеры или multi-node. Базовые model classes — `Qwen3ForCausalLM` и
`Qwen3_5ForCausalLM`; text-only обучение из checkpoint с multimodal wrapper
Qwen3.5 должно сохраняться. Обучение изображений/видео не входит в эту матрицу.
MoE и другие model classes перечисляются явно в capability-таблице и не
объявляются проверенными только по совпадению семейства в имени checkpoint.

Аппаратные ограничения описывать отдельно:

- A100: SDPA и FA2; FA3 не является поддержанным путём описанной реализации.
- H100: SDPA, FA2 и FA3 при совместимых версиях библиотек и CUDA.
- Неподходящее оборудование или отсутствующий kernel — диагностируемая ошибка,
  а не молчаливая смена backend или потеря изоляции.
- SDPA с блочной маской может выбирать другой внутренний kernel, чем SDPA
  без неё. Поддержка правильного результата не означает одинаковую скорость.

Базовый dtype для GPU-матрицы — bf16. Full fine-tuning и LoRA проверяются
отдельно на маленьких моделях. FP8 storage остаётся дополнительным режимом
замороженной базы и не является условием поддержки этих 24 сочетаний.

## Требования к устройству реализации

1. **Понятность.** Небольшие функции с одной ответственностью; явная схема
   «подготовка примера → packing → collator → adaptation к модели/backend».
2. **Прозрачность.** Один resolved packing config, определённые границы данных
   и labels, диагностируемая фактическая конфигурация, отсутствие скрытого
   изменения objective или backend.
3. **Расширяемость.** Новый backend или model adapter не требует копирования
   CPT/SFT factories. Специфика архитектуры не размазывается по dataset-коду.
4. **Документированность.** Публичные поля, инварианты, форматы tensors,
   ограничения и примеры поддержанных режимов описаны вместе с кодом.

Не строить универсальный trainer framework для ещё не реализованных задач.
SimPO/GRPO не входят в этот этап: их labels, парная/групповая структура и loss
нельзя объявлять поддержанными переносом обычного CPT/SFT collator.

### Разделение ответственности

| Слой | Ответственность | Что в него не входит |
| --- | --- | --- |
| CPT factory | Токенизация готовым tokenizer, BOS/EOS, существующие подстановки и targets, разбиение длинного текста по явной политике | Выбор attention kernels и построение SDPA mask |
| SFT factory | Chat template, выбор диалогов, маски ролей/think, ограничения длины примера | Собственная копия алгоритма document packing |
| Общий packer | `none`, `stream`, `isolated`; упаковка токенов и labels, границы независимых сегментов, остатки | Знание устройства Qwen3 или GatedDeltaNet |
| Общий collator | Padding/flattening, сборка нескольких samples/packs, перенос канонических границ | Токенизация, изменение SFT objective |
| Attention/model integration | Преобразование границ в mask/varlen kwargs, Qwen3.5 convolution и recurrent isolation | Упаковка исходного корпуса |
| Validation и Trainer | Проверки возможностей, сохранение metadata, loss accounting, logging, checkpoint/resume | Дублирующие условия вида `if sft.packing` по всему коду |

Сохранить `DatasetFactory`/`CollatorFactory` как точки подключения. По возможности
переиспользовать функции из `datasets/packing.py`, `datasets/collators.py`,
`sft_factory.py` и `core/validation.py`. Общий model/backend adapter размещать
отдельно от подготовки данных; новые файлы добавлять по ответственности, без
зеркальных `cpt_packing` и `sft_packing` реализаций.

### Общий конфиг и совместимость

Реализованный раздел `packing` в `MainConfig`:

```json
{
  "packing": {
    "mode": "isolated",
    "max_length": 4096,
    "loss_normalization": "tokens"
  }
}
```

`mode` принимает `none`, `stream`, `isolated`; `max_length` задаёт бюджет
pack для stream/isolated. Ограничение длины отдельного SFT-примера остаётся
в task-specific подготовке. `loss_normalization=tokens` явно включает новый
контракт суммарного causal CE / числа shifted targets; legacy-конфиги сохраняют
`legacy` нормализацию.

Отдельный `eval_mode` не вводится: CPT stream оценивается потоком, CPT isolated —
независимыми документами/фрагментами, SFT всегда отдельными диалогами без packing.
В CPT eval отключена training substitution. И train, и eval получают политику
из одного resolver; встроенные factories объявляют свою задачу атрибутом `task`.

| Требуемый режим | Factory | `packing.mode` |
| --- | --- | --- |
| R1 | CPT/CLM factory | `stream` |
| R2 | SFT factory | `none` |
| R3 | CPT/CLM factory | `isolated` |
| R4 | SFT factory | `isolated` |

Старые `sft.packing`, `sft.pack_chunk_size` и `data.block_size` преобразуются
в общий resolved config в одном месте. При наличии новых и старых полей
несогласованные значения отклоняются. При отсутствии нового раздела сохраняются
старые defaults: CPT stream, SFT без packing либо isolated при `sft.packing=true`,
SFT eval без packing. Совместимость не должна зависеть от порядка разбора JSON.

Старые production-конфиги не переписывать автоматически. Новые имена и правила
миграции документировать; CLI help, `data_prep_only` и обучение используют
одинаковый resolver. SFT stream не входит в четыре требуемых режима и не должен
включаться случайно из-за нового общего поля.

### Канонические границы и labels

Подготовленный пример содержит как минимум `input_ids` и `labels` одинаковой
длины. Packer сохраняет реальные длины сегментов и принадлежность padding;
границы не восстанавливаются поиском EOS, который может встретиться в данных.
Для stream независимая последовательность — весь блок, для isolated — отдельный
документ/фрагмент. Не поддерживать несколько несогласованных источников границ.

Из одного описания сегментов вычисляются позиции, `seq_idx`, offsets q/k и
максимальные длины. Labels уже могут содержать подстановки CPT или маски SFT:
packer не заменяет их безусловным `labels = input_ids`.

В R3/R4 начало каждого независимого сегмента получает `-100` до стандартного
causal shift. Padding также получает `-100`. В R1 границы исходных документов
не маскируются дополнительно. В R2 сохраняется отдельная обработка диалога.
Особенно проверить SFT с `only_target_loss=false`: изоляция не должна зависеть
от того, замаскировал ли upstream начало примера случайно.

Длинные CPT-тексты можно разделять на независимые фрагменты по заданной политике;
для SFT сохранить обработку целого диалога и существующие ограничения длины.
Отброшенные/обрезанные данные учитывать явно. Хвосты map batches и shards,
пустые примеры, BOS/EOS, совпадение PAD/EOS и изменение длины после substitution
должны иметь определённую семантику. Изменение состава packs при миграции cache
фиксируется отдельно от изменения контекста и loss.

### Backend и model adapters

Для R1/R2 использовать обычный модельный causal/padding путь. Изолированная
интеграция активируется только для R3/R4; отсутствие специальных masks/offsets
в stream является нормальным поведением, не ошибкой.

Для FA2/FA3 isolated передавать полный набор q/k offsets и max lengths,
а не полагаться только на распознавание сброшенных `position_ids`. Не добавлять
2D padding mask, которая переводит HF в unpadding-путь с потерей внутренних
границ. Сохранённый padding представить отдельным сегментом либо согласованно
удалить из всех tensors; первый вариант проще для начальной реализации.

Для SDPA isolated строить явную блочно-диагональную causal mask. Разрешённые
связи: тот же сегмент, causal-порядок и дополнительные ограничения архитектуры,
например sliding window, если включён в конфиге. Проверять формат маски именно
в HF adapter: boolean semantics прямого SDPA и additive HF masks различаются.
Учитывать O(T²) память и безопасное поведение padding query rows.

В Qwen3 маски должны попадать во все соответствующие attention layers, включая
sliding-attention при его наличии. В Qwen3.5 SDPA mask full-attention слоёв
не должна попадать в GatedDeltaNet как обычная 2D padding mask; нужны отдельные
пути для full attention, `seq_idx` convolution и `cu_seqlens` recurrence.
Это обязательная часть реализации SDPA, а не основание исключить эту клетку
из матрицы. Поведение fallback без FLA/causal-conv kernels должно быть явным.
[Qwen3 HF integration](https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/qwen3/modeling_qwen3.py),
[Qwen3.5 HF integration](https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py).

Предпочитать штатные HF interfaces. Если необходим adapter или локальная обёртка,
он должен быть небольшим, версионно проверяемым и включаться для конкретной
модели/режима. Не копировать целиком модель и не применять глобальные monkey
patches, меняющие обычные R1/R2. Проверить text-only загрузку, PEFT, checkpointing,
save/load и совместимость ключей state dict.

### Микробатчи и нормализация

R1/R2 сохраняют обычную форму `[B, L]` и поддерживают B > 1. R3/R4 должны
поддерживать логический B > 1: collator может объединять несколько packs
в `[1, T]`, сохраняя отдельные сегменты и перенумеровывая `seq_idx`.
Ограничение физического batch 1 у FLA не должно превращаться в постоянное
ограничение числа packs, взятых из DataLoader.

Максимальная длина varlen — длина самого длинного сегмента, а не T. Для SDPA
flattened mask растёт как T²; этот расход документируется и измеряется.
Поддержка B > 1 не означает обещания одинакового предельного B для всех kernels.

Для новых явно заданных режимов использовать документированную нормализацию
causal loss по числу действительных targets после shift. Сумма CE и target count
должны согласованно учитываться по accumulation steps и data-parallel ranks.
Не усреднять локальные средние при неравных counts и не дублировать множители
HF/Accelerate/DDP. Существующий `accepts_loss_kwargs=False` не менять без
проверки фактического loss path, включая Liger.

Legacy-конфиги не должны молча получать другое усреднение loss. Если новый
контракт меняет численный update, оформить явно выбранный режим нормализации
и migration note; сохранять resolved semantics в checkpoint. Изменение способа
упаковки при одинаковых независимых документах и token normalization не должно
менять смысл objective.

## Порядок выполнения

Реализация основного пути выполнена. Этапы с аппаратной приёмкой остаются
частично открытыми: RTX 4090 не заменяет A100/H100, CPU/Gloo не заменяет NCCL,
FSDP и DeepSpeed. Ниже сохранены критерии, чтобы hardware skip не закрыл этап.

| Этап | Статус и свидетельства |
| --- | --- |
| 1 | Реализован `PackingConfig`/resolver, legacy defaults, конфликты, task/backend validation; автономные tests |
| 2 | Общий Arrow packer, canonical segment_lengths, collator B > 1; проверены tails, PAD=EOS, custom labels, substitution с разными workers |
| 3 | Qwen3 SDPA/FA2: R1–R4 full/LoRA loss+gradients; CPU Trainer/optimizer/eval/resume; FA3 ждёт Hopper |
| 4 | Qwen3.5 SDPA/FA2: R1–R4 full/LoRA, изоляция и переменные сегменты; FA3 ждёт Hopper |
| 5 | Shifted target normalization, accumulation parity full/LoRA, eval weighting, all-ignored labels; GPU checkpointing/Liger SDPA/FA2 |
| 6 | Убраны вложенные barriers; два CPU/Gloo ranks прошли train/eval/exact resume R1–R4. Открыты multi-GPU/multi-node NCCL, FSDP/DeepSpeed и большой корпус |
| 7 | Документация кода и четыре JSON-шаблона обновлены; открыты A100/H100/FA3 и performance measurements |

Особенности совместимости: новый явный CPT stream сохраняет хвосты через map
batches, старый остаётся прежним; `natural_boundaries` поддерживается только
legacy конфигом. `ruadapt_packing.json` защищает resume от смены resolved policy,
но не от подмены файлов корпуса. Isolated SDPA использует плотную O(T²) mask;
количество логических packs больше не ограничено FLA physical batch=1.

### Этап 1 Конфиг и контроль существующих режимов

- Ввести общий `PackingConfig`, resolver legacy/new полей и capability-таблицу
  task/mode/model/backend. Убрать зависимость общего validation от `sft.packing`.
- Зафиксировать референсные R1/R2 и существующий R4 на маленьких fixtures;
  для R1 отдельно подтвердить разрешённый междокументный контекст внутри блока.
- Разделить проверку config до загрузки и фактических модели/kernels после неё.
  Не разрешать ещё не реализованный путь только изменением allowlist.

Готовность: новый конфиг однозначно разбирается, старые сохраняют смысл,
конфликты дают понятную ошибку; контрольные тесты R1/R2 проходят.

### Этап 2 Общая подготовка packs и collator

- Выделить общую упаковку из SFT и CPT, оставив в factories только task-specific
  подготовку. Substitution сохранить как преобразование готовых CPT-примеров.
- Реализовать канонические сегменты, `none`/`stream`/`isolated`, labels boundaries,
  padding и flattening нескольких packs; убрать дублирование границ.
- Версионировать cache по mode, policy и schema. Старые stream Arrow-файлы
  не принимать за isolated. Не материализовать без необходимости весь корпус
  в полных CPU-тензорах на каждом rank.

Готовность: автономные тесты всех четырёх форматов, детерминированности,
остатков, substitution и B > 1; известен объём потерянных/сохранённых токенов.

### Этап 3 Полная поддержка Qwen3

- Подключить R1–R4 к Qwen3; реализовать SDPA mask и FA2/FA3 varlen через общий
  attention adapter. Учесть configured causal/sliding-window semantics.
- Провести настоящие forward/loss/backward tests, затем короткие training smoke
  для full fine-tuning и LoRA с сохранением и загрузкой checkpoint.

Готовность: 12 сочетаний Qwen3 проходят на совместимом оборудовании;
SDPA isolated не заменяется eager/FA, FA3 не помечается успешным через skip.

### Этап 4 Полная поддержка Qwen3.5

- Использовать тот же packer/collator; добавить архитектурную передачу границ
  в convolution и FLA. Для SDPA развести full-attention mask и linear path.
- Проверить R1 без сброса state на границах исходных документов; R2 с batch > 1
  и padding; R3/R4 с переменными сегментами, padding и несколькими packs.
- Сохранить text-only загрузку и проверять реальные выбранные kernels.

Готовность: 12 сочетаний Qwen3.5 проходят; изменение документа не влияет
на другой в isolated-режиме через full attention, convolution или recurrence.

### Этап 5 Trainer loss и регрессии

- Провести единую нормализацию по действительным targets через Trainer,
  accumulation и loss implementation; сохранить явную legacy-совместимость.
- Проверить отдельную/packed обработку одинаковых документов по сумме CE,
  count, mean loss, gradients и optimizer update. Для R1 референс — непрерывный
  блок, а не независимые документы.
- Проверить `only_target_loss=true/false`, пустые labels, all-ignored batch,
  gradient checkpointing и включённый/выключенный Liger там, где он применяется.
- Защитить доставку metadata через Trainer и PEFT; служебные поля не должны
  исчезать из-за `remove_unused_columns` или ломать модельный forward.

Готовность: microbatch/accumulation не меняют objective при фиксированных данных
и выбранной нормализации; предыдущие checkpoint/freeze/CLI tests проходят.

### Этап 6 Distributed и подготовка большого корпуса

- Упорядочить cache synchronization: один владелец barriers, согласованная
  стратегия общего и локального cache. Проверить реальный factory path,
  а не только Trainer на заранее созданном dataset.
- Проверить NCCL на одном узле, затем минимум на двух; разные target counts
  между ranks, неполный последний batch, shuffle, сохранение и resume.
- Для FSDP/DeepSpeed выполнить отдельные smoke в разрешённых режимах freeze;
  не снимать существующий запрет частичной заморозки строк без отдельного решения.
- Логировать effective task/mode/model/backend, dtype, логический B, физическую
  форму, T, targets, fill, потери данных, cache schema и фактический FP8 storage.

Готовность: нет зависания подготовки/обучения, глобальный loss корректен,
resume воспроизводит данные и update в установленном допуске. Отдельно
фиксируются результаты DDP, FSDP и DeepSpeed, без переноса статуса между ними.

### Этап 7 Документация и аппаратная проверка

- Обновить `docs/PACKING_AND_ATTENTION.md` по фактически реализованному коду,
  включая Qwen3, общий конфиг, tensor contracts и рабочие режимы. Документация
  остаётся описанием кода, без даты документа и без превращения в план работ.
- Добавить небольшие JSON-примеры R1–R4 с заменяемыми model/backend и таблицу
  выбора для A100/H100; не размножать 24 почти одинаковых production-конфига.
- Описать legacy migration, eval semantics, память SDPA mask, ограничения FLA,
  правила длинных документов и отличие stream/isolated loss.
- Заполнить результаты 24-cell матрицы. A100 покрывает SDPA/FA2, H100 — также FA3;
  наличие hardware skip остаётся непроверенной клеткой, не успешной приёмкой.
- Сравнить useful input/target tok/s, peak VRAM/RAM и step time при одинаковых
  данных; не обещать одинаковую скорость SDPA, FA2 и FA3.

Готовность: каждый заявленный режим воспроизводим по документации и примеру;
аппаратные ограничения и непроверенные дополнительные возможности видны явно.

## Проверки и критерии завершения

| Проверка | Обязательный результат |
| --- | --- |
| Конфиг | Все четыре режима однозначны, legacy значения мигрируют, противоречия отклоняются |
| R1 CPT stream | Сохраняется обычный контекст внутри блока; нет лишней block mask, reset позиций или labels на границах исходных документов |
| R2 SFT none | Один диалог на sample, обычный B > 1, корректные padding и роли |
| R3/R4 isolated | Packed logits, реальный CE и gradients эквивалентны отдельным последовательностям в численном допуске |
| Причинность | Изменение будущих токенов не влияет на прошлые; изоляция не превращает внимание в bidirectional |
| Независимость | Изменение A не влияет на B в isolated; отрицательный контроль обнаруживает потерю mask/offsets/seq_idx |
| Границы | Длины 1, меньше convolution kernel, kernel tile boundaries, exact fill, padding, EOS=PAD, несколько packs |
| Labels | Начала independent sequences маскируются до shift; SFT и substitution targets сохраняются |
| Batch и loss | B = 1/B > 1, разный fill и accumulation дают эквивалентный update при token normalization |
| Модели/backend | Все 24 сочетания имеют forward/loss/backward результаты; full/LoRA проверены отдельно |
| Distributed | Реальные factories и Trainer, unequal counts, NCCL, два узла, cache, неполные batches, save/resume |
| Совместимость | Checkpoint, freeze, CLI и существующие production-конфиги не регрессируют |
| Документация | Описание соответствует коду; примеры и локальные ссылки проверены |

CPU-тесты используют синтетические tokenizer/datasets и маленькие конфигурации
моделей. GPU-тесты явно выбирают backend; зависимости и отсутствие нужного GPU
отмечаются отдельно. Correctness-тесты используют dropout 0 и заданные seed;
допуски учитывают порядок редукций и bf16, но уверенно обнаруживают утечки.
Проверка реального loss обязательна: тест только hidden states недостаточен.

Обязательные режимы нельзя объявить готовыми после успешного запуска одного
SFT на Qwen3.5. Завершение определяется матрицей, корректностью обучения,
поддержкой логического B > 1, совместимостью и документацией.

## Границы этого этапа

Создание/расширение/trim токенизатора, embedding initialization и LEP остаются
отложенными. Использование готового tokenizer и сохранение существующей CPT
substitution входят в текущую подготовку данных; исправления подготовки
самого tokenizer не возвращаются в scope.

SimPO/GRPO и другие preference trainers, multimodal document packing,
sequence/context parallelism и W8A8 compute не добавляются в этом этапе.
Неизменность базовых production-конфигов и сохранение результатов предыдущих
исправлений обязательны; изменение training schedule не входит в задачу.

## Ранее реализованные исправления

### Checkpoint и адаптеры

Статус: реализовано; CPU-тесты и двухпроцессный Gloo save/resume пройдены.

`SavePeftModelCallback` записывает текущие веса в
`best_model_checkpoint/lora_adapter`, а также выполняет запись с каждого rank.
Основной checkpoint должен создаваться штатным Trainer: это сохраняет optimizer,
scheduler, RNG и trainer state вместе с весами. Коллективная сборка состояния для
FSDP/DeepSpeed должна сохраняться; ограничение одним rank относится к записи.

Новые checkpoint содержат адаптер в корне. В Transformers 5.9 вложенный
`lora_adapter` распознаётся как отдельный именованный адаптер и нарушает resume;
поэтому такие дубликаты больше не создаются. Для старых checkpoint загрузчик
предпочитает канонические веса в корне. `final_lora_adapter` сохраняется для
совместимости через экспорт уже собранных артефактов, без повторного обхода модели
в callback. Текущие веса не записываются в каталог другого шага.

Проверки: неизменность лучшего адаптера после следующего save, загрузка адаптера,
состояние optimizer/scheduler/RNG, корректная ротация checkpoint, эквивалентность
непрерывного обучения и resume. Короткая проверка на двух процессах.

### Заморозка старого словаря

Статус: реализовано; tied/untied, LoRA/без LoRA, resume и optimizer после
Accelerate проверены на CPU.

Hooks, зарегистрированные до PEFT, теряются при копировании `modules_to_save`.
Установить их после PEFT и окончательного связывания embeddings на фактически
обучаемые параметры; полностью замороженные параметры пропускать. Для tied
embeddings проверять идентичность Parameter и отсутствие дубликатов в optimizer.

Для частичной заморозки требовать нулевой embed weight decay. При resume защитить
замороженные строки от загруженного momentum optimizer. Проверять границы
`freeze_idx`. Проверки: tied/untied, LoRA/без LoRA, обновление новых строк при
неизменности старых после нескольких шагов и после resume.

Текущая поддержка: torch AdamW (включая fused), single-device/DDP. Для частичной
заморозки FSDP/DeepSpeed отклоняются: шардированные параметры не сохраняют простую
индексацию строк словаря. Это ограничение не относится к обычному SFT/LoRA с
полностью замороженной базой. В checkpoint сохраняется ruadapt_freeze.json;
несовместимая граница freeze_idx при resume отклоняется. Для старых checkpoint
без metadata выдаётся предупреждение; уже изменённые старым кодом веса не
восстанавливаются. Momentum замороженных строк обнуляется перед шагом AdamW.

### Изоляция документов при packing

На предыдущем этапе действовали guards Qwen3.5 + FA2/FA3 + B=1 и hidden-state
parity test. Эти ограничения заменены общей реализацией; новый adapter поддерживает
Qwen3/Qwen3.5, SDPA block-causal mask и FA varlen, логический B > 1. Guards для
удаления колонок, неизвестных architectures/backend и отсутствующих FLA kernels
сохранены. Новый тест сравнивает causal CE и gradients с независимыми документами;
stream имеет отдельный референс, сохраняющий междокументный контекст.
FA3 остаётся hardware skip новой матрицы.

### FP8 исключения

Статус: реализовано; CPU-регрессия обеих веток пройдена.

`_rewrite_modules_to_not_convert` должен сохранять все пути при text_only=false.
Переписывание language_model и удаление vision/MTP-путей применяются только к
text_only=true. Регрессионный multimodal-тест обязательно содержит текстовые пути.
Проверить исходный список на неизменность и обе формы quantization_config.

### Входной путь SFT

Статус: реализовано; маскирование и полный CLI SFT smoke test пройдены.

Исправить имена `<|im_start|>` и `<|im_end|>` при автоопределении. Явные ID имеют
приоритет, включая ID 0. Отсутствующие токены диагностируются до загрузки данных.
Проверить маскирование на синтетическом диалоге и явно обработать неподдержанное
многотокенное представление границ. Это исправление потребителя готового
токенизатора; создание и преобразование словаря остаются отложенными.

### CLI обучения

Статус: реализовано; установка пакета, console/module help, data_prep_only,
SFT train и автоматический resume проверены на CPU.

Console entry point вызывает main без обязательного config. Добавить cli(),
которая вызывает fire.Fire(main), и использовать её в console script и __main__.
Перенести импорт fire в CLI-обёртку. Проверить установленную команду, запуск
модуля, передачу config и data_prep_only, а также короткое обучение.

## Окружение и общие правила проверок

Маленькие модели создаются из конфигурации, токенизаторы — локально из fixtures.
Автономные тесты не зависят от GPT-2 в Hub и `/workdir/models`. Production-модели
используются только в явно выделенных интеграционных проверках.

Базовый стек аудита: Python 3.12, torch 2.11.0+cu128, transformers 5.9.0,
PEFT 0.19.1. В локальном training-образе отсутствует fire; установить объявленную
зависимость для проверки CLI. GPU-нагрузку начинать после проверки доступности
устройств. Полное production-обучение для исправлений не требуется.

Каждый активный пункт закрывается только после соответствующих проверок.
Непроверенные hardware/backend-комбинации отмечаются отдельно от успешных тестов.

## Отложенные дефекты

Эти исправления зафиксированы, но не входят в текущую реализацию.

| Область | Дефект и воспроизведение | Будущее исправление и проверка |
| --- | --- | --- |
| CPT tokenizer properties | `_get_tokenizer_properties` индексирует encoding `"1 "` без проверки длины; на WordLevel фикстуре получается IndexError | Проверить определение leading-space без предположений о числе токенов; сейчас CPT tests используют совместимый synthetic BPE |
| BPE extension | core.py вызывает split у list-merges; ниже set(merges) тоже не принимает списки | Нормализовать пары во всём пути; сравнить оба формата и загрузить итоговый tokenizer |
| Trim | Новые free_token не имеют id, следующий цикл падает с KeyError | Один проход назначения окончательных ID; проверка mapping, padding, added_tokens_decoder и переноса весов |
| Batched init | set(special_tokens_map.values()) падает на additional_special_tokens; debug-индекс 260577 ломает малые словари | Нормализовать специальные токены, убрать привязку к конкретным ID; проверить mean/wmean/MLP |
| LEP | Вместо связывания weight выходной Linear заменяется на Embedding; forward падает | Связать Parameter, локализовать EMB_MODULES; tied/untied forward и save/load |
| MLP head training | f-string в initialization/head/train.py:475 не разбирается Python 3.10/3.11 | Промежуточная переменная и синтаксическая матрица версий Python |

Связанные проверки: согласованность tokenizer/config/generation_config после trim,
размеры vocabulary после LEP, совместимость обученной MLP-головы с применением.
Статус всех пунктов таблицы: отложено по указанию пользователя.

## Результаты предыдущего этапа

Следующие результаты относятся к предыдущему этапу исправлений. Они не
подтверждают новую матрицу из 24 сочетаний и не закрывают этапы плана выше.
Изменения предыдущего этапа находятся в рабочем дереве.

| Проверка | Команда внутри test image | Результат |
| --- | --- | --- |
| Полный автономный набор | `python -m pytest tests/ -q -p no:cacheprovider --tb=short` | 172 passed, 3 skipped; пропуски относятся к GPU packing и тяжёлому FP8 parity |
| Завершающая регрессия checkpoint и freeze после поддержки optimizer-обёртки | `python -m pytest tests/training/test_freeze_peft.py tests/training/test_checkpoint_resume.py -q -p no:cacheprovider --tb=short` | 17 passed; включает один тест, добавленный после полного прогона |
| DDP save/resume | `PYTHONPATH=. python -m torch.distributed.run --standalone --nproc_per_node=2 dev/validate_training_ddp.py --output <fresh-directory>` | 2 процесса, Gloo, 4 шага; точное совпадение adapter weights с непрерывным обучением; оба rank сохранили RNG |
| GPU packing | `python -m pytest tests/training/test_sft_packing.py -k isolation -q -p no:cacheprovider --tb=short` | RTX 4090: FA2 passed, FA3 skipped; hidden states, LoRA gradients, изменение первого документа и отрицательный контроль проверены |
| Статические проверки | `git diff --check`; Ruff F821/F822/F823 для training и новых тестов | Пройдены |

CLI smoke собирает и устанавливает локальную копию пакета без сети и зависимостей,
затем выполняет data preparation, два шага SFT и resume до четвёртого шага.
Production-конфиги на диске не изменялись; тесты разбора конфигов в CPU-окружении
используют временные копии с use_cpu=true.

Среда: `refalmachine/ngc_pytorch_25_03_torch_2.11_vllm_0.21:v1.0`, Python 3.12,
torch 2.11.0+cu128, transformers 5.9.0, PEFT 0.19.1. Временный CPU-контейнер
дополнен fire 0.7.1. Для Gloo smoke явно установлен ACCELERATE_TORCH_DEVICE=cpu:
иначе установленный Accelerate выбирает cpu:0, который torch.load не принимает
как map_location при distributed resume.

Оставшиеся проверки: FlashAttention 3 на Hopper, GPU NCCL/DDP, FSDP/DeepSpeed,
полная загрузка multimodal FP8 и 27B parity. Сохранение FSDP/DeepSpeed передано
штатному HF/Accelerate, но эти backend не проверены выполненным Gloo-тестом.
Production-run 27B не запускался. Дефекты из отложенного списка не исправлялись.

## Проверки общей реализации

- `tests/training/test_document_packing.py`: 24-cell матрица × full/LoRA;
  FA3 пропускается без Hopper. SDPA и FA2 проверены на RTX 4090.
- `tests/training/test_packing_training.py`: factories → Trainer для R1–R4,
  token-normalized optimizer update, eval, checkpoint policy, all-ignored loss;
  Qwen3.5 LoRA + gradient checkpointing с Liger 0.8.1 и без него.
- `dev/validate_packing_ddp.py`: два CPU/Gloo ranks, четыре режима, неодинаковые
  target counts, неполный eval batch, SGD update против однопроцессного эталона
  и точное совпадение parameters после resume.
- `configs/examples/{cpt_stream,sft_none,cpt_isolated,sft_isolated}.json`:
  шаблоны с заменяемыми путями, без изменения production-конфигов.

Для полной приёмки остаются аппаратные проверки FA3/A100/H100, multi-node NCCL,
FSDP/DeepSpeed, throughput/VRAM/RAM на больших данных. Все отложенные дефекты
токенизатора/его подготовки остаются вне реализации.

Итоговая регрессия: `python -m pytest tests/ -q -p no:cacheprovider --tb=short`
— **227 passed, 18 skipped**. Из пропусков 16 относятся к новой FA3 матрице,
один к прежнему FA3 isolation test, один к тяжёлому FP8 parity. Новый DDP script
дополнительно прошёл все четыре режима (global SGD parity + eval + exact resume).
Четыре JSON-примера разобраны `parse_config_from_json` и проходят
`validate_training_config`. Проверки выполнялись в контейнере
`refalmachine/ngc_pytorch_25_03_torch_2.11_vllm_0.21:v1.0` с установленными
`fire==0.7.1` и `liger-kernel==0.8.1`; см. версии стека в документации.


## Общие настройки длины и укладки

Реализованы независимые sequence.max_length/overflow и packing.strategy/isolate/tail.
CPT и SFT используют один packer; dense isolated заполняет блоки с сохранением
границ attention. SFT labels строятся до нарезки; фрагменты без targets сохраняются
с конечным нулевым loss. Tail=keep передаёт isolated packs без дополнения, eval
считает loss по логическим packs переменной длины. Старые конфиги сохраняют дефолты.

Добавлены загрузка списков/глобов JSONL и потоковая утилита split_jsonl с группировкой
по хешу текста либо ID до токенизации/packing. Пример полного CPT с eval/checkpoints:
research/experiments/legal_cpt/config.example.json (в отдельном репозитории research). Детали: docs/DATASET_PREPARATION.md.
Аппаратная проверка новых сочетаний FA3 остаётся задачей для Hopper.

## Ускорение подготовки большого корпуса

Реализовано:

- Dense packing читает Arrow buffers и копирует токены через NumPy, без Python
  объектов на каждый токен. После прохода по длинам документов диапазоны выходных
  packs распределяются между `data.preprocessing_num_workers` процессами.
- Результат совпадает с последовательным `iter_packs`: порядок, overflow,
  segment_lengths, labels, границы split и единственный хвост. Работает для CPT/SFT,
  stream/isolated; whole/none продолжают использовать последовательный путь.
- CPT без substitution пропускает промежуточный map, копировавший документы/labels.
- Готовый dense cache загружается без пересчёта длин, независимо от числа workers;
  отдельная версия реализации и FileLock защищают cache.
- `data_prep_only --prepared_dataset_dir` сохраняет manifest ссылок на готовые
  Arrow-файлы. `data.prepared_dataset_dir` включает загрузку без вызова factories.
  Несовместимые параметры/код, изменённые исходники или отсутствующий cache
  вызывают ошибку, а не многократную подготовку в training allocation.
- Slurm-скрипт поддерживает PREPARE_ONLY=1 (один узел, один видимый GPU для
  проверки training config) и PREPARED_RUN_DIR для training. Через afterok
  многоузловое обучение запрашивает ресурсы только после успешной подготовки.
  Готовый manifest ссылается на cache; для dense не создаётся вторая копия корпуса.

Проверки: 141 passed, 18 skipped в целевом наборе packing/factories/Trainer/CLI/
config/benchmark; аппаратные пропуски FA3 сохранены. 11 тестов Slurm-запускалки
прошли, включая подготовку отдельно, 16 training nodes без preprocessing и
отказ при незавершённой подготовке. Проверки Slurm используют mock scheduler.
Дополнительная регрессия прежних dataset factories/SFT и финального cache/manifest:
87 passed, 1 skipped. Наборы частично пересекаются, их результаты не суммируются.
`dev/validate_packing_ddp.py` прошёл все шесть режимов на двух CPU/Gloo ranks:
глобальный SGD update, взвешенный eval и точный resume с новым dense cache.

Локальный `python -m dev.benchmark_packing --documents 10000 --workers 1 4`:
40 577 747 токенов, L=4096, 9 907 packs; эталон 17.53 с, Arrow/1 worker 0.73 с,
Arrow/4 workers 0.50 с, одинаковые токены/labels/segments. Корпус помещается в
page cache, замер не содержит fsync; это не обещание такого же ускорения на
кластерном filesystem. Осталось измерить время/память на полном юридическом
корпусе и подтвердить раздельные jobs в реальном Slurm.

Команды, ограничения manifest и cache:
[подготовка данных](../docs/DATASET_PREPARATION.md#раздельная-подготовка-и-обучение).

## Документация проекта

Статус: запланировано. Документация описывает фактический код и его ограничения;
планы развития и результаты отдельных экспериментов остаются в dev/. Даты создания
в новые руководства не добавлять. Неподдержанные возможности и непроверенные
сочетания отмечать явно, особенно SimPO/GRPO, FA3 и большие distributed-сценарии.
Работа над документацией не возвращает отложенные исправления tokenizer/init/LEP
в текущую реализацию: их известные дефекты нужно описать как ограничения.

Уже есть тематические документы:

- [Packing и attention](../docs/PACKING_AND_ATTENTION.md): контракты, изоляция,
  нормализация, backend и аппаратные ограничения.
- [Подготовка данных](../docs/DATASET_PREPARATION.md): shards, train/val,
  cache, обычное обучение и Slurm. Примеры обезличены; описаны локальные конфиги.
- [FP8 storage](../FP8.md): существующий путь хранения замороженной базы.
- [Структура проекта](../STRUCTURE.md): карта файлов, которую нужно сверить с кодом.

Это частичное покрытие, а не готовый справочник всего проекта.

### Порядок и состав работ

| Приоритет | Документ или раздел | Что описать | Источник для проверки |
| --- | --- | --- | --- |
| 1 | Навигация в README и docs/README.md | Назначение проекта; маршруты пользователя: готовая модель → CPT/SFT и изменение словаря → initialization → CPT → LEP; ссылки на справочники и рабочие примеры | Публичные CLI, pyproject.toml, существующие docs |
| 1 | docs/CONFIGURATION.md | Все собственные поля model, data, sequence, packing, unified_dataset, sft, lora, freeze, training; dataset_factory/collator_factory; тип, default, допустимые значения, зависимости, ошибки и legacy aliases | config/schema.py, config/packing.py, core/validation.py, train.py |
| 1 | docs/TRAINING.md | Путь config → модель/PEFT/freeze → dataset → collator → Trainer; full/LoRA, train/eval, token loss, optimizer/scheduler, checkpoint/resume/export; как различаются sample, pack, microbatch, accumulation и global batch | training/core, training/datasets, train.py и tests/training |
| 2 | docs/ARCHITECTURE.md | Основные идеи и контракты между модулями; tokenizer и размеры embeddings, tied/untied weights, подготовка labels, document isolation, FP8 storage против FP8 compute, LEP; входные/выходные артефакты каждого этапа | Реализация основных модулей и их вызовы |
| 2 | docs/TOKENIZATION.md | Extension/replacement/trim, BPE decomposition и форматы merges; перенос ID и весов, special tokens, согласование tokenizer/model/generation config; ограничения текущей реализации | tokenization/, её CLI и tests/tokenization |
| 2 | docs/INITIALIZATION.md | Mean/weighted mean/MLP, подготовка BPE-датасета, cache hidden states, обучение и применение головы, mini-CPT как проверка; форматы данных/артефактов и tied embeddings | initialization/ и вызываемые функции tokenization/ |
| 2 | docs/LEP.md | Идея Layer Embedding Projection, роли base/adapted/instruct моделей, compose/merge, параметры и формы матриц, адаптеры text-only; известные ограничения и проверки результата | ushanka/, utils/adapter.py, scripts/merge_lora.py |
| 2 | docs/EVALUATION_AND_INFERENCE.md | Запуск llmtf_open и vLLM, формат входов/выходов, загрузка модели/адаптера, фиксация настроек оценки и сравнение моделей | evaluation/, inference/, соответствующие scripts |
| 2 | Раздел расширения проекта | DatasetFactory/CollatorFactory, добавление backend/model adapter, общие utils и инварианты, которые должен сохранять новый модуль | Протоколы factories, packing_attention.py, utils/, тесты |
| 3 | docs/OPERATIONS.md | Подготовка окружения, torchrun/Slurm, cache и подготовительный этап, GPU memory, метрики скорости и benchmark warmup, диагностика зависаний, сохранение/resume | CLI библиотеки; экспериментальные запускатели и измеритель перенесены в research/experiments/legal_cpt/ |

### Требования к справочнику конфигурации

- Разделять default dataclass, вычисленное значение resolver и значение в примере.
  Объяснить приоритеты старых/новых полей и ошибки несовместимых настроек.
- Для стандартных TrainingArguments дать ссылки на документацию совместимой версии
  Transformers и описать используемые проектом параметры/переопределения. Не выдавать
  скопированный список всех полей библиотеки за собственную стабильную схему RuAdapt.
- На небольших примерах показать четыре основных режима, whole/dense, overflow/tail,
  token normalization, global batch и различия train/eval. Ссылаться на подробный
  документ packing, не заводить вторую расходящуюся спецификацию.
- Разобрать warmup scheduler отдельно от warmup измерителя, max_steps против epochs,
  совместимость eval/save/load_best_model_at_end и влияние output_dir на resume.
- Примеры должны проходить реальный parser/validator. Аппаратные ограничения проверки
  обозначать явно; успешный разбор JSON не считать проверкой обучения.

### Критерии готовности

1. Для каждого основного модуля понятны назначение, входы/выходы, CLI/API,
   минимальный пример, ограничения и способ проверки результата.
2. Утверждения сверены с кодом и существующими тестами; непроведённые проверки
   не обозначены как успешные. Исторические результаты отделены от текущей поддержки.
3. README и STRUCTURE согласованы с руководствами; устранены устаревшие ограничения
   и неверные заявления о наличии тестов/поддержке режимов. Ссылки ведут на существующие файлы.
4. В примерах только нейтральные пути, публичные model IDs и вымышленные адреса.
   Личные настройки находятся в окружении или local/; опубликованные логи и runtime
   configs очищены от логинов, идентификаторов кластерных проектов и адресов.
5. Сначала завершить справочник конфигурации и руководство обучения, затем описание
   остальных модулей. Документация текущего кода не требует одновременного рефакторинга
   или реализации будущих задач.
