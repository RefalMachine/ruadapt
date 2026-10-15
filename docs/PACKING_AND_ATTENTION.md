# Packing и attention в RuAdapt

Документ описывает подготовку данных, границы контекста и loss в активном
`ruadapt/training`. Общий механизм packing используется для CPT и SFT;
способ подготовки исходного примера остаётся ответственностью соответствующей
factory. Attention backend, способ упаковки и состав обучаемых параметров
выбираются независимо, в пределах описанных ниже ограничений.

## Режимы и текущее состояние

| Режим | Контекст и labels | Eval |
| --- | --- | --- |
| CPT `stream` | Документы объединены в поток, разрезанный на блоки. Внутри блока разрешён контекст через границы документов; обычный causal LM loss | Такой же поток и разбиение, без training substitution |
| SFT `none` | Один диалог — один sample; обычный padded batch из нескольких диалогов. Labels задаёт SFT preprocessing | Отдельные диалоги |
| CPT `isolated` | Документы/фрагменты независимы внутри pack. Первые labels сегментов исключены из causal loss | Независимые документы/фрагменты, упакованные тем же механизмом; без substitution |
| SFT `isolated` | Диалоги или явно нарезанные фрагменты в pack; междиалоговый контекст запрещён. Сохраняются маски ролей и think, дополнительно маскируется начало каждого сегмента | Отдельные диалоги/фрагменты, без packing |

Параметра `eval_mode` нет. Eval определяется задачей и выбранным training-режимом.
Общие настройки также допускают CPT без packing и SFT без междиалоговой изоляции;
последний вариант позволяет ответу видеть чужие диалоги внутри pack и должен выбираться явно.
CPT/CLM в этом документе используют одну текстовую factory; CLM с подстановкой
labels отличается подготовкой примера, а не отдельной реализацией packing.

| Модель и backend | CPT stream | SFT none | CPT isolated | SFT isolated |
| --- | --- | --- | --- | --- |
| Qwen3 / SDPA | Проверено | Проверено | Проверено | Проверено |
| Qwen3 / FA2 | Проверено | Проверено | Проверено | Проверено |
| Qwen3 / FA3 | Реализован путь, hardware skip | То же | То же | То же |
| Qwen3.5 text / SDPA | Проверено | Проверено | Проверено | Проверено |
| Qwen3.5 text / FA2 | Проверено | Проверено | Проверено | Проверено |
| Qwen3.5 text / FA3 | Реализован путь, hardware skip | То же | То же | То же |

«Проверено» означает tiny-модель, реальный causal loss и backward для full
fine-tuning и LoRA на RTX 4090. Для isolated проверены logits/gradients против
раздельных документов и отсутствие влияния изменения одного документа на другой.
Это не измерение скорости 27B и не подтверждение multi-node production run.
FA3 использует штатный HF varlen interface, но новая матрица не проверена на Hopper.
Существующий production SFT v9 с FA3 относится к прежнему режиму и не заменяет
проверку всех новых сочетаний.

## Устройство кода

```text
CPT: tokenize → substitution → document records ─┐
                                               ├→ shared packer → collator → model
SFT: chat template → role/think masks → records ┘
```

| Компонент | Ответственность |
| --- | --- |
| [config/packing.py](../ruadapt/training/config/packing.py) | Один `PackingPolicy`: task из factory, mode, max_length, loss_normalization; legacy defaults, конфликты и eval semantics |
| [unified_factory.py](../ruadapt/training/datasets/unified_factory.py) | Загрузка CPT, фильтрация, токенизация, substitution до упаковки |
| [sft_factory.py](../ruadapt/training/datasets/sft_factory.py) | Chat template и маски assistant/think до общей обработки длины |
| [document_packing.py](../ruadapt/training/datasets/document_packing.py) | Последовательный эталон `iter_packs` и общий dispatcher `pack_dataset` |
| [arrow_packing.py](../ruadapt/training/datasets/arrow_packing.py) | Параллельная dense-укладка по глобальным диапазонам токенов, эквивалентная эталону |
| [prepared.py](../ruadapt/training/datasets/prepared.py) | Manifest готовых Arrow-файлов; загрузка без повторного preprocessing |
| [collators.py](../ruadapt/training/datasets/collators.py) | Padding; `DocumentCollator` выводит все границы из `segment_lengths`, объединяет несколько packs |
| [packing_attention.py](../ruadapt/training/core/packing_attention.py) | Локальные hooks экземпляра модели: SDPA full-attention mask, проверка FLA path; штатные FA2/FA3 kwargs |
| [trainer.py](../ruadapt/training/core/trainer.py) | Нормализация loss, отдельный eval collator, token-weighted eval, checkpoint metadata |
| [validation.py](../ruadapt/training/core/validation.py) | Проверки config и фактически загруженного типа модели/backend |

Hooks устанавливаются до PEFT/Trainer wrapping, не добавляют parameters или
ключей `state_dict` и не меняют обычный forward без packing metadata. Модели
не копируются целиком и глобально не подменяются. Isolated adapter поддерживает
`Qwen3Attention`, `Qwen3_5Attention` и `Qwen3_5GatedDeltaNet`. Совпадение только
имени backend у другой архитектуры не гарантирует корректность её packing.
Qwen3.5 поддерживается в текстовом пути; multimodal samples не упаковываются.

## Конфигурация и совместимость

```json
"packing": {
  "mode": "isolated",
  "max_length": 4096,
  "loss_normalization": "tokens"
}
```

`max_length` — длина блока/pack, а не максимальная длина всего flattened forward.
В старом интерфейсе SFT длину диалога задаёт `sft.max_tokens_count`; длинные
диалоги отбрасываются, в isolated размер pack должен быть не меньше этого лимита.
Длинный CPT-документ по умолчанию делится на независимые фрагменты длиной не более
`max_length`. Новые настройки разделяют обработку длинных примеров и укладку:

```json
"sequence": {"max_length": 4096, "overflow": "split"},
"packing": {
  "strategy": "dense",
  "isolate": true,
  "max_length": 4096,
  "tail": "pad",
  "loss_normalization": "tokens"
}
```

| Поле | Значения и смысл |
| --- | --- |
| `sequence.max_length` | Лимит токенизированного примера, включая chat template; применяется после построения labels |
| `sequence.overflow` | `split`: последовательные фрагменты; `drop`: исключение целиком; `truncate`: оставить начало, потеряв хвост; `error`: остановить подготовку |
| `packing.strategy` | `none`: один фрагмент на sample; `whole`: последовательная укладка целых фрагментов; `dense`: резать фрагменты на границе pack ради плотного заполнения |
| `packing.isolate` | Изоляция документов/фрагментов внутри pack; при strategy=none не применяется |
| `packing.tail` | `pad`: сохранить и дополнить; `drop`: исключить последний неполный pack; `keep`: сохранить фактическую длину |

`overflow` применяется до укладки. Поэтому `drop + dense` исключает длинные исходные
примеры, но может делить короткие ради заполнения pack. `whole` требует
`sequence.max_length <= packing.max_length`; `dense` такого ограничения не имеет.
`none` требует `isolate=false` и `tail=keep` (либо отсутствия этих полей).
При strategy whole/dense без mode isolate по умолчанию true, но в конфиге лучше
задавать его явно. Явный mode должен согласовываться со strategy/isolate.

Если sequence отсутствует, CPT использует split с лимитом pack, SFT — drop с
лимитом `sft.max_tokens_count`. В явной секции sequence дефолт overflow — split;
неуказанный max_length получает тот же лимит задачи. Противоречащие явные значения
sequence.max_length и sft.max_tokens_count отклоняются.

Старая запись mode=isolated соответствует whole + isolate=true + tail=pad;
mode=stream — dense + isolate=false + tail=keep. Старые JSON не переписываются.
Новая секция sequence или явный strategy требуют нормализации tokens, в том числе
для фрагментов без targets. Старый SFT без этих настроек сохраняет прежний грубый
фильтр длины по символам; новый путь использует токены и отключает этот фильтр.

Пример при L=8: A=6, B=4, C=6.

```text
whole: [A6 PAD2] [B4 PAD4] [C6 PAD2]
dense: [A6 B1:2] [B2:2 C6]
```

При isolated B1 и B2 — независимые сегменты в разных packs; продолжение не видит
предыдущую часть. Позиции и labels на границах обрабатываются как для отдельных
документов. Для SFT маски assistant строятся до нарезки, но контекст вопроса может
остаться в другом pack. Фрагменты без targets сохраняются: их вклад в loss равен
нулю; обычный optimizer step (включая weight decay/momentum) автоматически не
пропускается. Исходные диалоги вообще без размеченных ответов по-прежнему фильтруются.
SFT eval не упаковывается, однако выбранный sequence overflow применяется и к нему:
при split оцениваются независимые фрагменты, а не восстановленный полный диалог.
Для нового SFT без packing требуется dynamic_padding=true; pad_to_multiple_of
продолжает задавать округление длины обычного батча.

У dense все packs, кроме последнего, полные. Tail=drop не теряет остатки на границах
внутренних map batches, только единый последний хвост. У whole неполные packs бывают
и в середине; drop отбрасывает только последний. В isolated tail=keep collator
передаёт все packs с фактическими длинами без padding, включая whole. Tail=pad/drop
дополняют внутренние неполные whole packs до L. Для не изолированного пути keep
использует обычный прямоугольный батч и может добавлять padding до batch maximum.

Конфиги конкретных юридических CPT-экспериментов и Slurm-запуски находятся
в отдельном репозитории `research`, в `experiments/legal_cpt/`.
Общие CLI подготовки и обучения описаны в [DATASET_PREPARATION.md](DATASET_PREPARATION.md).
Подготовка нескольких JSONL и validation: [DATASET_PREPARATION.md](DATASET_PREPARATION.md).

Для isolated обязательно `training.remove_unused_columns=false`: иначе Trainer
удалит `segment_lengths` до collator. Проверяются конфиг, фактический backend
после загрузки и комплект metadata первого training batch. Не допускается молчаливый
переход запрошенного isolated backend на другой. Для новых режимов нормализация
по умолчанию `tokens`; `legacy` выбирается явно, если нужно прежнее усреднение.

Готовые шаблоны с заменяемыми путями модели/данных:

- [CPT stream](../configs/examples/cpt_stream.json).
- [SFT без packing](../configs/examples/sft_none.json).
- [CPT isolated](../configs/examples/cpt_isolated.json).
- [SFT isolated](../configs/examples/sft_isolated.json).

Запуск после заполнения путей:

```bash
python -m ruadapt.training.train --config configs/examples/cpt_isolated.json
```

Примеры задают full fine-tuning и FA2. `model.attn_implementation` можно заменить
на `sdpa` или `flash_attention_3` на совместимом оборудовании. Размеры batches,
длина и LR в шаблонах не являются измеренным режимом для конкретной большой модели.
LoRA включается существующим разделом `lora`, независимо от packing.

Если `packing` и `sequence` отсутствуют, resolver сохраняет legacy semantics:

| Старый конфиг | Resolved policy |
| --- | --- |
| CPT `data.block_size` | `stream`, заданная длина либо 512, нормализация `legacy` |
| SFT `sft.packing=false` | `none`, нормализация `legacy` |
| SFT `sft.packing=true` | `isolated`, длина `sft.pack_chunk_size`, нормализация `legacy`; eval `none` |

Явно заданные противоречащие старые и новые поля вызывают ошибку. Если новая
длина не указана, используется legacy длина соответствующей задачи. Production
конфиги автоматически не переписываются.

Legacy CPT использует прежний `make_pack_fn → PackedDataset`: границы map batches
влияют на блоки, остатки отбрасываются, готовые данные материализуются в CPU tensors.
Явный новый `packing` включает общий Arrow-путь: остаток переносится между
документами; обработка последнего хвоста определяется tail, padding добавляет только collator.
Это меняет состав блоков, число samples и иногда targets даже при `mode=stream`.
Substitution нового пути получает seed по индексу документа; legacy сохраняет
прежнюю зависимость от map batch. Миграция данных и переход с `legacy` на `tokens`
— отдельные изменения, которые влияют на воспроизводимость update.

`natural_boundaries` остаётся только в legacy CPT, совместно с новым `packing`
отклоняется. Старые `pack_records`/`PackedSFTCollator` оставлены как API совместимости
со старым пятиколоночным форматом; современные factories возвращают канонические
сегменты и используют `DocumentCollator`. Ограничение B=1 старого adapter не
распространяется на общий путь.

## Границы документов и tensor contract

Исходная запись содержит `input_ids` и `labels` одинаковой длины. Labels могут
уже отличаться от inputs из-за подстановок CPT и маскирования SFT. Packer сохраняет
их, за исключением обязательного `-100` в начале isolated-сегмента.

Канонический pack имеет три колонки:

```text
input_ids:       [a0 a1 a2 b0 b1]
labels:          [-100 a1 a2 -100 b1]   # пример CPT isolated
segment_lengths: [3, 2]
```

Границы определяются длинами, не поиском EOS. EOS может встретиться внутри
диалога; совпадение PAD и EOS не удаляет реальные EOS из targets. Пустые записи
пропускаются. SFT-диалоги не разрезаются packer: фильтрация/truncation остаётся
в SFT preprocessing. Переполнение контракта вызывает ошибку. Для CPT начало
каждого выделенного фрагмента также маскируется и позиции начинаются заново;
контекст через границу этих фрагментов не сохраняется.

В stream документы последовательно заполняют блоки: позиции не сбрасываются на
исходных границах, boundary labels дополнительно не маскируются. Каждый блок
сам по себе независим; это не перенос состояния через весь корпус. EOS сообщает
модели о границе как токен, но не запрещает внимание к предыдущему документу.

При isolated и tail=pad/drop каждый оставшийся pack дополняется до L отдельным
padding-сегментом, затем B packs объединяются в `[1, T]`, где T = B × L.
При tail=keep дополнения нет: T равен сумме фактических длин packs; служебное поле
`_packing_lengths` сохраняет их границы для token-weighted eval и не передаётся модели. Пример для B=2, L=8:

```text
pack 0: A длиной 3, B длиной 2, padding 3
pack 1: C длиной 6, padding 2

position_ids: 0 1 2 | 0 1 | 0 1 2 | 0 1 2 3 4 5 | 0 1
seq_idx:      0 0 0 | 1 1 | 2 2 2 | 3 3 3 3 3 3 | 4 4
cu_seq_lens: [0, 3, 5, 8, 14, 16]
max_length_q = max_length_k = 6
```

| Поле forward | Форма / тип | Назначение |
| --- | --- | --- |
| `input_ids`, `labels`, `position_ids` | `[1,T]`, int64 | Токены, causal targets, позиции внутри сегмента |
| `seq_idx` | `[1,T]`, int32 | Идентификатор сегмента для causal convolution Qwen3.5 |
| `cu_seq_lens_q`, `cu_seq_lens_k` | `[S+1]`, int32 | Кумулятивные offsets для FA; q используется также FLA |
| `max_length_q`, `max_length_k` | Python int | Максимальная длина сегмента, включая pad segments, не T |

`attention_mask` уровня модели в isolated не передаётся. Padding имеет собственный
сегмент и labels `-100`; он изолирован, но всё ещё расходует compute. Обычные
R1/R2 используют `[B,L]` и 2D padding mask без packing metadata.

## SDPA, FA2 и FA3: общие различия

SDPA — PyTorch API выбора attention kernels. `flash_attention_2` и
`flash_attention_3` — отдельные пути интеграции Transformers с внешними kernels.
Использование SDPA не означает, что внутри обязательно вызван flash kernel,
и не эквивалентно выбору HF `flash_attention_2`.
[PyTorch SDPA](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html).

| Свойство | SDPA | FA2 | FA3 |
| --- | --- | --- | --- |
| Обычный causal/padded batch | Штатный путь модели | Штатный путь модели | Штатный путь модели на совместимой GPU |
| Изоляция нескольких документов | Явная block-causal mask | Varlen offsets q/k | Varlen offsets q/k |
| Произвольная плотная mask | Поддерживается API, kernel selection зависит от формы | Не является интерфейсом изоляции данного пути | Не является интерфейсом изоляции данного пути |
| Границы GatedDeltaNet | Отдельные FLA/convolution metadata | Те же metadata | Те же metadata |
| A100 | Да, с подходящим kernel | Да | Описанный Hopper FA3 — нет |
| H100 | Да | Да | Да при наличии совместимой сборки |

Backend сам не узнаёт границы потерянных документов. Одного `position_ids=0`
в начале каждого примера недостаточно как общего контракта изоляции.

## SDPA с isolated packing

Adapter строит булеву mask `[1,1,T,T]`: True разрешает связь, если query/key
принадлежат одному сегменту и key не находится в будущем. Диагональ разрешена,
в том числе padding rows, поэтому нет полностью замаскированных query rows.
Для Qwen3 со sliding window дополнительно ограничивается causal расстояние.
Этот формат передаётся непосредственно full-attention модулям с
`is_causal=false`, поскольку causal условие уже включено в mask.

В Qwen3.5 модельная 2D mask также используется linear-attention путём. Поэтому
передать произвольную 4D mask на вход всей модели нельзя: adapter вводит её
только в full-attention modules, а GatedDeltaNet получает свои offsets/seq IDs.
Проверки с реальным backward подтверждают этот путь в используемом HF стеке.

Mask требует O(T²) памяти независимо от того, сколько ненулевых блоков внутри.
Одна bool mask при T=8192 занимает 64 MiB, при T=16384 — 256 MiB; временные bool
маски и workspace kernel увеличивают пик. Реализация не создаёт промежуточную
матрицу расстояний int64, но это не делает путь линейным по памяти.

SDPA может выбрать memory-efficient или math kernel в зависимости от GPU,
dtype, mask и сборки PyTorch; успешный вызов не доказывает flash acceleration.
Нельзя обещать одинаковые память/скорость isolated SDPA и FA varlen.
[Выбор SDPA backend](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/sdp_utils_cpp.h).

## FlashAttention 2

FA2 varlen получает concatenated токены и offsets независимых последовательностей.
При causal attention каждый такой сегмент обслуживается независимо, без плотной
T×T маски. Для full attention объём полезных связей определяется примерно
суммой квадратов длин сегментов, а не квадратом общей длины. Остальные части
модели, включая MLP, продолжают обрабатывать весь T.
[FA2 API](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/flash_attn/flash_attn_interface.py).

RuAdapt передаёт явно обе arrays offsets и обе максимальные длины. Это исключает
зависимость изоляции от эвристики HF, распознающей packing по `position_ids`.
Полезно, в частности, для batch, где каждая реальная последовательность короткая
или присутствуют сегменты padding. Модельная 2D mask не передаётся: HF unpadding
для обычного batch не должен заменить внутренние границы документов.
[HF FlashAttention integration](https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/modeling_flash_attention_utils.py).

Нужны совместимые GPU, dtype, head dimension и установленный flash-attn.
Для рассматриваемого NVIDIA пути FA2 поддерживает Ampere/Ada/Hopper; это не
означает одинаковой производительности на этих архитектурах.
[Поддержка FA2](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/README.md#nvidia-cuda-support).

## FlashAttention 3

Описанный FA3 — Hopper implementation из каталога `hopper` FlashAttention.
Он ориентирован на H100/H800; наличие CUDA на A100/RTX 4090 не делает этот путь
доступным. Нужна совместимая отдельная сборка и подходящий CUDA stack.
[Требования FA3](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/README.md#flashattention-3-beta-release).

FA3 также предоставляет varlen q/k/v, offsets и maximum lengths. Поэтому
смысл изоляции и канонические данные RuAdapt одинаковы для FA2/FA3; отдельного
«FA3 packer» нет. Различаются kernel, устройство и версия integration.
[FA3 varlen API](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/hopper/flash_attn_interface.py).

BF16/FP16 forward+backward относятся к training-пути; FP8 attention forward
не следует считать готовым FP8 training backward. В описанном FA3 API нет
параметра attention dropout, эквивалентного FA2 `dropout_p`; конфигурацию с
ненулевым attention dropout нельзя считать автоматически переносимой.
[Заявленные режимы FA3](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/README.md#flashattention-3-beta-release).
Новая GPU-матрица RuAdapt пропускает FA3 без Hopper; прохождение остальных тестов
не подтверждает FA3 или его производительность.

## Qwen3 и Qwen3.5

Qwen3 использует full attention: для isolated необходимы сегментная causal
изоляция, сброс позиций и корректные labels. Adapter учитывает заданное
sliding-window ограничение соответствующего attention module.
[Qwen3 implementation](https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/qwen3/modeling_qwen3.py).

Qwen3.5 — гибрид full attention и GatedDeltaNet. Например, у 27B text tower
48 linear-attention и 16 full-attention слоёв. Изоляция должна работать в трёх
местах: full attention, causal convolution и recurrent state GatedDeltaNet.
Выбор FA2/FA3 влияет только на full-attention часть.
[Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py).

В isolated `seq_idx` разделяет convolution, `cu_seq_lens_q` разделяет FLA recurrence.
FLA varlen требует физический batch 1. RuAdapt удовлетворяет этому через flattening
нескольких логических packs; ограничение не запрещает DataLoader batch > 1.
[FLA varlen](https://github.com/fla-org/flash-linear-attention/blob/v0.4.0/fla/ops/gated_delta_rule/chunk.py).

Isolated Qwen3.5 требует CUDA, FLA и causal-conv1d kernels. При отсутствии kernels
adapter завершает forward понятной ошибкой: обычный fallback не считается
эквивалентным сбросу состояния на границах. Передача `past_key_values` или
`use_cache=true` запрещена; adapter устанавливает `use_cache=false`. Stream не
сбрасывает linear state между документами внутри блока, но не переносит его
между training batches. `text_only=true` сохраняет загрузку без vision tower.

## Микробатч и нормализация loss

`per_device_train_batch_size=B` означает число dataset samples на rank.
В stream это блоки, в SFT none — диалоги, в isolated — packs. Число независимых
документов внутри pack переменно и не равно B.

| Режим | Физический forward | Увеличение B |
| --- | --- | --- |
| Stream / SFT none | `[B,L]` | Обычный рост batch и памяти |
| Isolated FA2/FA3 | `[1,B×L]` | Больше сегментов; FLA batch остаётся 1 |
| Isolated SDPA | `[1,B×L]` плюс mask | Размер mask растёт квадратично по B×L |

Gradient accumulation накапливает несколько таких forwards. Формальное число
packs на update равно B × accumulation × data-parallel world size для полных
окон; число supervised targets зависит от длин, masking и хвостов.

В `loss_normalization=tokens` objective — сумма causal CE, делённая на число
действительных targets после shift, по всему accumulation window и всем ranks.
Не учитываются padding, `-100` и первая колонка каждой физической sequence.
В isolated начало каждого сегмента маскируется дополнительно, поэтому последний
токен документа не обучается предсказывать первый токен другого.

```text
inputs:  a0 a1 a2 | b0 b1 b2
labels:  -100 a1 a2 | -100 b1 b2
shift:       a1 a2 | -100 b1 b2
```

Сброс attention без masking labels оставил бы неверный boundary target.
Для stream этот переход внутри блока является частью выбранного objective;
сравнивать stream loss с независимыми документами как один objective неверно.

Trainer считает shifted target count, суммирует его между ranks и передаёт
`num_items_in_batch` в causal loss. HF/DDP world-size compensation применяется
один раз. All-ignored batch имеет конечный нулевой loss. Label smoothing и
пользовательский `compute_loss_func` в этом режиме отклоняются, чтобы другой
objective не получил неверный знаменатель. На tiny Qwen3 проверен одинаковый
optimizer update для B=1/accumulation=2 и B=2/accumulation=1, включая LoRA.

Eval в новом режиме также token-weighted. Для каждого логического sample
считаются сумма CE и count; `gather_for_metrics` удаляет дубликаты неполного
последнего distributed batch. Метрики: `eval_loss` и `eval_target_count`.
CE считается дополнительным проходом по logits блоками по 128 targets, чтобы
ограничить дополнительный FP32 workspace. Это добавляет вычисления при eval
и не устраняет память самих logits. Legacy-нормализация сохраняет прежнее
поведение Trainer и не обещает инвариантности к группировке samples.

## Кеширование, distributed и checkpoint

`iter_packs` остаётся последовательным эталоном. `whole`/`none` пишут Arrow через
`Dataset.from_generator`. Для `dense` используется Arrow/NumPy путь: сначала
читаются длины документов, применяется overflow и считается глобальная сумма
длин; затем процессы обрабатывают диапазоны выходных токенов, кратные размеру
pack. Число процессов берётся из `data.preprocessing_num_workers`, но не превышает
число задач. Задача содержит до 256 packs; прогресс показывает задачи, а не документы.
Токены копируются числовыми буферами, без создания Python-объекта на каждый токен.

Границы документов, sequence-фрагментов и packs, labels и единственный конечный
хвост совпадают с эталоном независимо от числа процессов. В частности, это не
независимая упаковка каждого входного shard с отдельным хвостом. В памяти находятся
массивы длин/смещений O(число документов) и ограниченные буферы каждой задачи;
все токены корпуса не собираются в RAM. На больших данных скорость ограничивается
также общей файловой системой; увеличение workers не гарантирует линейный выигрыш.

Ключ кеша включает входной dataset fingerprint, всю resolved policy,
split_long и версию реализации. Stream cache не переиспользуется как isolated.
Готовый dense cache доступен при другом числе workers без повторного сканирования;
FileLock защищает его построение, а ready manifest публикуется после записи shards.
Новый путь не использует старый generator cache. Для CPT без substitution пропущен
промежуточный map, который дублировал input_ids в labels; packer использует исходные
токены. Индекс документа задаёт seed активной substitution независимо от workers.
В eval substitution отключена, поэтому training-fragments после расширения токенов
не обязаны совпадать с eval-fragments.

`main_process_first` в training entrypoint единолично управляет barriers подготовки.
Внутри `ensure_tokenized`/`ensure_packed` дополнительных barriers нет. При общем
cache остальные ranks читают результат rank 0; при локальном cache отсутствующий
результат строится на доступной каждому узлу файловой системе. Детерминизм требует
одинаковых данных, config и библиотек на узлах. Tokenizer/preprocessing callbacks
и состояние входных Arrow datasets также участвуют в HF fingerprints.

Для большого Slurm allocation используйте отдельное подготовительное задание и
`data.prepared_dataset_dir`: Trainer читает ссылки на готовые Arrow-файлы, не
вызывая factories повторно. Несовместимый manifest приводит к ошибке, а не к
повторной подготовке на выделенных GPU. Порядок запуска и ограничения manifest
описаны в [подготовке данных](DATASET_PREPARATION.md#раздельная-подготовка-и-обучение).

Для нового CPT `unified_dataset.compute_stats=true` делает дополнительный
последовательный проход и пишет `dataset_stats/stats_train.json`: policy, schema,
число packs/segments/documents, input/target tokens, fill, substitutions и частоты
токенов с учётом `freeze_idx`. Память частот — O(vocabulary). Legacy histogram
outputs относятся к прежнему `PackedDataset` пути.

Checkpoint сохраняет `ruadapt_packing.json` с resolved policy; resume с другой
policy отклоняется. Старый checkpoint без metadata вызывает предупреждение.
Metadata не содержит hash корпуса и не гарантирует, что пользователь не заменил
исходные данные. Optimizer, scheduler и RNG сохраняются штатным Trainer.

Двухпроцессный CPU/Gloo тест покрывает все четыре режима: preprocessing barriers,
train, SGD update против однопроцессного эталона при разных counts между ranks,
token-weighted eval с неполным последним batch и точный resume.
Это не NCCL, не multi-node и не проверка шардирования. FSDP/DeepSpeed подключаются
существующим Trainer/Accelerate путём; аппаратных smoke новой матрицы для них
нет. Частичная заморозка строк embeddings по-прежнему разрешена только для
single-device/DDP и требует `embed_weight_decay=0`.

## Варианты дообучения и оборудование

Полное дообучение, LoRA, частичная заморозка embeddings и FP8 storage меняют
память весов/gradients/optimizer, но не заменяют правила контекста и labels.
FP8 storage — хранение замороженной базы с деквантизацией для BF16 вычисления,
а не W8A8 training и не FP8 attention. См. [FP8.md](../FP8.md).

### A100

A100 поддерживает FA2; SDPA выбирает подходящий kernel. Hopper FA3 на A100
не применяется. Размер VRAM определяет допустимые длины/B, но не добавляет
поддержку kernels.
[Аппаратные требования FlashAttention](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/README.md).

| Задача | Варианты | Ограничения |
| --- | --- | --- |
| CPT stream, full/LoRA | SDPA, FA2 | Обычный causal stream |
| SFT none, full/LoRA | SDPA, FA2 | Обычный padded batch |
| CPT/SFT isolated, full/LoRA | SDPA, FA2 | Общий adapter; SDPA quadratic mask, Qwen3.5 требует FLA/convolution |
| Частичная заморозка embeddings | Backend соответствующего режима | Single-device/DDP; ограничения optimizer остаются |
| LoRA с FP8 storage | Attention отдельно | В HF 5.9 loader на A100 база автоматически деквантизируется |
| SimPO/GRPO, full/LoRA | Потенциально SDPA/FA2 для training attention | Активный training core не реализует эти задачи |

У A100 compute capability 8.0. В используемом HF FineGrainedFP8 loader при
capability ниже 8.9 включается `dequantize=True`; RuAdapt применяет FP8 storage
patch уже после загрузки. Успешный `verify_fp8_storage` не требует ненулевого
числа FP8-модулей, поэтому проверка сама по себе не доказывает экономию памяти
на A100; summary может содержать `fp8_frozen=0`.
[NVIDIA GPU capabilities](https://developer.nvidia.com/cuda/gpus),
[HF FP8 quantizer](https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/quantizers/quantizer_finegrained_fp8.py).
Это ограничение loader, не результат нового A100 run.

### H100

| Задача | Варианты | Ограничения |
| --- | --- | --- |
| CPT stream, full/LoRA | SDPA, FA2, FA3 | Модель и установленный kernel должны поддерживать выбранный dtype |
| SFT none, full/LoRA | SDPA, FA2, FA3 | Обычный batch, FA3 требует совместимой сборки |
| CPT/SFT isolated, full/LoRA | SDPA, FA2, FA3 | Общие segment boundaries; новые FA3 сочетания аппаратно не проверены |
| SFT LoRA + FP8 storage | В production v9 — FA3 | Исторический Qwen3.5-27B run; это не тест всех новых режимов |
| CPT LoRA + FP8 storage | Общие loader/factory допускают сочетание | Отдельный production CPT run этим документом не подтверждается |
| Full fine-tuning FP8 базы | Не предоставляется `fp8_storage` | Этот режим требует замороженной базы/LoRA |
| Частичная заморозка embeddings | Backend соответствующего режима | Те же single-device/DDP и optimizer ограничения |
| SimPO/GRPO | Потенциально SDPA/FA2/FA3 для full attention | Trainers/objectives не интегрированы |

[Конфиг SFT v9](../configs/sft_v9_fp8_4gpu.json) не является готовым конфигом
full CPT: полное обучение требует gradients и optimizer states всей базы.
Производительность FA3 относительно FA2 сравнивается на одном оборудовании,
одинаковых данных и настройках; смена A100 на H100 меняет больше, чем kernel.

### SimPO и GRPO

В активном `ruadapt/training` нет SimPO/GRPO trainer, objective, factory или CLI
режима. Архивные preference-скрипты не составляют поддержку нового общего
packing interface.

SimPO работает с предпочтениями chosen/rejected, используя нормированные
log-probabilities ответов. Независимо упаковать ответы можно лишь при сохранении
принадлежности паре, границ prompt/response и длины каждого ответа; общий
token-mean causal CE не реализует этот objective.
[SimPO](https://arxiv.org/abs/2405.14734).

GRPO включает генерацию группы completions, rewards, group-relative advantages
и policy updates. Packing training tokens и batching rollout с KV/recurrent
cache — разные части выполнения. Выбор training FA не выбирает автоматически
rollout backend; потребуются metadata групп и response-level statistics.
[GRPO trainer](https://huggingface.co/docs/trl/grpo_trainer).
Текущий `PackingPolicy` решает задачу независимых causal sequences для CPT/SFT,
а не задаёт эти дополнительные objectives.

## Проверки, производительность и границы поддержки

Основные воспроизводимые проверки:

```bash
pytest tests/training/test_document_packing.py tests/training/test_packing_training.py -v
PYTHONPATH=. python -m torch.distributed.run --standalone --nproc_per_node=2 \
  dev/validate_packing_ddp.py --output /path/to/fresh/output
```

GPU matrix проверяет loss/gradients для всех четырёх режимов, моделей и
backend, с full/LoRA. Неподходящее оборудование даёт skip. Дополнительно проверены
factories/Trainer всех режимов и Qwen3.5 LoRA с gradient checkpointing и Liger
на SDPA/FA2. Модульные тесты покрывают boundaries, tails, labels substitution,
PAD=EOS, worker determinism и SDPA sliding mask.

Packing повышает долю полезных токенов, но `input_tokens` и supervised targets
различаются. SFT prompts, padding и первые labels isolated-сегментов не входят
в loss. Сравнение скорости должно фиксировать длины сегментов, B, accumulation,
fill, target tok/s, peak VRAM/RAM и communication; одно input tok/s не описывает
objective. Padding в текущем isolated collator сохраняется. Для Qwen3.5 скорость
FLA и MLP может ограничивать эффект ускорения full attention.

Проверенный стек: torch 2.11.0+cu128, Transformers 5.9.0, PEFT 0.19.1,
FLA 0.4.0, Liger 0.8.1. Ссылки FA API относятся к официальному тэгу v2.8.3,
включая `hopper`; это не утверждение о версии установленного wheel FA3.
Минимальные зависимости `pyproject.toml` шире проверенного стека и сами по себе
не гарантируют packing integration на каждой версии. Входная подготовка
токенизаторов не переписана; сохраняются её отдельные ограничения.

Нет аппаратного подтверждения новой матрицы на A100/H100, FA3, multi-node NCCL,
FSDP и DeepSpeed; нет production throughput benchmark нового adapter. Не
предоставляются multimodal packing, sequence/context parallelism,
SimPO/GRPO или W8A8 full-training. Удаление padding из isolated flattened packs
доступно через tail=keep; dense с tail=drop создаёт только полные packs.


Новые стратегии дополнительно проверяются в tests/training/test_packing_strategies.py:
overflow, dense/whole, tail, границы attention на Qwen3/Qwen3.5 SDPA/FA2,
переменная длина и фрагменты без targets с Liger и без него. FA3 проверки пропускаются
без Hopper. dev/validate_packing_ddp.py проверяет dense/keep на двух CPU Gloo ranks:
эквивалентность SGD, token-weighted eval и точное возобновление checkpoint.
Старая packing metadata из четырёх полей принимается только с эквивалентной старой
укладкой и overflow; переход на dense/keep требует нового запуска.
