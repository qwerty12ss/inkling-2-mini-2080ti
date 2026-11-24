# Запуск Inkling 2 Mini на двух RTX 2080 Ti

Рецепт открывает локальный OpenAI-совместимый сервер с одним активным запросом. Оба профиля используют две **RTX 2080 Ti по 22 ГиБ** с NVLink, FP16 KV-кэш и оригинальные NVFP4/MXFP8-эксперты. Три прохода повторяемой группы соответствуют конфигурации в чекпойнте. Размер окна включает и вход, и генерируемый ответ.

| Аргумент запуска | ID в API | Общее окно | KV на каждую карту | Изображения |
|---|---|---:|---:|---|
| `text-512k` | `inkling-2-mini-3loop` | 524 288 | 9,75 ГиБ | только текст |
| `vision-448k` | `inkling-2-mini-3loop-vision-gpu-448k` | 458 752 | 8,625 ГиБ | визуальный энкодер размещён на обеих GPU |

Обычная 2080 Ti с заводскими 11 ГиБ не вмещает эти профили. В контрольном мультимодальном прогоне на 458 496 входных токенов на картах оставалось около 680 и 682 МиБ. Дисплей и параллельные CUDA-процессы тоже расходуют эту память.

## Окружение и веса

Выполните [сборку движка](build.md) и активируйте подготовленное виртуальное окружение. Команды ниже запускаются из корня репозитория. `WORK_DIR` должен совпадать с каталогом, переданным `runtime/prepare.py --work-dir`. Требуются GNU/Linux, Bash, GNU coreutils и клиент Hugging Face `hf` в активном окружении.

```bash
export WORK_DIR="$PWD/.work"
export MODEL_DIR="$WORK_DIR/model"
export DENSE_DIR="$WORK_DIR/dense12"

scripts/fetch-weights.sh
python scripts/pack-dense.py
```

Загрузчик получает служебные файлы, основной шард (19,53 ГБ) и `mtp.safetensors` (0,64 ГБ) из закреплённой ревизии. Распакованные источники, окружение Python и каталоги компиляции потребуют дополнительного места. Чек-лист SHA-256 содержит все 12 файлов для инференса. Условия распространения весов опубликованы авторами. [Ревизия весов и драфтера](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6) · [Контрольные суммы](../scripts/weights.sha256)

Упаковщик на CPU готовит 96 матриц плотных проекций внимания для двух TP-рангов. Формат включает раскладку FP16 для тензорных ядер и данные для восстановления исходных BF16-значений. Пакет занимает около 937 МиБ на диске. При проверке рецепта заново упаковали официальный шард и сравнили тензоры, метаданные и данные восстановления всех 96 матриц с пакетом, на котором проводились замеры. Все значения совпали. При запуске движок также сверяет развёрнутые GPU-матрицы с оригинальными весами. Если чекпойнт и пакет уже подготовлены, их можно проверить отдельно.

```bash
scripts/fetch-weights.sh --check
python scripts/pack-dense.py --check
```

## Выбор устройств и старт

Проверьте топологию. В строке нужной пары на исходном стенде стояло `NV2`; оба направления чтения и записи P2P были доступны. Если в машине больше двух карт, подставьте идентификаторы нужной пары из `nvidia-smi -L`.

```bash
nvidia-smi -L
nvidia-smi topo -m
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w

export CUDA_VISIBLE_DEVICES=0,1
scripts/serve.sh text-512k --dry-run
scripts/serve.sh text-512k
```

`--dry-run` выводит точные аргументы сервера без выделения видеопамяти. При обычном запуске процесс работает на переднем плане и использует `$WORK_DIR/.venv/bin/python`. На системе с несколькими NUMA-узлами можно установить `numactl` и задать `NUMA_NODE` по столбцу NUMA Affinity для выбранных GPU. По умолчанию привязку определяет ОС. Порт задаётся через `PORT`, по умолчанию 8000. Сервер слушает loopback.

Для другого профиля завершите текущий сервер через Ctrl-C, дождитесь освобождения памяти на обеих картах и выполните одну из команд.

```bash
scripts/serve.sh vision-448k
# Парный текстовый контроль с обычной генерацией по одному токену
scripts/serve.sh text-512k --no-drafter
```

Режим K5 задаётся ёмкостью исходного DSpark (`num_speculative_tokens=7`) и бюджетом пяти предложений для одного запроса (`[[1,1,5]]`). При `--no-drafter` сервер запускает основную модель без `--speculative-config`. В обоих случаях префил обрабатывает до 2 048 входных токенов за чанк, интервал сохранения префикса составляет 2 048, используется полный TP2.

## Текстовый запрос

Во втором терминале дождитесь HTTP 200 на `/health`. Первый запуск на новом окружении может занять время на компиляцию ядер. Успешный запрос к `/v1/models` подтверждает загрузку выбранного профиля.

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini-3loop","messages":[{"role":"user","content":"Compute 17 × 23. Give the result."}],"reasoning_effort":"high","thinking_token_budget":64,"max_tokens":160,"temperature":0}'
```

Параметр `max_tokens` включает токены рассуждения. Для длинного сеанса можно указать `"thinking_token_budget":65536,"max_tokens":131072`, если их вмещает оставшаяся часть окна. Например, при входе в 524 032 токена на текстовом профиле помещаются ровно 256 выходных. Клиенту с ограниченным сетевым таймаутом для первого запроса на 512K нужно учесть как минимум несколько минут холодного префила.

## Запрос с изображением

Запустите `vision-448k` и укажите в `IMAGE_FILE` свой PNG или JPEG. Изображение кодируется локально и уходит прямо в движок.

```bash
IMAGE_FILE=photo.jpg python - <<'PY'
import base64, json, mimetypes, os, urllib.request
from pathlib import Path

path = Path(os.environ["IMAGE_FILE"])
mime = mimetypes.guess_type(path.name)[0]
assert mime in {"image/jpeg", "image/png"}
image_url = "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode()
body = {
    "model": "inkling-2-mini-3loop-vision-gpu-448k",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": "Describe this image in one sentence."}
    ]}],
    "temperature": 0, "reasoning_effort": "low", "thinking_token_budget": 32, "max_tokens": 160
}
req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                             data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as response:
    result = json.load(response)
print(result["choices"][0]["message"]["content"])
print(json.dumps(result["usage"], indent=2))
PY
```

В `usage.prompt_tokens` уже входят позиции изображения. `usage.prompt_tokens_details.cached_tokens` показывает повторно использованный префикс. Для сравнения холодных запусков проверяйте, что `cached_tokens=0`. Для оценки тёплого запроса повторите вход с той же солью `cache_salt`. Счётчики `vllm:spec_decode_num_draft_tokens_total` и `vllm:spec_decode_num_accepted_tokens_total` доступны на `/metrics` с активным драфтером. Разность до и после одного запроса показывает фактически предложенные и принятые токены.
