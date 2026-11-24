# Источники кода, весов и изображения

| Компонент | Источник и условия |
|---|---|
| Основная модель, токенизатор, визуальный энкодер, DSpark | Скрипт загружает оригинальный чекпойнт из зафиксированной ревизии Thinking Machines Lab. [Модель и лицензия авторов](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6) |
| Python-код vLLM и нативная база для SM75 | Патч применяется к закреплённому снимку vLLM. Нативные расширения собираются из форка weicj. [vLLM](https://github.com/vllm-project/vllm/tree/fdda4ee7232366f4490f6484e98533de0289ac98) · [Форк weicj](https://github.com/weicj/vLLM-2080Ti-Definitive/tree/fe9872d764b54ed6a1ac35804a4b6ebd813b7f56) · [Apache 2.0](licenses/vllm-Apache-2.0.txt) |
| Адаптации Inkling для потребительских GPU | Использованы наработки трёх реализаций. [Pento95/HyperInkling](https://github.com/Pento95/HyperInkling) · [snailbrainx/inkling-2-mini-sm80](https://github.com/snailbrainx/inkling-2-mini-sm80) · [peaster/inkling-2-mini-vllm-2x3090](https://github.com/peaster/inkling-2-mini-vllm-2x3090) · [Лицензия и уведомления](runtime/licenses/) |
| Внимание FlashInfer | Шаблоны и изменения заголовков для сборки шести вариантов prefill. [FlashInfer](https://github.com/flashinfer-ai/flashinfer) · [Лицензия](runtime/flashinfer/LICENSE) · [NOTICE](runtime/flashinfer/NOTICE) |
| Групповое умножение skinny на Turing | Адаптация исходника 1Cat-vLLM с ядром из v100-skinny. [Исходник 1Cat](https://github.com/1CatAI/1Cat-vLLM/blob/c15d1446fa00d3196243da5de951d8ed3586de0a/csrc/sm70_turbomind/ops/skinny_moe_qpn_sm70.cu) · [v100-skinny](https://github.com/dnv2003/v100-skinny) · [Apache 2.0](runtime/native/skinny/LICENSE) · [MIT](runtime/native/skinny/LICENSE.v100-skinny) |
| Фотография для проверки зрения | Rachel Michetti, Pikolo Espresso Bar, CC0. Для OCR добавлена нижняя подпись. [Атрибуция и оригинал](assets/fixtures/README.md) |

Версии зависимостей и команды сборки приведены в [инструкции по сборке](docs/build.md).
