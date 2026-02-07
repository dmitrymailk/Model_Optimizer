# TensorRT Inference Optimization: Context Corruption Fix

## Проблема

При бенчмаркинге TensorRT pipeline для генерации изображений обнаружилась проблема:

- **Одиночный запуск** TRT pipeline работал корректно (результат совпадал с PyTorch)
- **Повторные запуски** в benchmark loop давали некорректный результат

### Симптомы

```
TensorRT Pipeline: Avg Latency = 83.91 ms
Speedup: 0.45x  ← TRT медленнее PyTorch!

Качество выхода:
MAE: 0.36 (должно быть < 0.01)
PSNR: 6.95 dB (должно быть > 30 dB)
```

## Исследование

### Гипотеза 1: Создание нового контекста на каждый вызов

```python
def infer(self, feed_dict):
    context = self.engine.create_execution_context()  # Дорого!
    ...
```

**Результат:** Качество корректное, но скорость упала (81ms vs 45ms PyTorch).

### Гипотеза 2: Большой пул контекстов

Создать достаточно контекстов для всех итераций benchmark:

```python
pool_size = 3 * (warmup + benchmark_rounds)  # ~645 контекстов
self._contexts = [self.engine.create_execution_context() for _ in range(pool_size)]
```

**Результат:** Закончилась память GPU. Не подходит.

### Гипотеза 3: Минимальный пул + синхронизация

```python
self._contexts = [self.engine.create_execution_context() for _ in range(3)]

def infer(self, feed_dict):
    torch.cuda.synchronize()  # Полная синхронизация!
    context = self._contexts[self._context_idx]
    self._context_idx = (self._context_idx + 1) % 3
    ...
```

**Результат:** ✅ Работает!

## Корневая причина

TensorRT execution context сохраняет внутреннее состояние между вызовами. При быстрых последовательных вызовах без полной синхронизации:

1. Предыдущий вызов мог не завершиться полностью
2. Внутренние буферы контекста содержат остаточные данные
3. Новый вызов читает/пишет в конфликтующую память

Проблема проявляется только при **повторных вызовах одного контекста** в tight loop.

## Решение

### Оптимальная конфигурация

```python
class TRTEngine:
    def __init__(self, engine_path):
        ...
        # Пул из 3 контекстов для ротации
        self._contexts = [self.engine.create_execution_context() for _ in range(3)]
        self._context_idx = 0
        
    def infer(self, feed_dict):
        # Полная синхронизация GPU перед переключением контекста
        torch.cuda.synchronize()
        
        # Циклическая ротация контекстов
        context = self._contexts[self._context_idx]
        self._context_idx = (self._context_idx + 1) % 3
        ...
```

### Почему это работает

1. **3 контекста** — достаточно для pipeline (encoder + unet + decoder), каждый получает "свой" контекст в рамках одного прогона
2. **`torch.cuda.synchronize()`** — гарантирует завершение всех GPU операций, предотвращает race conditions
3. **Ротация** — контекст не переиспользуется немедленно, есть время на "очистку"

## Результаты

| Метрика | До оптимизации | После оптимизации |
|---------|----------------|-------------------|
| TRT Latency | 83.91 ms | **13.19 ms** |
| PyTorch Latency | 37.78 ms | 44.91 ms |
| Speedup | 0.45x | **3.40x** |
| MAE (качество) | 0.36 ❌ | 0.0008 ✅ |
| PSNR | 6.95 dB ❌ | 54.67 dB ✅ |

## Файлы

- `benchmark_pipeline.py` — основной скрипт с оптимизированным `TRTEngine`
- `compare_outputs.py` — утилита для сравнения качества выходов
- `trt_full_inference.py` — референсный скрипт одиночного запуска

## Рекомендации

1. **Для production:** используйте пул из 2-3 контекстов с ротацией
2. **Для отладки:** сравнивайте с одиночным запуском (`trt_full_inference.py`)
3. **Для максимальной скорости:** рассмотрите CUDA Graphs (если входные размеры фиксированы)
