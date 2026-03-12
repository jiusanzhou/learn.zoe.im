# Lesson 7: Scheduler — 调度策略

## 目标

1. 实现完整的调度器，统一管理 prefill 和 decode
2. 实现多种调度策略：FCFS、LPM（最长前缀匹配）
3. 资源预算管理：避免 OOM
4. 请求抢占（preemption）机制

## SGLang 调度器核心

```
Scheduler.event_loop_normal():
  while True:
    recv_requests()           # 接收新请求
    process_input_requests()  # 验证、入队
    
    batch = get_next_batch_to_run()  # 核心调度决策
    if batch:
      run_batch(batch)
      process_batch_result(batch, result)

get_next_batch_to_run():
  1. 把上一轮 prefill 结果合并到 running_batch
  2. new_batch = get_new_batch_prefill()  ← PrefillAdder 决定
  3. if new_batch: return new_batch       ← prefill 优先
  4. else: return running_batch           ← decode
```

## 调度策略对照

| 策略 | SGLang 名称 | 描述 |
|------|------------|------|
| 先来先服务 | FCFS | 按到达顺序，默认 |
| 最长前缀 | LPM | 优先调度能复用最多缓存的请求 |
| DFS 权重 | DFS_WEIGHT | 深度优先，同子树的请求一起调度 |
| 最长输出 | LOF | 长输出请求优先（减少 bubble） |
| 优先级 | Priority | 用户指定优先级 |

## 代码

见 `mini_engine/scheduler.py`
