# Small Model Figures

Generated from the completed small-model run under `real_runs/small`.

For `gpu_1`, PCIe RX/TX is interpreted as CPU<->GPU transfer. For `gpu_2`, `nvidia-smi dmon` reports aggregate PCIe RX/TX per GPU, so the plot shows CPU<->GPU plus GPU<->GPU/NCCL traffic together; it cannot split those paths after the fact.

CPU and GPU timeline figures include shaded phases with time ranges. CPU active ranges are inferred from server CPU activity; GPU active ranges are inferred from first and last nonzero GPU SM or PCIe samples.

- Summary CSV: `generated_figures/small_summary.csv`
- Overview: `overview/small_performance_memory_overview.png`
- Runtime comparison: `overview/small_runtime_comparison_by_workload.png`

## Per-Run Figures

### cpu_64
- chatbot: `cpu_64/small_cpu_64_chatbot_resource_timeline.png`
- code_generation: `cpu_64/small_cpu_64_code_generation_resource_timeline.png`
- long_conversation: `cpu_64/small_cpu_64_long_conversation_resource_timeline.png`

### gpu_1
- chatbot: `gpu_1/small_gpu_1_chatbot_transfer_memory_timeline.png`
- code_generation: `gpu_1/small_gpu_1_code_generation_transfer_memory_timeline.png`
- long_conversation: `gpu_1/small_gpu_1_long_conversation_transfer_memory_timeline.png`

### gpu_2
- chatbot: `gpu_2/small_gpu_2_chatbot_transfer_memory_timeline.png`
- code_generation: `gpu_2/small_gpu_2_code_generation_transfer_memory_timeline.png`
- long_conversation: `gpu_2/small_gpu_2_long_conversation_transfer_memory_timeline.png`

