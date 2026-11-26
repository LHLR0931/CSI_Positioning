"""实时定位后端（2026-10-22）：两台 Pi 实时取数 → 四链路对齐 → F-MLP 推理 → HTTP 接口。

模块：
    pcap_stream  流式 pcap 解析（与 data/load_real_stage2 同一套偏移与 BSSID 判据）
    sources      数据源：SSH 实时 tcpdump，或回放已采 pcap（测试用）
    linkbuf      四条链路的帧缓冲与链路状态（在线、帧率、最近一帧距今多久）
    fmlp         F-MLP 的训练与推理（与 crossday / frozen_fusion_d3 同一协议）
    train        当天标定轮解析 + 训练（命令行）
    server       HTTP 服务（命令行），接口见 backend/API_CONTRACT.md
"""
