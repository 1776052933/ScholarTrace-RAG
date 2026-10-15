# M0/M1 版本归档与 M1 工程验收（2026-10-15）

## 版本与可复现性

| 阶段 | Git 位置 | 与当时十题运行源码的关系 |
| --- | --- | --- |
| M0 | `archive/m0-reconstructed`，标签 `m0-reconstructed-2026-10-15`，提交 `2bb0c5e` | 从初始提交补入评测器重建。重建代码指纹 `0a3c960c8c92dd405d4e53d0f90273282838d15c4c72ee1c11bbaeb0c2ca8e10`，与原始运行记录 `f2d4a169aec3cfe5f2fc2d2d509721859463f27fa2bd9a04ed33948145fceae3` 不一致，因此**不是原始 M0 的精确源码**。详细说明见 M0 分支的 `docs/M0_ARCHIVE_STATUS.md`。 |
| M1 | `main`，标签 `m1-accepted-2026-10-15` | 当前源码指纹 `cdcefd281acdd1ac385f95d7e9f751f2be785788b89e953536d7183e2434b58e`，与最终 M1 十题运行记录一致。 |

M1 为参与 `code_hash()` 的源码路径设置了 `.gitattributes` 的 `-text`，保留评测时的原始换行字节。该设置只影响 Git 存储与检出；已核对暂存内容与评测工作区逐字节相同，代码内容没有改动。

两次运行的 QASPER v0.3 开发集 SHA256 均为 `2ae7ee62a65b1c4225791c70de80c2aad4e8998cf1fd4f09a53103db4f21af93`；十个 question ID 与 BGE-M3 模型修订也一致。原始 M0 数值只能作为已保存的历史参考基线，不能把重建分支的新运行结果当成该次运行的严格复现。M0/M1 的质量对照和指标限制见 `evaluation/M1_COMPARISON.md`。

## 本地运行记录

| 运行 | 被 Git 忽略的完整 ZIP | SHA256 |
| --- | --- | --- |
| M0 十题 | `evaluation/runs/_archives/m0_qasper_dev10.zip` | `1420ABEC4BB6154B0193E53575A911AD72084980E7C648888EFFA495C50367A0` |
| M1 最终十题 | `evaluation/runs/_archives/m1_qasper_dev10_final.zip` | `54990D7053890525BE5CE84A519E8E6F2F84472B82D570CCC643E831DE54A7F6` |

两个 ZIP 均包含运行清单、逐题记录、汇总、文档副本和 Chroma 索引；均已通过 ZIP 完整性检查。结构化索引 `evaluation/ARCHIVE_INDEX.json` 记录了压缩包校验和、题目 ID、数据及代码指纹。压缩包含论文内容，**只保存在本机，不提交到 Git**。换机时需连同数据集文件和模型缓存自行迁移；Git 标签仅保存源码与公开评测说明。

## M1 工程验收

环境：Windows，`pywork` Python 3.10.20；测试依赖见 `requirements-test.txt`。命令与结果：

```powershell
$env:PAPER_RUN_REAL_TESTS='0'
python -m pytest -q -m 'not slow' --ignore=paper_downloader/tests/test_config_cache.py --ignore=paper_downloader/tests/test_integration.py
# 378 passed, 2 warnings

python -m pytest -q test_m1_rag.py test_m1_identity.py test_m1_service.py
# 7 passed

python main.py --help
# 正常显示 CLI 参数，无需 DeepSeek 密钥
```

M1 专项测试覆盖两页 PDF 的页码与结构化引用、PDF/TXT 并存时的来源选择、嵌入模型加载失败显式报错、持久化索引的模型指纹和维度校验、无版本旧向量拒绝、论文 ID 去重，以及服务层与 CLI 共用工作流。数据文件 SHA256 和当前代码指纹均再次校验，与 M1 运行清单一致。

**验收边界：** 全仓库直接收集测试时，旧下载器中的 `test_config_cache.py` 与 `test_integration.py` 因导入仓库不存在的 `paper_downloader.src.cache` 而失败；它们被显式排除后，其余 378 项全部通过。两个排除文件的功能尚未验收。真实学术 API、DeepSeek 在线调用和完整 LitSearch/QASPER 数据集评测不属于本次工程验收。两条测试警告分别来自 PyPDF2 弃用提示和一次 Windows 异步 SSL 资源清理；没有导致测试失败。

## 后续复查

在 `main` 标签上复跑上面的离线测试，并运行 `python -c "from evaluation.m0_common import code_hash; print(code_hash())"` 核对 M1 指纹。压缩包用 `Get-FileHash -Algorithm SHA256 evaluation/runs/_archives/*.zip` 核对；正式扩大样本前，先修复下载器旧 `cache` 模块与测试的对应关系。M2 可以在现有 M1 验收结果上继续开发。
