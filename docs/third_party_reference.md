# 第三方参考与本地材料

GreyHound参考源码：[wutianyuan1/Greyhound](https://github.com/wutianyuan1/Greyhound)，本地核查commit `d641b3997c17de700e1ef7d37e353418df7e336d`。需要查看对照源码时，在项目根目录执行：

```bash
git clone https://github.com/wutianyuan1/Greyhound.git third_party/Greyhound
git -C third_party/Greyhound checkout d641b3997c17de700e1ef7d37e353418df7e336d
```

该目录不纳入本项目Git历史；第三方代码保留其原许可证。当前FLASH运行代码不依赖该checkout；对照文档的源码链接需要本地检出后查看。

`中文论文.pdf`、`greyhound.pdf`作为本地参考材料保存，不随本仓库发布。论文要求转录、方法边界与对照证据入口见`chapter3_requirements.md`、`method.md`及`flash_improvement_log.md`。
