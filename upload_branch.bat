@echo off
:: 防范国际化与中文字符重构导致的乱码呈现问题
chcp 65001 >nul
echo 🚀 正在创建新分支并同步
:: 执行强行隔离并剥离出一条崭新的版本树主轴线 
git checkout -b 3.5
:: 索引目前文件夹下的全部迭代改动并置入缓存区
git add .
:: 为此一次独立升级标注整体版本的自述明细摘要
git commit -m "feat: new data v3.51"
:: 新增指针跟踪节点并正式将游离网络压栈部署到云存储远端
git push -u origin 3.5
echo.
echo ✅ 上传完成！新分支已成功推送到云端，原 main 分支未受影响。
Pause