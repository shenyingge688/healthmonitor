@echo off
:: 防范国际化与中文字符重构导致的乱码呈现问题
chcp 65001 >nul
echo 🚀 正在一键同步 HybridWarningNet 多分类诊断机代码到 GitHub...
:: 索引目前文件夹下的全部迭代改动并置入缓存区
git add .
:: 追加标准化工程提交动作，使用自动 feat 标签头规范树形图历史
git commit -m "feat: "
:: 执行推图强制动作，连接 origin 地址上的 main 分支
git push origin main
echo.
echo ✅ 上传完成！代码已成功推送到云端。
Pause