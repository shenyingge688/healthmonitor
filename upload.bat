@echo off
:: 强制终端使用 UTF-8 编码，防止中文乱码
chcp 65001 >nul

echo 🚀 正在一键同步 HybridWarningNet 多分类诊断机代码到 GitHub...

:: 1. 把所有修改过的文件装进包裹
git add .

:: 2. 给包裹封口，贴上全英文标签
git commit -m "feat: "

:: 3. 正式推送到云端
git push origin main

echo.
echo ✅ 上传完成！代码已成功推送到云端。
pause