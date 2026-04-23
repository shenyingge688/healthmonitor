@echo off
:: 强制终端使用 UTF-8 编码，防止中文乱码
chcp 65001 >nul

echo 🚀 正在创建新分支并同步

:: 1. 创建并切换到新分支 (如果分支已存在会提示，但不影响后续推送)
git checkout -b 3.0

:: 2. 把所有修改过的文件装进包裹
git add .

:: 3. 给包裹封口，贴上全英文标签
git commit -m "feat: auto update V3 multi-class diagnostic engine"

:: 4. 正式将新分支推送到云端
git push -u origin 3.0
echo.
echo ✅ 上传完成！新分支已成功推送到云端，原 main 分支未受影响。
pause