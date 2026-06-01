@echo off
:: 强制使用标准 UTF-8 编码显示，防止中文乱码
chcp 65001 >nul
echo 🚀 欢迎使用一键分支推送脚本 (带网络自动穿透功能)

:: 1. 动态获取分支名
set /p BRANCH_NAME=请输入要推送的分支名称 (当前您可输入 3.51): 
if "%BRANCH_NAME%"=="" (
    echo ❌ 分支名不能为空！
    pause
    exit /b
)

echo.
echo 📦 正在处理本地代码...
:: 智能判断：如果分支已存在就直接切换过去，如果不存在就新建并切换
git checkout %BRANCH_NAME% 2>nul || git checkout -b %BRANCH_NAME%
git add .
:: 智能提交：如果没有改动它会自动跳过，不影响后续推送
git commit -m "feat: sync branch %BRANCH_NAME%"

echo.
echo 🔍 正在嗅探系统代理环境以突破 443 超时限制...
set SYSTEM_PROXY=
FOR /F "tokens=3" %%A IN ('reg query "HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings" /v ProxyServer 2^>nul') DO (
    set SYSTEM_PROXY=%%A
)

if defined SYSTEM_PROXY (
    echo ✅ 成功捕获加速器代理 [%SYSTEM_PROXY%]，已临时注入 Git。
    git config --local http.proxy http://%SYSTEM_PROXY%
    git config --local https.proxy http://%SYSTEM_PROXY%
) else (
    echo ⚠️ 未检测到加速器，将尝试直连 GitHub...
)

echo.
echo ⏳ 正在向云端推送 [%BRANCH_NAME%] 分支，请稍候...
git push -u origin %BRANCH_NAME%

:: 推送完成后，立刻清理临时代理，防止干扰您电脑上的其他环境
if defined SYSTEM_PROXY (
    git config --local --unset http.proxy
    git config --local --unset https.proxy
    echo 🧹 临时代理环境已安全释放。
)

echo.
echo ✅ 上传流程全部结束！
pause