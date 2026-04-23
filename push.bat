@echo off
echo 
git init
git add .
git commit -m "自动提交：预警"
git branch -M main
git remote add origin https://github.com/shenyingge688/healthmonitor.git
git push -u origin main

echo 
pause