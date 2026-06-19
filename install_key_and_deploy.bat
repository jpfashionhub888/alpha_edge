@echo off
echo.
echo ================================================
echo   AlphaEdge - Installing SSH Key on Server
echo ================================================
echo.
echo You will be asked for the root password ONE TIME.
echo After this, all deployments are passwordless.
echo.

type "C:\Users\giris\.ssh\id_alpha_edge.pub" | ssh -o StrictHostKeyChecking=no root@67.205.185.84 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && chmod 700 ~/.ssh && echo KEY_INSTALLED_OK"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo SSH key installed! Now deploying...
    echo.
    ssh -o StrictHostKeyChecking=no -i "C:\Users\giris\.ssh\id_alpha_edge" root@67.205.185.84 "cd /root/alpha_edge && git pull origin main && systemctl restart alphaedge && echo DEPLOY_OK && systemctl status alphaedge --no-pager"
    echo.
    echo ================================================
    echo   Deployment complete!
    echo ================================================
) else (
    echo.
    echo ERROR: Key installation failed. Check password.
)
