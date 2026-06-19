# setup_ssh_and_deploy.ps1
# Run this ONCE to install your SSH key on the server.
# After this, all future deployments are passwordless.

$SERVER = "root@67.205.185.84"
$PUB_KEY = Get-Content "C:\Users\giris\.ssh\id_alpha_edge.pub"

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  AlphaEdge — One-Time SSH Key Setup" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Step 1: Installing public key on server..." -ForegroundColor Yellow
Write-Host "(You will be asked for the root password ONE TIME)" -ForegroundColor Yellow
Write-Host ""

# Copy public key to server (requires password this one time)
$cmd = "mkdir -p ~/.ssh && echo '$PUB_KEY' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && chmod 700 ~/.ssh && echo 'KEY_INSTALLED_OK'"
$result = echo $PUB_KEY | ssh -o StrictHostKeyChecking=no $SERVER "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && chmod 700 ~/.ssh && echo KEY_INSTALLED_OK"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "SSH key installed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Step 2: Deploying latest code..." -ForegroundColor Yellow
    
    ssh -o StrictHostKeyChecking=no -i "C:\Users\giris\.ssh\id_alpha_edge" $SERVER @"
echo '=== Pulling latest code ==='
cd /root/alpha_edge && git pull origin main

echo ''
echo '=== Restarting AlphaEdge service ==='
systemctl restart alphaedge

echo ''
echo '=== Service Status ==='
systemctl status alphaedge --no-pager -l

echo ''
echo 'Deployment complete!'
"@
} else {
    Write-Host "Key installation failed. Please check your password." -ForegroundColor Red
}
