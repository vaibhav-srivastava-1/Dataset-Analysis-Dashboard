# PowerShell script to help deploy to GitHub
# Run this script after creating your GitHub repository

Write-Host "=== GitHub Deployment Helper ===" -ForegroundColor Cyan
Write-Host ""

# Get GitHub username
$githubUsername = Read-Host "Enter your GitHub username"

if ([string]::IsNullOrWhiteSpace($githubUsername)) {
    Write-Host "GitHub username is required!" -ForegroundColor Red
    exit 1
}

# Get repository name
$repoName = Read-Host "Enter repository name (default: data-science-dashboard)"
if ([string]::IsNullOrWhiteSpace($repoName)) {
    $repoName = "data-science-dashboard"
}

Write-Host ""
Write-Host "Setting up remote repository..." -ForegroundColor Yellow

# Add remote
git remote add origin "https://github.com/$githubUsername/$repoName.git" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Remote might already exist. Updating..." -ForegroundColor Yellow
    git remote set-url origin "https://github.com/$githubUsername/$repoName.git"
}

# Set branch to main
git branch -M main

Write-Host ""
Write-Host "Ready to push! Run the following command:" -ForegroundColor Green
Write-Host "git push -u origin main" -ForegroundColor Cyan
Write-Host ""
Write-Host "Note: You may need to authenticate with GitHub." -ForegroundColor Yellow
Write-Host "If prompted, use a Personal Access Token (not your password)." -ForegroundColor Yellow
Write-Host ""
Write-Host "After pushing, go to https://share.streamlit.io to deploy!" -ForegroundColor Green

