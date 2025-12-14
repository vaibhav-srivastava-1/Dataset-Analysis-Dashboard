# Deployment Guide - Streamlit Community Cloud

This guide will help you deploy your Data Science Analysis Dashboard to Streamlit Community Cloud.

## Prerequisites

1. A GitHub account
2. Your code ready in a local repository

## Step-by-Step Deployment

### Step 1: Initialize Git Repository (if not already done)

```bash
git init
git add .
git commit -m "Initial commit: Data Science Analysis Dashboard"
```

### Step 2: Create GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the **"+"** icon in the top right corner
3. Select **"New repository"**
4. Fill in the details:
   - **Repository name**: `data-science-dashboard` (or any name you prefer)
   - **Description**: "Interactive Data Science Analysis Dashboard with Streamlit"
   - **Visibility**: Choose Public (required for free Streamlit Cloud)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click **"Create repository"**

### Step 3: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Run these in your terminal:

```bash
# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/data-science-dashboard.git

# Rename branch to main (if needed)
git branch -M main

# Push your code to GitHub
git push -u origin main
```

### Step 4: Deploy to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in the deployment details:
   - **Repository**: Select your repository (`YOUR_USERNAME/data-science-dashboard`)
   - **Branch**: `main` (or `master` if that's your default branch)
   - **Main file path**: `app.py`
   - **App URL**: Choose a custom subdomain (optional)
5. Click **"Deploy"**

### Step 5: Wait for Deployment

- Streamlit will automatically install dependencies from `requirements.txt`
- The deployment process usually takes 1-2 minutes
- You'll see build logs in real-time
- Once complete, your app will be live!

## Important Files for Deployment

Your repository should contain:
- ✅ `app.py` - Main Streamlit application
- ✅ `requirements.txt` - Python dependencies
- ✅ `README.md` - Project documentation
- ✅ `.gitignore` - Git ignore rules

## Troubleshooting

### Build Fails

1. **Check requirements.txt**: Ensure all dependencies are listed
2. **Check Python version**: Streamlit Cloud uses Python 3.11 by default
3. **Check file paths**: Make sure `app.py` is in the root directory

### App Crashes After Deployment

1. **Check logs**: Click "Manage app" → "Logs" to see error messages
2. **Test locally**: Run `streamlit run app.py` locally to catch errors
3. **Check imports**: Ensure all imports are in requirements.txt

### Common Issues

- **Module not found**: Add missing package to `requirements.txt`
- **File not found**: Use relative paths, not absolute paths
- **Memory issues**: Optimize your code for large datasets

## Updating Your App

After making changes:

```bash
git add .
git commit -m "Update: description of changes"
git push origin main
```

Streamlit Cloud will automatically redeploy your app when you push to the main branch.

## Custom Domain (Optional)

1. Go to your app settings on Streamlit Cloud
2. Click "Settings" → "General"
3. Add your custom domain (requires domain verification)

## Support

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Streamlit Community Forum](https://discuss.streamlit.io/)

## Your App URL

After deployment, your app will be available at:
```
https://YOUR_APP_NAME.streamlit.app
```

Enjoy your deployed Data Science Dashboard! 🚀

