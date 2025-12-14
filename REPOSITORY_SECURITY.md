# 🔒 Repository Security Settings

This guide will help you restrict write access to your repository so only you (and Streamlit Cloud) can make changes, while keeping it public for viewing.

## Step 1: Access Repository Settings

1. Go to your repository: https://github.com/vaibhav-srivastava-1/Dataset-Analysis-Dashboard
2. Click on **"Settings"** tab (top right of the repository page)
3. You'll see various settings categories on the left sidebar

## Step 2: Disable Forking (Optional but Recommended)

1. In Settings, scroll down to **"Features"** section
2. Uncheck **"Allow forking"** 
   - This prevents others from creating copies that might confuse people
   - Note: This doesn't prevent viewing, just prevents forking

## Step 3: Set Up Branch Protection Rules

1. In Settings, click on **"Branches"** in the left sidebar
2. Under **"Branch protection rules"**, click **"Add rule"**
3. Configure the rule:
   - **Branch name pattern**: `main` (or `*` for all branches)
   - Check these options:
     - ✅ **Require a pull request before merging**
     - ✅ **Require approvals**: Set to `1` (you'll approve your own PRs)
     - ✅ **Dismiss stale pull request approvals when new commits are pushed**
     - ✅ **Require status checks to pass before merging** (optional)
     - ✅ **Require conversation resolution before merging**
     - ✅ **Do not allow bypassing the above settings** (IMPORTANT!)
     - ✅ **Restrict who can push to matching branches**: Only allow yourself
   - Under **"Restrict pushes that create matching branches"**:
     - ✅ Check **"Restrict pushes that create matching branches"**
     - Add yourself as the only allowed user
4. Click **"Create"**

## Step 4: Disable Issues and Pull Requests (Optional)

If you want to completely prevent others from contributing:

1. In Settings, go to **"General"** section
2. Scroll to **"Features"**
3. Uncheck:
   - ❌ **Issues** (if you don't want issue reports)
   - ❌ **Pull requests** (if you want to completely block contributions)
   - ❌ **Discussions** (if you don't want discussions)

**Note**: You can keep these enabled if you want feedback, but disable merging from others.

## Step 5: Manage Collaborators

1. In Settings, click **"Collaborators"** in the left sidebar
2. Click **"Add people"**
3. **DO NOT add anyone** unless you specifically want them to contribute
4. If someone is already listed, you can remove them by clicking the gear icon next to their name

## Step 6: Verify Streamlit Cloud Access

Streamlit Cloud uses GitHub's API with read-only access for deployments, so it will continue to work even with these restrictions.

## Step 7: Test Your Settings

1. Try to push a change (you should be able to)
2. The repository remains public (anyone can view)
3. Others cannot push directly to main
4. Others cannot create pull requests (if disabled) or they require your approval

## Quick Summary

✅ **What's Protected:**
- Only you can push to main branch
- Pull requests require your approval
- Repository remains public (viewable by anyone)

✅ **What Still Works:**
- You can push changes normally
- Streamlit Cloud can deploy automatically
- Anyone can view the code

❌ **What's Blocked:**
- Others cannot push directly
- Others cannot merge without your approval
- Others cannot bypass protection rules

## Alternative: Make Repository Private

If you want complete privacy:

1. In Settings → General
2. Scroll to **"Danger Zone"**
3. Click **"Change visibility"** → **"Make private"**
4. Only you (and people you invite) can see it

**Note**: Streamlit Cloud free tier requires public repositories. If you make it private, you'll need Streamlit Cloud for Teams (paid).

---

**Your repository is now secure!** Only you can make changes, while it remains publicly viewable.

