`git log --no-merges issue54..origin/master`  
a..b syntax is a log filter that asks Git to display only those commits that are on the latter branch that are not on the first branch.  
`git push -u origin featureB:featureBee`  
This is called a *refspec*.   
`git request-pull origin/master myfork`  
base branch \<url of the repository to pull from\>.  
`git push -f origin featureA`  
If featureA is rebased, then you have to sepcify the `-f` to your push command in order to be able to replace the `featureA` branch on the server with a commit that isn't a descendant of it.  
`git merge --squash <another branch>`  
produce the working tree and index state as if a real merge happened (except for the merge information), but do not actually make a commit. This means your future commit will have one parent only and allows you to introduce all the chnages from another branch and then make more changes before recording the new commit.  
# Maintain a repository  
Create a simple branch name based on the theme of the work you'are going to try. The name should have a namespace, e.g., `sc/ruby_client`.  
`git apply` and `git am` can be used to apply an emailed patch.  
`git diff --word-diff`  
In github, you simply commit and push your topic branch again, and the Pull Request will automatically update. The "Files Changed" tab on a Pull Request will show the "unified" diff, which is basically `git diff master...<branch name>` for the branch this Pull Request is based on.  
One thing to notice is even if the merge **could** be a fast-forward, GitHub will perform a **non-fast-forward** merge.  
If you pull the branch down and merge it locally and push again, the Pull Request will automatically be closed.  
If you want to reference any Pull Request or Issue from any other one, you can simply put #<num> in any comment or description. You can also be more specific if the Issue or Pull request lives somewhere else; write username#<num> if you’re referring to an Issue or Pull Request in a fork of the repository you’re in, or username/repo#<num> to reference something in another repository.  
Users have to have an account and an uploaded SSH key to access the project if the SSH URL is given.  
`git pull <url> patch -1` is a simple way to merge  in a remote branch without having to add a remote.  
`git log <target branch> --not <base branch>` shows you what is introduced.  
`git diff <base branch>...<current branch>`  
This command shows you only the work your curent topic branch has introduced since its common ancestor with the base branch.    
