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
  
