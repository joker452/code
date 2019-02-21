# 基础指令
1. git tag  
git中的标签有两种，一种是轻量标签，另一种是标注标签。  
轻量标签类似于一个不会移动的分支，并不会包含额外的信息。使用时只需要提供一个标签名称即可。  
标注标签会创建一个标签项目。标注标签包含了标注者的信息，打标签的日期，以及标签信息。使用时需使用`-a`或者`-s`选项，后者不仅标注，而且还提供签名，使用`-m`选项指定标签信息。 
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

# 引用
* 在一个引用的后面加上`^`代表其父引用。在Windows环境中，需要对`^`做特殊处理。 

```
git show HEAD^     # will NOT work on Windows 
git show HEAD^^    # OK,  
git show "HEAD^"   # OK
``` 
如果在`^`后面加上一个数字，则用于指明哪一个父引用。这种语法只对合并提交有效，其中第一个父引用是合并时你所在的分支，第二个父引用则是并入的分支。  
`~`也可以用于表示父引用，但是它永远代表第一个父引用，这使得当`~`后面有数字时，它的语义会与`^`不同。这两种方式可以混合使用。  
* `<branch 1>..<branch 2>`用于指明，那些在第二个分支中可达，而在第一个分支中不可达的提交。如果省略其中某一个，那么默认为`HEAD`。注意分支名之间不需要空格。  
在引用前使用`^`或者`--not`可以表明从该分支不可达的意思，示例如下：  

```
git log refA refB ^refC
git log refA refB --not refC
```
* `<branch1>...<bracnh 2>`用于表示集合的差操作。同样分支名之间不需要空格。这种情况下，很有用的一个选项是`--left-right`，可以显示提交来自哪一个分支。  
使用`git reflog`来查看引用的历史记录。其中`reflog`是`reference log`的意思。比较重要得用途是恢复破坏性的操作时查看提交的SHA-1值。用`<reference@{number}>`来指明查看哪一个。使用`git log -g`以`git log`的形式来查看引用历史记录。需要注意的是此记录局限于本地，不会被上传到上游也不会被复制下来，并且在一定时间后会自动删除，所以超过一定时间后，一些破坏性的操作将永远不可恢复！  
* 对单个文件中的改变进行部分筛选
对于`git add`, `git reset`, `git checkout`可以使用`--patch`来完成对一个文件中的改变进行部分的筛选操作。   
* 在不提交的情况下使工作目录和缓存区恢复干净  
`git stash`默认情况下只会打包已被追踪但未缓存的变化以及已缓存的变化，但不会管没有被追踪的变化。使用`-u`参数来打包未被追踪的变化，但此时仍不理会显示忽略的文件，使用`-all`来打包所有的变化。使用`--keep-index`参数会在打包后保存缓存区的状态。打包后恢复的状态是`HEAD`指向的提交。后面没有任何参数时等价于`git stash push`。 
`git stash list`可以查看所有已经保存的变化。  
`git stash apply`用于将变化应用于当前分支上（不一定是保存变化时所在的分支），默认情况下对于之前已经缓存的变化会变为未缓存的状态，需要重新使用`git add`将其缓存，可以使用`--index`参数指明尝试自动回复缓存的状态。如果没有指明使用哪一个存储，则应用最新储存的变化，可以使用`name@{number}`方式来指定应用哪一个变化。应用变化后不会将其删除，`git stash pop`则会直接应用最新的储存的变化并将其删除。  
`git stash drop`删除指定的储存的变化。  
* 清空未被追踪的文件  
`git clean`用于清空未被追踪的文件。使用`-n`参数来输出哪些会被删除但不进行删除操作。`-x`参数会删除那些被忽略的文件，如在.gitignore中声明的文件。默认是递归删除，但只删除文件，使用`-d`参数来删除那些在文件被移除后会变空的文件夹。  
