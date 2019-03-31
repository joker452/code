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

# reset, checkout
HEAD指向当前branch，branch指向某一个commit。  
切换分支或者克隆时，会让HEAD指向最新的分支，然后将索引区用commit的快照填充，之后把索引区的内容拷贝到工作目录。  
checkout会改变HEAD的指向，reset的第一步则是移动HEAD指向的分支指向的commit（注意HEAD也被连带改变了）。  
`git reset --soft`不改变索引区和工作目录，它实际上取消了上一次`git commit`，但上一次的commit object没有被删除。  
`git reset [--mixed]`用新的HEAD指向的提交替换索引区。取消了上一次`git commit`和`git add`。  
`git reset --hard`进一步把工作目录替换为索引区的内容。  
如果给reset提供了文件路径，那么会跳过移动HEAD的步骤，因为HEAD是指针，不能用指针指向提交的一个部分。  
`git reset [--mixed HEAD] file_path`是`git add`的逆操作。可以用`git reset <SHA-number> file_path`指明从哪个commit中拷贝，此操作只修改索引区。  
`git checkout`会改变HEAD，索引区和工作目录。  
`git checkout [<tree-ish>] [--] file_path`带`<tree-ish>`参数时，会会改变索引区和工作目录，否则用索引区内容替代工作目录，两种情况下都不会移动HEAD。  

# 配置Git  
1. /etc/gitconfig 用`--system`配置，系统所有用户。  
2. ~/.gitconfig或者~/.config/git/config，用`--global`配置，针对每一个用户。  
3. .git/config，用`--local`配置，针对每一个仓库。使用`git config`时的默认级别。  
## 客户端
`git config core.autocrlf true`自动转换。  
`git config core.autocrlf input`，提交时转化为LF，checkout时不会做改变。  
Git可以处理6种主要的涉及到空白的问题，三种默认开启（blank-at-eol（行末空格），blank-at-eof（文件尾空行），space-before-tab（行起始处在tab前的空格）），三种默认关闭（indent-with-non-tab（以空格而非tab开头的行），tab-in-indent（用tab而非空格做缩进），cr-at-eol（允许行末出现cr））。  
## 服务器端
`git config --system receive.denyNonFastForwards true`来拒绝强制推送。  

## Git属性  
可以用项目根目录的.gitattributes或者.git/info/attributes文件来配置Git在特定的文件或者目录上的行为。  
可以用属性文件设置两类filter, smudge和clean。前者在checkout之前运行，后者在stage之前运行。
```git
*.pbxproj binary // 声明二进制文件，git不会修复换行符或者在diff时输出  
*.docx diff=word // 任何匹配.docx的文件用word过滤器，需安装doc2txt  

test/ export-ignore // git archive时忽略test文件夹  
*.c filter=indent // 设置indent filter，并指明在smudge和clean时的行为
git config --global filter.indent.clean indent 
git config --global filter.indent.smudge cat  
database.xml merge=ours git config --global merge.ours.drive true // 对特定文件使用特定合并策略
```
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
`git stash`默认情况下只会打包已被追踪但未缓存的变化以及已缓存的变化，但不会管没有被追踪的变化。使用`-u`参数来打包未被追踪的变化，但此时仍不理会显式忽略的文件，使用`-all`来打包所有的变化。使用`--keep-index`参数会在打包后保存缓存区的状态。打包后恢复的状态是`HEAD`指向的提交。后面没有任何参数时等价于`git stash push`。 
`git stash list`可以查看所有已经保存的变化。  
`git stash apply`用于将变化应用于当前分支上（不一定是保存变化时所在的分支），默认情况下对于之前已经缓存的变化会变为未缓存的状态，需要重新使用`git add`将其缓存，可以使用`--index`参数指明尝试自动回复缓存的状态。如果没有指明使用哪一个存储，则应用最新储存的变化，可以使用`name@{number}`方式来指定应用哪一个变化。应用变化后不会将其删除，`git stash pop`则会直接应用最新的储存的变化并将其删除。  
`git stash drop`删除指定的储存的变化。  
* 清空未被追踪的文件  
`git clean`用于清空未被追踪的文件。使用`-n`参数来输出哪些会被删除但不进行删除操作。`-x`参数会删除那些被忽略的文件，如在.gitignore中声明的文件。默认是递归删除，但只删除文件，使用`-d`参数来删除那些在文件被移除后会变空的文件夹。  

# Git内部  
做底层工作的子指令称为plumbing commands，对用户更友好的指令称为porcelain commands。  
Git中存储的内容表示为blob对象（binary big object?，类似于Unix中inode或者文件内容）对文件名的存储使用tree（类似于Unix下文件夹项目），tree还允许把一组文件存储到一起。  

# GitHub搜索技巧  
下面列举一些GitHub上常用的筛选搜索结果的方式  
* `user:<username>`：搜索指定用户  

* `org:<orginaztion name>`:搜索指定组织  

* `license:<license name>`：搜索指定许可证。  

* `pushed/created: relational op <time>`：搜索指定时间条件下有push操作或者在指定时间范围内创建的仓库。  

* `size: realational op <size>`：搜索满足指定大小条件的仓库，单位是K。范围可以用`..`指定，比如`20..50`。

* `stars: realtional op <stars>`：搜索得到星数满足条件的仓库。  

* `in: readme/description/name <keyword>`：限定关键词出现的位置。  