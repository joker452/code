# Haskell
模块名需要大写  
每一个文件只能包含一个模块，文件名就是模块名，`Foo.Bar`模块应该在`Foo/Bar.hs`或者`Foo\Bar.hs`文件中。   
```haskell  
import Data.List -- import everything exported from Data.List
import Data.Tree (Tree(Node)) -- import only the Tree data type and its Node constructor from Data.Tree  
import MyModule hiding (remove_e, remove_f) -- algebraic datatypes and type synonyms cannot be hidden.  
import [qualified] MyModuleWithAVeryLongModuleName as Shorty -- work for both qualified and unqualified  
import MyModule as My
import MyCompletelyDifferentModule as My -- multiple modules can be renamed the same as long as there are no conflicting items  

module MyModule2 (Tree(Branch, Leaf)) where -- export constructors in the list, or (Tree(..)) to export all constructors

data Tree a = Branch {left, right :: Tree a} 
            | Leaf a
```
`action`: Haskell中的值的一种，每一个`action`都有自己的类型，如`IO String`。  
被约束的名字可能会与函数符合操作符`.`混淆，一种解决方法是函数复合时总是使用空格，如`reverse . remove_e`。  
`action`从来不会接受参数，比如`putStrLn :: String -> IO ()`是一个`function`，而`putStrLn "hello"`是一个`action`。  
`do`用于连接`action`。  
`return`是一个接受任何类型的参数返回一个返回值是该类的`action`。它可以用于将一个  
类型转换为`action`，使得该类型可以出现在`do block`中。  
f :: Integer -> Integer
f n
  | n <= 0 =1
f n = n * f (n - 1)
haskell 有tag
\lambda盐酸没有tag
盒子有头，里面payload有double
type和构造函数可以重名，二者命名空间不同
bool stack里两个地址
```haskell
data Pair = I Int | D Double  
data Position = MakePosition { getX :: Double, getY :: Double} -- record syntax
p = MakePosition { getY = 3.0, getX = 4.0} 
data () = () 
data (,) a b = (,) a b
data (,,) a b c = (,,) a b c 
data (,,,) a b c d = (,,,) a b c d -- tuple has special syntax
```
`I`,`D`称为标签，在构造函数和模式匹配中使用。  
当代数类型用于枚举时，构造函数同时也是该类型的所有可能取值。  
代数类型和其构造函数必须首字母大写。  
模式匹配本质上是通过找到类型的构造函数来将值进行分解。  
对拥有多于一个参数的构造函数的模式进行匹配时，括号是必须的。如：  
```haskell
data AlgDataType = Constr1 Type11 Type12
                 | Constr2 Type21
                 | Constr3 Type31 Type32 Type33
                 | Constr4  
foo (Constr1 a b) = -- also give names to values that come along with the constr  
foo Constr4 = 
```  
`_`可用于匹配任何模式。  
`x@pat`用于匹配模式`pat`并将匹配的到的值  
模式的一般化定义：  
```haskell  
pat ::= _
      | var
      | var @ ( pat )
      | ( Constructor pat1 pat2 ... patn )  
```  
data List t = E | C t (List t)
t是type variable，必须小写，类型必须大写。
副作用:任何导致求值过程受到其自身以外的影响的事。  
```haskell  
f1 :: Maybe a -> [Maybe a]
f1 m = [m,m]

f2 :: Maybe a -> [a]
f2 Nothing  = []
f2 (Just x) = [x]
```  
pattern matching drives evaluation  
* Expressions are only evaluated when pattern-matched  
* only as far as necessary for the match to proceed, and no farther!  
GHC uses a technique called graph reduction, where the expression being evaluated is actually represented as a graph, so that different parts of the expression can share pointers to the same subexpression. This ensures that work is not duplicated unnecessarily.   

case...of...中的绑定只会在对应的表达式分支里有效。  
Haskell并不区分值和函数。  
对于`$`的理解就是两侧都有()。`$`是右结合，`&`是左结合，优先级为1，定义在Data.Function中。  
::的优先级在各种符号中最低，但\ ... -> ... :: Type中的Type指的不是整个匿名函数的类型，而是->右侧表达式的类型，其实这是因为即Type被函数体包含进去了， 所以如果需要标注整个匿名函数的类型， 请加上括号：  
```haskell
(\x -> x + 1) :: Int-> Int  
```  
闭包：函数体里引用了外围作用域中的自由变量的函数。  
```haskell
giveMeN :: Int -> [a] -> [a]
giveMeN n xs
    | n <= 0 = []
    | otherwise = go $ zip [0..] xs
    where 
      go [] = []
      go ((i, x): xs) = if i `rem` n == 0 then x : go xs    -- use free variable n
                              else go xs
```  
where经常用来书写仅在一个函数内部用到的帮助函数。它如果出现在任何绑定发生的地方，则用来补充说明绑定右侧的表达式。  
```haskell  
case i `rem` 5 of
    0 -> x: rest
    _ -> rest
    where                -- all branches can see
     rest = go xs

case xs of
    (x: xs') -> ...
     where ...            -- local to each branch
    [] -> ...
     where ...
```