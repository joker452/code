# Haskell  
结构相等  
Stu id1 n1 s1与Stu id2 n2 s2各个域对应相等  
data Stu = Stu String \_ \_  
        \| Normal xxx  
Specialise 特别的拷贝，提前生成好，不再每次遇到具体类型时生成，不同类型的机器码不同。  
类型类是语法糖  
模块名需要大写  
mapping x -> T  x的类型是T
a list of mapping有顺序  
DList [Char]->[Char]  
('\_':) (" \_" ++)
deriving (Eq)是结构相等  
无类型lambda演算可以表达递归，通过不动点  
类型和证明有微妙的关系   

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
\lambda演算没有tag  
盒子有头，里面payload有double
type和构造函数可以重名，二者命名空间不同
bool stack里两个地址  
newtype和有单参数单一构造函数的data都定义一个单值的构造函数。但是newtype定义的构造函数时严格的，而后者则是惰性的。  
newtype右侧被包裹的类型必须是上层类型。  
newtype定义的数据类型只在代码层面发生表面的打包、解包，底层运行时，打包和解包的过程会消失。  
newtype直接把构造函数接受的参数的盒子上的标签换成了自己定义的新标签，相当于把之前数据的盒子更换掉了。  
每个data都有自己的盒子，上面标记着构造函数和类型，同时装着通往被装载的数据的指针，newtype则不然，类型检查阶段二者没有什么区别，但是生成机器码时，编译器知道所有的newtype类型的构造函数只有一个参数，所以可以把参数的盒子换掉。  

```haskell
data D = D Int			-- N undefined causes an error when evaluated
newtype N = N Int     -- D undefined won't as long as you don't try to peek inside

-- record syntax is still allowed, but only for one field
newtype State s a = State { runState :: s -> (s, a) }

-- this is not allowed:
-- newtype Pair a b = Pair { pairFst :: a, pairSnd :: b }

```

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
**折叠是顺序递归操作的抽象**。
Haskell中所有的函数都可以看作是单参数的函数。函数进行部分应用时，返回一个新的函数表示剩下的运算。
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

## 类型类  
类型类的实例声明时不允许写类型标记，为了写文档时，可以使用GHC扩展
```haskell  
{-# LANGUAGE InstanceSigs #-}
data A = A ...
instance Eq A where
	(==) :: A -> A -> Bool                 -- type record
    x == y = ...
```  
只有当试图使用类型类去限定类型时，才真正用到了实例声明中的函数，编译器会自动选择合适的实现，或者传递合适的字典。  
编译器会自动选择符合类型类约束的类型实现，这个过程并不需要数据本身包含额外的信息。大部分情况下，重载方法的具体实现都是编译时静态决定的，这样可以提高性能，同时避免一大类动态类型重载的bug。  
Haskell中多个类型类约束，更加接近Java的接口或者C++中虚函数，因为Haskell中值本身不携带约束信息。    
使用read时，待解析的字符串需要用"包围起来，待解析的字符用'。  
```haskell
read "\"abc\""  :: String
read "'a'"  :: Char
```  
推导Ord之前需要推导Eq。自动推倒的Ord比较规则如下：  
1. 按照构造函数书写的顺序，后面的构造函数创建的值大于前面的。  
2. 相同的构造函数，按照从前向后的位置顺序比较参数大小。  
**永远不要在数据类型声明前加上类型类的约束**  
构造函数不需要参数的数据类型才可以使用Enum推导，推导的Show实例则是直接把构造函数当做字符串输出，推导的边界类实例中上下边界是按照构造函数的书写顺序决定的。  
编译器在推导时会默认选择最通用的类型，往往会带来很大的性能损失，建议给所有用到数值计算的函数加上合适的类型说明或者类型约束，优化计算速度。  
`quot`向0取整，`div`向下取整。`quot/rem`比`div/mod`速度快一些，如果商和余数都需要的话，用`quotRem`比分别调用快，这同样适用于`divMod`。  
默认数字字面量的类型是重载的，相当于代码所有整数数字前面有一个默认的fromInteger，小数数字前面有一个默认的fromRational，大部分时候不用给数字的字面量添加类型说明去区分Int、Integer，Float、Double。  
Rational是Ratio Integer的类型别名。Ratio在Data.Ratio中定义，表示分数类型。%是Ratio a类型的构造函数，参数时分子和分母。Rational表示的数字类型可以用于不受小数精度限制的分数计算。  

## 底
(\_|\_)表示无法计算的值，它可以出现在任何种类的计算中，可能是任何类型，一般用undefined :: a表示。  
对Double#包一层Double的盒子是因为它不包含对应的(\_|\_)表示，任意0，1组合都能够对应合法的Double#。这些类型称为底层类型(unlifted type)，包进盒子后，称为上层类型(lifted type)。  
闭包、函数、构造函数等的通用内存结构。  
```
+------+---------+
|header| payload |
+--+---+---------+
   |
   |     +------------+
   |     | info table |
   +-----> +------------+
         |entry code |
         . ...     .
         . ...     .
```  
Haskell中，所有绑定都具有标记语义（denotational semantics）/引用透明原则(referential transparency)，指随时可以把一个绑定换成它对应的表达式，而不影响程序的求值。一个后果是无法预测函数调用发生在什么时候，甚至不确定它是不是被优化掉了。
命令式编程语言中，函数指的并不是数学意义上的输入输出映射关系，而是一系列步骤组成的执行单元。   
1. :print 试着在不求值的情况下，打印绑定的相关信息。如果遇到的是任务盒，则会在打印出的绑定名称前加上\_来代表是一个未被求值的任务盒。  
2. :sprint 和:print类似，但是不会绑定新的变量名。如果遇到任务盒，简单显示为\_。  
3. :force 和:print类似，但是会强制对绑定的表达式求值。  

完全求值之后的表达式称为常态（normal form），弱常态(weak head normal form)，弱常态有几种情况。  
1. 形如\x-> ...的匿名函数表达式，如\x -> 2 + 2。  
2. 某些不饱和的内置函数，如(+2)和sqrt。  
3. 表达式最外层是一个构造函数，如(1+1, 2+2)，最外层是构造函数（,）。  

这些求值的状态并不一定是最后的常态，可能会含有未求值的部分，但是在底层中的表示都和计算任务不同，计算任务一定是函数应用的形式。  
1. 所有使用构造函数创建的数据都处于弱常态。  
2. 弱常态可以成为一个任务盒，例如匿名函数传递了参数。  
3. 弱常态表达式可能会包含任务和，如(1+1,2+2)中的1+1.  
4. 处于弱常态的表达式成为常态的条件是表达式中的任务盒都被计算成常态。  
5. 任务盒求值的结果往往是一个新的处于弱常态的表达式。  
把计算结果绑定到一个名称上，如果计算结果的类型是固定的话，同一个名称的绑定将不会重复计算，底层会通过对盒子坐标记，来判断任务盒有没有被求值。  
可以通过控制让一个函数获得按值传递的语义，GHC支持很多类型的严格性标注，包括在数据类型声明中和模式匹配中等。  