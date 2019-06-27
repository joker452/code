# 《函数式语言程序设计（2019春）》期末项目测试集

本仓库包含了《函数式语言程序设计（2019春）》期末项目公开的测试集。
目录结构符合 stack 项目的规范，可用其替换你的项目中的 `test/` 目录。

## 测例

每个测例定义为一个 `Example` 类型，参见 `Util.hs` 中的定义：

```haskell
data Example = Example { name :: String
                       , program :: Program
                       , expectedType :: Maybe Type
                       , expectedResult :: Result
                       }
                       deriving (Show)
```

其中：

- `name` 为测例名称（与源代码文件名一致）
- `program` 为测例程序的语法树（`Program` 类型）
- `expectedType` 为正确的类型推断结果： `Just t` 说明类型正确且为 `t`；`Nothing` 说明类型错误
- `expectedResult` 为正确的求值结果（`Result` 类型）

## 测试集

测试集分为两个部分：

- `ExamplesCore.hs` 包含了基础功能的测试程序
- `ExamplesAST.hs` 包含了ADT的测试程序

## 源代码

为了让阅读测例更加轻松，我们设计了一种 OCaml 和 Haskell 混搭风格的具体文法，并将所有测例都采用这种形式书写出来。
详见 `benchmarks/`。
其中 `ExamplesCore.hs` 的测例对应于 `benchmarks/simple` 和 `benchmarks/complex`；
而 `ExamplesADT.hs` 的测例对应于 `benchmarks/adt` 和 `benchmarks/adt-ill-typed`。

### 具体文法

```grammar
<program> ::= <adt>+ `;` <expr> | <expr>
<adt> ::= `type` <ident> `=` <constructor>*|
<constructor> ::= <ident> <ty>*

<ty> ::= `(` <ty> `)`
       | `[` <ty> `]`
       | `bool` | `int` | `char`
       | `string`
       | <ident>
       | <ty> `->` <ty>
<typed> ::= `(` <ident> `:` <ty> `)`

<expr> ::= `(` <expr> `)`
         | `[` <expr>*, `]`
         | <bool> | <int> | <char>
         | <string>
         | <ident>
         | <unary-op> <expr>
         | <expr> <binary-op> <expr>
         | `fun` <typed>+ `=>` <expr>
         | `if` <expr> `then` <expr> `else` <expr>
         | `let` <ident> `=` <expr> `in` <expr>
         | `let` `rec` <ident> <typed>+ `:` <ty> `=` <expr> `in` <expr>
         | `match` <expr> `with` <branch>+|
<unary-op> ::= `-` | `not`
<binary-op> ::= EMPTY
              | `*` | `/` | `%`
              | `+` | `-`
              | `::`
              | `=` | `<>` | `>` | `>=` | `<` | `<=`
              | `not`
              | `&&`
              | `||`
<branch> ::= <pat> `=>` <expr>

<pat> ::= <bool> | <int> | <char>
        | <string>
        | `(` <pat> `)`
        | `[` <pat>*, `]`
        | <ident> <pat>*
        | <pat> `::` <pat>

<bool> ::= `true` | `false`
<int> ::= DECIMAL
<ch> ::= `'` CHARACTER `'`
<str> ::= `"` CHARACTER* `"`
<ident> ::= [_A-Za-z][_A-Za-z0-9']
```

为了简单，我们用：

- `+` 表示一个文法符号出现至少一次
- `*` 表示一个文法符号出现任意多次（可能零次）
- `*|` 表示一个文法符号出现任意多次，且被 `|` 隔开
- `+|` 表示一个文法符号出现至少一次，且被 `|` 隔开
- `*,` 表示一个文法符号出现任意多次，且被 `,` 隔开

该语言的中缀符号优先级和结合性符合 OCaml 惯例。
支持 C 系风格的单行注释：以 `//` 开头。
支持 OCaml 风格的多行注释：以 `(*` 开头且以 `*)` 结尾。

### 列表类型擦除

由于 AST 中未定义列表相关的类型和表达式，故在测例的 AST 表示中已将其擦除，并用底层的 ADT 结构表达。

类型转换规则为：若 `t` 擦除为类型 `t'`，则 `[t]` 擦除为类型 `t'' = TData s'`，其中 `s' = "[" ++ show t' ++ "]"`。
且该类型自动对应于底层的 ADT 声明

```haskell
ADT s' [ ("[]@" ++ s, [])
       , ("::@" ++ s, [t', t'']) ]
```

例如，`[int]` 会被翻译为 `TData "[int]"` 与相关联的 ADT 声明（写成具体文法）

```haskell
type [int] = []@int | ::@int int [int]
```

表达式转换规则为：

- `[]` 擦除为调用对应 ADT 声明中的 "[]@?" 函数，其中 "?" 需要替换为列表元素类型，即：`EVar "[]@?"`。
- `e1 :: e2` 擦除为调用对应 ADT 声明中的 "::@?" 函数，其中 "?" 需要替换为列表元素类型，即：`EApply (EApply (EVar "::@?") e1') e2'`，其中 `e1'` 和 `e2'` 为分别擦除 `e1` 和 `e2` 的结果。
- `[e1, e2, ..., en]` 是 `e1 :: e2 :: ... :: en :: []` 的语法糖。

模式转换规则为：

- `[]` 擦除为对应 ADT 声明中的 "[]@?" 构造函数，其中 "?" 需要替换为列表元素类型，即：`PData "[]@?"`。
- `p1 :: p2` 擦除为对应 ADT 声明中的 "::@?" 构造函数，其中 "?" 需要替换为列表元素类型，即：`PData "::@?" [p1, p2]`，其中 `p1'` 和 `p2'` 为分别擦除 `p1` 和 `p2` 的结果。
- `[p1, p2, ..., pn]` 是 `p1 :: p2 :: ... :: pn :: []` 的语法糖。

此外，字符串被解释为字符列表，然后再根据上述规则转换。
