module MyIntList where

import Prelude hiding (length, head, tail, init, last, take, drop)

data IntList = Cons Int IntList | Nil
    deriving (Show)

length :: IntList -> Int

length Nil = 0
length (Cons x xs) = 1 + length xs


