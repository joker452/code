module MyIntList where

import Prelude hiding (length, head, tail, init, last, take, drop)

data IntList = Cons Int IntList | Nil
    deriving (Show)

length :: IntList -> Int

length Nil = 0
length (Cons x xs) = 1 + length xs

head :: IntList -> Int
head Nil = error "invalid input for head"
head (Cons x _) = x

tail :: IntList -> IntList
tail Nil = error "invalid input for tail"
tail (Cons _ xs) = xs

init :: IntList -> IntList
init Nil = error "invalid input for init"
init (Cons x Nil) = Nil
init (Cons x xs) | length xs == 0 = Nil
                 | length xs == 1 = Cons x Nil
                 | otherwise = Cons x (init xs)

last :: IntList -> Int
last Nil = error "invalid input for last"
last (Cons x xs) = case xs of Nil -> x; _ -> last xs

take :: Int -> IntList -> IntList
take _ Nil = Nil
take n (Cons x xs) | n <= 0 = Nil
                   | otherwise = Cons x (take (n - 1) xs)

drop :: Int -> IntList -> IntList
drop _ Nil = Nil
drop n (Cons x xs) | n <= 0 = Cons x xs
                   | otherwise = drop (n - 1) xs
