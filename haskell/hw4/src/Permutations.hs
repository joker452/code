module Permutations (
    perms)where

myMap :: (a -> [b]) -> [a] -> [b]
myMap _ [] = []
myMap f (x: xs) = f x ++ myMap f xs

insert :: a -> [a] -> [[a]]
insert x [] = [[x]]
insert x (y: ys) = [x: y: ys] ++ map ([y] ++ ) (insert x ys)

perms :: [a] -> [[a]]
perms [] = [[]]
perms (x: xs) = myMap (insert x)  (perms xs)

