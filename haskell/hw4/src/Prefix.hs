module Prefix where

isPrefix :: Eq a => [a] -> [a] -> Bool
isPrefix [] _ = True
isPrefix _ [] = True
isPrefix (x: xs) (y: ys) = x == y && isPrefix xs ys