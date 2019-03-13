module BinaryHeight where

import BinTree

height :: BinTree.BinTree a -> Integer
height Nil = 0
height (Node left right _) =
    max (height left) (height right) + 1

