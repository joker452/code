module BinaryBalance where

import BinTree
import BinaryHeight(height)

isBalance::BinTree.BinTree a -> Bool
isBalance Nil = True
isBalance (Node left right _) = isBalance left &&
                                isBalance right &&
                                (abs (height left - height right) < 2)
