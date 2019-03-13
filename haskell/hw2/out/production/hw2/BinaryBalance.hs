module BinaryBalance where

import BinTree
import qualified BinaryHeight(height)
isBalance::BinTree.BinTree a -> Bool
isBalance Nil = True
isBalance (Node left right _) = isBalance left &&
                                isBalance right &&
                                (abs (BinaryHeight.height left - BinaryHeight.height right) < 2)
