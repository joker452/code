module RoseTreeType ( RoseTree (..) ) where

data RoseTree a = Node a [RoseTree a] deriving (Show, Eq)
