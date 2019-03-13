module BinTree where
data BinTree a = Node (BinTree a) (BinTree a) a
                | Nil


