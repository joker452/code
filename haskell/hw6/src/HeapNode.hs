module HeapNode where

-- |The node type of the Binomial Heaps.
--
-- When doing QuickCheck homework, you should NOT access @Node@ type -- the implementation that your properties check may not have @Node@!
data Node a = Node { 
  -- |@value n@ returns the element contained in node @n@.
  --
  -- The element can be of any type you specify: only key; key and value;...
  value :: a,
  -- |The rank of this node, used by binomial heap.
  rank :: Int,
  -- |The children of this node, used by binomial heap.
  children :: [Node a]
} deriving (Show)