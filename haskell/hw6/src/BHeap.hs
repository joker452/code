-- |An almost totally wrong Binomial Heap implementation of priority queue.
--
-- However, the example properties (@prop_1_empty_is_empty, prop_2_findMin_the_only_element, prop_3_tautology@) in @test/Spec.hs@ are too weak to detect bugs from this implementation.
--
-- Write more, critical properties in @test/Spec.hs@ to detect bugs from this implementation AND more subtle bugs in other possible implementations!
module BHeap where

import PriorityQueue
import HeapNode

newtype BHeap a = BHeap [Node a] deriving (Show)

instance PriorityQueue BHeap where
  -- | Correct
  empty = BHeap []
  -- | Correct
  isEmpty (BHeap l) = null l
  -- | Wrong. Should insert to a well-designed place, not simply to the head.
  insert x (BHeap xs) = BHeap (Node {value=x, rank=0, children=[]}:xs)
  -- | Wrong. Should place the elements to well-designed places.
  meld (BHeap xs) (BHeap ys) = BHeap (xs ++ ys)
  -- | Wrong.
  findMin (BHeap xs) = value $ head xs
  -- | Wrong.
  deleteMin (BHeap xs) = BHeap $ tail xs