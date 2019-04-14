-- |The interface of priority queue.
module PriorityQueue where
  --min-Heap (小顶堆 / 最小堆)

-- |Interface for __priority queue__.
--
-- This is the KEY to completing the homework! By thinking about the following specifications of these interfaces, you need to come up with some properties such that:
--
-- 1. a correctly implemented priority queue must hold, and
--
-- 2. a wrongly implemented priority queue may not hold.
--
-- Some examples in this document use @pPrint@ to print data structures in readable format. Get it from http://hackage.haskell.org/package/pretty-simple .
--
-- Remark: The example codes show the behavior of one possible correct implementation of @PriorityQueue@ (In fact, a binomial heap implementation). However, a correct implementation only need to satisfy the specifications described in English. It does not have to precisely imitate the example codes.
class PriorityQueue h where

  -- |Create an empty priority queue. 
  -- 
  -- User usually needs to specify the type. E.g. Use 'BHeap.BHeap' implementation, and set the element type to @Integer@:
  -- 
  -- @
  -- e = empty :: BHeap Integer
  -- > e
  -- BHeap []
  -- @
  empty :: (Ord a) => h a

  -- |@isEmpty hv@ checks whether @hv@ is an empty priority queue. E.g.
  -- 
  -- @
  -- e = empty :: BHeap Integer
  -- > isEmpty e
  -- True
  -- @
  isEmpty :: h a -> Bool

  -- |@insert x hv@ returns a new priority queue created by inserting @x@ into priority queue @hv@. E.g.
  -- 
  -- @
  -- e = empty :: BHeap Integer
  -- h1 = insert 9 e
  -- > h1
  -- BHeap [Node {value = 9, rank = 0, children = []}]
  -- > pPrint h1
  -- BHeap 
  --   [ Node 
  --       { value = 9
  --       , rank = 0
  --       , children = []
  --       } 
  --   ]
  -- @
  insert :: (Ord a) => a -> h a -> h a

  -- |An ultility to create a priority queue: inserting one by one. We guarantee this is correct if @insert@ is correct. E.g. (see also @toList@)
  -- 
  -- @
  -- h3 = insertAll [9,3,6] (empty :: BHeap Integer)
  -- > toList h3
  -- [3,6,9]
  -- @
  insertAll :: (Ord a) => [a] -> h a -> h a
  insertAll [] hv = hv
  insertAll (x:xs) hv = insertAll xs (insert x hv)

  -- |An ultility to create a priority queue: inserting one by one into an empty priority queue. We guarantee this is correct if @insert@ and @empty@ are correct. E.g. (see also @toList@)
  -- 
  -- @
  -- h3 = fromList [9,3,6] :: BHeap Integer
  -- > toList h3
  -- [3,6,9]
  -- @
  fromList :: (Ord a) => [a] -> h a
  fromList xs = insertAll xs empty

  -- |@meld h1 h2@ returns a new priority queue which exactly contains the elements of priority queues @h1@ and @h2@. E.g.
  --
  -- @
  -- h3 = fromList [9,3,6] :: BHeap Integer
  -- h4 = fromList [7,1,2,4] :: BHeap Integer
  -- > toList $ meld h3 h4
  -- [1,2,3,4,6,7,9]
  -- @
  meld :: (Ord a) => h a -> h a -> h a

  -- |@findMin hv@ returns the minimum element in priority queue @hv@.  
  --
  -- When @hv@ is empty, the behavoir is not specified -- It may raise an error.
  --
  -- Common usage: To sort a list of elements, simply @insert@ them one by one into a priority queue, then @findMin@ and @deleteMin@ one by one from the priority queue. See @toList@.
  -- 
  -- Common usage: Smallest / largest k elements.
  -- 
  -- E.g.
  --
  -- @
  -- h3 = fromList [9,3,6] :: BHeap Integer
  -- h4 = fromList [7,1,2,4] :: BHeap Integer
  -- > findMin h3
  -- 3
  -- > findMin h4
  -- 1  
  -- > findMin $ meld h3 h4
  -- 1
  -- @
  findMin :: (Ord a) => h a -> a

  -- |@deleteMin hv@ returns a new priority queue created by deleting the miminum element from priority queue @hv@. 
  --
  -- When @hv@ is empty, the behavoir is not specified -- It may raise an error.
  -- 
  -- E.g.
  --
  -- @
  -- h3 = fromList [9,3,6] :: BHeap Integer
  -- h4 = fromList [7,1,2,4] :: BHeap Integer
  -- > findMin $ deleteMin h3
  -- 6
  -- > findMin $ deleteMin h4
  -- 2  
  -- @
  deleteMin :: (Ord a) => h a -> h a

  -- |An ultility to check out all elements from a priority queue, in ascending order. We guarantee this is correct if @isEmpty@, @findMin@ and @deleteMin@ are correct. See also @fromList@.
  toList :: (Ord a) => h a -> [a]
  toList xs = if isEmpty xs
    then []
    else findMin xs : toList (deleteMin xs)
