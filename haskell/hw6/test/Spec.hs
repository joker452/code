import Test.QuickCheck
import PriorityQueue(PriorityQueue(..))
import BHeap(BHeap(..))
import Data.List(sort)

-- You may import more

main :: IO ()
main = do
    -- example, weak properties
    quickCheck prop_1_empty_is_empty
    quickCheck prop_2_findMin_the_only_element
    quickCheck prop_fromList_empty
    quickCheck prop_toList_empty
    quickCheck prop_insert
    quickCheck prop_insertAll
    quickCheck prop_fromtoList
    quickCheck prop_meld_1
    quickCheck prop_meld_2
    quickCheck prop_meld_3
    quickCheck prop_meld_4
    quickCheck prop_findMin
    quickCheck prop_deleteMin
    quickCheck prop_order


    
    -- quickCheck or verboseCheck more properties here!

-- 1. Empty queue should be empty
prop_1_empty_is_empty :: Bool
prop_1_empty_is_empty = isEmpty empty_BHeap_of_Integer

prop_fromList_empty :: [Integer] -> Bool
prop_fromList_empty xs = if null xs then isEmpty (fromList xs :: BHeap Integer)
                                    else isEmpty (fromList xs :: BHeap Integer) == False

prop_toList_empty :: BHeap Integer -> Bool
prop_toList_empty h = if isEmpty h then null $ toList h  else toList h /= []

empty_BHeap_of_Integer :: BHeap Integer
empty_BHeap_of_Integer = empty

-- 2. For all integer n, insert n to an empty priority queue, then findMin from it, the result should be n
prop_2_findMin_the_only_element :: Integer -> Bool
prop_2_findMin_the_only_element n = findMin s == n where
    s = insert n empty_BHeap_of_Integer

prop_insert :: Integer -> BHeap Integer -> Property
prop_insert n h = not (isEmpty h) ==> let hMin = findMin h in
                                          if n < hMin then n == findMin (insert n h) else hMin == findMin (insert n h)

prop_insertAll :: [Integer] -> BHeap Integer -> Bool
prop_insertAll xs h = toList h1 == toList h2
          where h1 = insertAll xs h
                h2 = insertAll (reverse xs) h

prop_fromtoList :: [Integer] -> Bool
prop_fromtoList xs = toList (fromList xs :: BHeap Integer) == sort xs

prop_meld_1 :: BHeap Integer -> BHeap Integer -> Bool
prop_meld_1 hx hy = toList (meld hx hy) == toList hy ++ toList hx

prop_meld_2 :: BHeap Integer -> BHeap Integer -> Bool
prop_meld_2 hx hy = toList (meld hx hy) == toList (meld hy hx)

prop_meld_3 :: BHeap Integer -> BHeap Integer -> Property
prop_meld_3 hx hy = not (isEmpty hx) && not (isEmpty hy) ==> findMin (meld hx hy) == min (findMin hx) (findMin hy)

prop_meld_4 :: BHeap Integer -> BHeap Integer -> Property
prop_meld_4 hx hy = not (isEmpty hx) && not (isEmpty hy) ==> findMin (insert m $ meld hx hy) == findMin (meld hx $ insert m hy)
                    where m = min (findMin hx) (findMin hy) - 1

prop_findMin :: [Integer] -> Property
prop_findMin xs = not (null xs) ==> findMin (fromList xs :: BHeap Integer) == minimum xs

prop_deleteMin :: [Integer] -> Property
prop_deleteMin xs = not (null xs) ==> toList (deleteMin (fromList xs ::BHeap Integer)) == tail (sort xs)

prop_order :: BHeap Integer -> Property
prop_order h = not (isEmpty h) ==> tail (sort $ toList h) == toList (deleteMin h) && minimum (toList h) == findMin h



-- | Generator of @BHeap a@, used to generate random @BHeap@s
instance (Arbitrary a, Ord a) => Arbitrary (BHeap a) where
    arbitrary = do
        avs <- arbitrary -- :: Gen [a] -- see also @vector :: Arbitrary a => Int -> Gen [a]@ 
        return (fromList avs)
