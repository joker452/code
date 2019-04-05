{-# OPTIONS_GHC -F -pgmF htfpp #-}

import Bachelor
import Interleave
import MaxSubstrSum
import Data.List(findIndex)
import Test.Framework
import System.Timeout(timeout)

addTimeout test = timeout (3 * 10 ^ 6) test >>= assertJustVerbose "3 seconds timeout exceeded"
testsWithTimeouts = wrap addTimeout htf_thisModulesTests -- magical preprocessing! 2019-03-06
main = htfMain testsWithTimeouts
f n | n <= 0 = []
    | otherwise = n: n: f (n - 1)

daytimes = ("D", 1) : map (\(x, y) -> (x, y + 1)) daytimes
nights = ("N", 1) : map (\(x, y) -> (x, y + 1)) nights
daysAndNights = interleave daytimes nights
intPairs = interleaveLists [ [ (x, y) | y <- [1..] ] | x <- [1..]]

test_1_1 = assertEqual (bachelor (0: f 4999999)) 0
test_2_1 = assertEqual (interleave [1, 2] [3, 4]) [1, 3, 2, 4]
test_2_2 = assertEqual (take 10 $ interleave [1, 2] [3..]) [1, 3, 2, 4, 5, 6, 7, 8, 9, 10]
test_2_3 = assertEqual (take 6 daysAndNights) [("D", 1), ("N", 1), ("D", 2), ("N", 2), ("D", 3), ("N", 3)]
test_2_4 = assertEqual (findIndex (==("N", 3))  daysAndNights) (Just 5)
test_2_5 = assertEqual (findIndex (==(5, 5)) intPairs) (Just 40)
test_2_6 = assertEqual (findIndex (==(10, 10)) intPairs) (Just 180)
test_3_1 = assertEqual (solution [4, -5, 1, 2, 3, 0, -2, 1]) 6
test_3_2 = assertEqual (solution [-5]) 0
test_3_3 = assertEqual (solution [1.. 10^6]) 500000500000
