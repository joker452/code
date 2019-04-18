{-# OPTIONS_GHC -F -pgmF htfpp #-}
import MyListType(List(..))
import MyList
import ChurchNumerals(c0,cSucc,cToInt,cPlus,cMult,cExp)

import Test.Framework -- dependencies: HTF
import System.Timeout(timeout)

import Data.List(elemIndex) -- simpler than findIndex
import Data.Maybe(fromJust)

-- TODO in test report, Failure, not expected timeout 2019-03-06
addTimeout test = timeout (3*10^6) test >>= assertJustVerbose "3 seconds timeout exceeded"
testsWithTimeouts = wrap addTimeout htf_thisModulesTests -- magical preprocessing! 2019-03-06

main = htfMain testsWithTimeouts

closeEnough :: Double -> Double -> Bool
closeEnough actual expect = abs (actual-expect) < 1e-6

t0 :: List Int
t0 = Nil
t1 = 1:~Nil
t2 = 1:~2:~Nil
t3 = 1:~2:~3:~Nil
t3' = 1:~2:~4:~Nil
t3'' = 2:~1:~1:~Nil
map' f Nil = Nil; map' f (x:~xs) = f x :~ map' f xs
tn = 1 :~ map' (+1) tn -- 无限长的列表，相当于 [1..]

e1 = t0 == t0
e2 = t2 == 1:~2:~Nil
e3 = t1 == t0
e4 = t1 == tn
e5 = tn /= t1

o1 = t0 <= t0
o2 = t0 < t0
o3 = t3 < t3'
o4 = t3' < t3''
o5 = t3 < tn
o6 = t3' < tn

s1 = show t0
s2 = show t1
s3 = show t3'

-- assertEqual as preprocessor macro https://hackage.haskell.org/package/HTF-0.13.2.2/docs/Test-Framework-HUnitWrapper.html
test_1_1_1 = assertEqual True e1
test_1_1_2 = assertEqual True e2
test_1_1_3 = assertEqual False e3
test_1_1_4 = assertEqual False e4
test_1_1_5 = assertEqual True e5

test_1_2_1 = assertEqual True o1
test_1_2_2 = assertEqual False o2
test_1_2_3 = assertEqual True o3
test_1_2_4 = assertEqual True o4
test_1_2_5 = assertEqual True o5
test_1_2_6 = assertEqual False o6

test_1_3_1 = assertEqual "[]" s1
test_1_3_2 = assertEqual "[1]" s2
test_1_3_3 = assertEqual "[1,2,4]" s3

c2 = \f -> \x -> f (f x)
c3 = \f -> \x -> f (f (f x))

test_2_1 = assertEqual 0 $ cToInt c0
test_2_2 = assertEqual 3 $ cToInt $ cSucc c2
test_2_3 = assertEqual 5 $ cToInt $ cPlus c2 c3
test_2_4 = assertEqual 6 $ cToInt $ cMult c2 c3
test_2_5 = assertEqual 8 $ cToInt $ cExp c2 c3
