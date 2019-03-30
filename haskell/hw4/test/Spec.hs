import Brute
import Collatz
import Prefix
import Permutations
import Data.List
main :: IO ()
main = do
    print $ sort (getPINs "8") == sort ["5", "7", "8", "9", "0"]
    print $ sort (getPINs "01") == sort ["81", "82", "84", "01", "02", "04"]
    print $ sort (getPINs "359") == sort ["226","228","229","246","248","249"
                             , "256","258","259","266","268","269"
                             , "286","288","289","326", "328","329"
                             , "346","348","349","356","358","359"
                             , "366","368", "369","386","388","389"
                             , "626","628", "629","646","648","649"
                             , "656","658", "659","666","668","669"
                             , "686","688","689"]
    print $ collatzs 1 == [1]
    print $ sort (collatzs 100) == sort [100,50,25,76,38,19,58,29
                                         ,88,44,22,11,34,17,52,26
                                         ,13,40,20, 10,5,16,8,4,2,1]
    print $ isPrefix [1, 2] [1, 2, 3]
    print $ isPrefix [1, 2, 3, 4] [1, 2, 3]
    print $ isPrefix [1, 2, 3] [1, 2, 3]
    print $ isPrefix [1, 2, 3, 4] [1..]
    print $ isPrefix [1..] [1, 2, 3, 4]
    print $ isPrefix [2] [1, 2] == False
    print $ sort (perms [1, 2, 3]) == sort (permutations [1, 2, 3])
    print $ sort (perms [1]) == sort (permutations [1])
    print $ sort (perms ["a", "c", "b"]) == sort (permutations ["a", "b", "c"])
1
