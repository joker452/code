import GCD(myGCD)
import BinTree
import BinaryHeight(height)
import BinaryBalance(isBalance)

main :: IO ()
main = do
    print $ myGCD 2 4 == 2
    print $ myGCD (-2) 4 == 2
    print $ myGCD 13 13 == 13
    print $ myGCD 13 (-13) == 13
    print $ myGCD (-37) (-600) == 1
    print $ myGCD 20 100 == 20
    print $ myGCD 624129 2061517 == 18913
    print $ myGCD (-10^30 + 2^30) (10^100 - 2^31) == 2147483648
    print $ height Nil == 0
    print $ height (Node (Node (Node Nil Nil 0) Nil 0) Nil 0) == 3
    print $ height (Node Nil (Node (Node Nil Nil 0) Nil 0) 0) == 3
    print $ isBalance Nil
    print $ isBalance (Node Nil Nil 0)
    print $ isBalance (Node Nil (Node (Node Nil Nil 0) Nil 0) 0)
    print $ isBalance (Node (Node (Node Nil Nil 0) Nil 0) (Node (Node Nil Nil 0) (Node Nil Nil 0) 0) 0)






