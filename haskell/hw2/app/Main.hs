module Main where

import GCD(myGCD)

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
