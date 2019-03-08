module Main where
import qualified APlusB(solution)
import qualified LCM(solution)
import qualified Distance(solution)
eq :: Double -> Double -> Double -> Bool
eq tolerance a b = tolerance > abs(a - b)
main :: IO()
main = do
    print $ APlusB.solution 2 2 == 4
    print $ APlusB.solution (-1) 2 == 1
    print $ LCM.solution 6 4 == 12
    print $ LCM.solution 6 7 == 42
    print $ LCM.solution 6 8 == 24
    print $ eq 1e-6 (Distance.solution 1 (1, 2) (2, 3)) 2.0
    print $ eq 1e-6 (Distance.solution 2 (1, 2) (2, 3)) 1.4142135623730951
    print $ eq 1e-6 (Distance.solution 3 (1, 2) (2, 4))  2

