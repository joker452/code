module Distance
    (solution
    )where

solution :: Integer -> (Double, Double) -> (Double, Double) -> Double

solution p (x1, y1) (x2, y2)
    | p == 1 = abs (x1 - x2) + abs (y1 - y2)
    | p == 2 = sqrt ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
    | otherwise = max (abs (x1 - x2)) (abs (y1 - y2))