module MaxSubstrSum (
    solution)where

import Data.List(inits, tails)

upd :: (Integer, Integer) -> Integer -> (Integer, Integer)
upd (max_all, max_now) x | max_now' < 0 = (max_all, 0)
                         | otherwise = if max_now' > max_all then (max_now', max_now') else (max_all, max_now')
                           where max_now' = max_now + x

solution :: [Integer] -> Integer
solution xs = fst $ foldl upd (0, 0) xs