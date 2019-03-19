
import MyIntList
import Prelude hiding(length, head, tail, init, last, take, drop)
empty = Nil
l1 = Cons 1 Nil
l2 = Cons 1 (Cons 2 Nil)
main :: IO ()
main = do
    print $ length empty == 0
    print $ length l1 == 1
    print $ length l2 == 2
    --print $ head empty
    print $ head l1 == 1
    print $ head l2 == 1
    --print $ tail empty
    print $ tail l1
    print $ tail l2
    --print $ init empty
    print $ init l1
    print $ init l2
    --print $ last empty
    print $ last l1 == 1
    print $ last l2 == 2
    print $ take (-1) empty
    print $ take 0 empty
    print $ take 2 empty
    print $ take (-2) l1
    print $ take 0 l1
    print $ take 1 l1
    print $ take 2 l1
    print $ take (-2) l2
    print $ take 1 l2
    print $ take 2 l2
    print $ take 3 l2
    print $ drop (-1) empty
    print $ drop 1 empty
    print $ drop 0 empty
    print $ drop (-1) l1
    print $ drop 1 l1
    print $ drop 0 l1
    print $ drop 2 l1
    print $ drop 1 l2
    print $ drop 2 l2
    print $ drop 3 l2
