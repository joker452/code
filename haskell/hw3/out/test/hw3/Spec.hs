
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
