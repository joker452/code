
import RoseTreeType
import RoseTree
import TriangleType
import Trans

tree = Node 1 [Node 2 [Node 3 [], Node 4[]], Node 5 []]
a = Point 1 2
b = Point 2 3
c = Point 3 4
t = Triangle a b c

main :: IO ()
--main = print $ fmap id tree
--       print $ fmap ((+1) . (+2)) tree == Node 4 [Node 5 [Node 6 [], Node 7[]], Node 8 []]
--       print $ fmap (+1) . fmap (+2) $ tree == Node 4 [Node 5 [Node 6 [], Node 7 []], Node 8 []]
main = undefined
