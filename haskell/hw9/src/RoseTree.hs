module RoseTree where

import RoseTreeType

mapRoseTree :: (a -> b) -> RoseTree a -> RoseTree b
mapRoseTree f (Node a nodes) = case nodes of
                                            [] -> Node (f a) []
                                            xs -> Node (f a) (map (mapRoseTree f) xs)

instance Functor RoseTree where fmap = mapRoseTree