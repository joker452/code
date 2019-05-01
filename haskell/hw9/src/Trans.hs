module Trans where

import MyLens
import TriangleType
import TriangleTrans ( coordXLens
                     , coordYLens
                     , triALens
                     , triBLens
                     , triCLens )

transPoint :: (Integer, Integer) -> Point -> Point
transPoint (deltaX, deltaY) =
  (coordXLens %~ (+ deltaX)) . (coordYLens %~ (+ deltaY))

transTriangle :: (Integer, Integer) -> Triangle -> Triangle
transTriangle delta =
  (triALens %~ transPoint delta) .
  (triBLens %~ transPoint delta) .
  (triCLens %~ transPoint delta)
