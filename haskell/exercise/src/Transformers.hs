{-# LANGUAGE InstanceSigs #-}
module Transformers where

newtype MaybeT m a = MaybeT { runMaybeT :: m (Maybe a) }

instance (Functor m) => Functor (MaybeT m) where
    fmap :: (a -> b) -> MaybeT m a -> MaybeT m b
    fmap f (MaybeT ma) = MaybeT $ fmap (fmap f) ma

instance (Applicative m) => Applicative (MaybeT m) where
    pure :: a -> MaybeT m a
    pure a = MaybeT $ pure (pure a)
    (<*>) :: MaybeT m (a -> b) -> MaybeT m a -> MaybeT m b
    (MaybeT mab) <*> (MaybeT ma) = MaybeT $ (<*>) <$> mab <*> ma

instance (Monad m) => Monad (MaybeT m) where
    return :: a -> MaybeT m a
    return = pure
    (>>=) :: MaybeT m a -> (a -> MaybeT m b) -> MaybeT m b
    (MaybeT ma) >>= f = MaybeT $ do
                                  v <- ma
                                  case v of
                                        Nothing -> return Nothing
                                        Just y -> runMaybeT (f y)

newtype EitherT e m a = EitherT { runEitherT :: m (Either e a) }

instance (Functor m) => Functor (EitherT e m) where
    fmap :: (a -> b) -> EitherT e m a -> EitherT e m b
    fmap f (EitherT ema) = EitherT $ (fmap . fmap) f ema

instance (Applicative m) => Applicative (EitherT e m) where
    pure :: a -> EitherT e m a
    pure a = EitherT $ pure (pure a)
    (<*>) :: EitherT e m (a -> b) -> EitherT e m a -> EitherT e m b
    (EitherT emab) <*> (EitherT ema) = EitherT $ (<*>) <$> emab <*> ema

instance (Monad m) => Monad (EitherT e m) where
    return :: a -> EitherT e m a
    return = pure
    (>>=) :: EitherT e m a -> (a -> EitherT e m b) -> EitherT e m b
    (EitherT ema) >>= f = EitherT $ do
                                     ea <- ema
                                     case ea of
                                             Left e -> return $ Left e
                                             Right r -> runEitherT $ f r

swapEither :: Either e a -> Either a e
swapEither ea = case ea of
                        Left e -> Right e
                        Right r -> Left r

swapEitherT :: (Functor m) => EitherT e m a -> EitherT a m e
swapEitherT (EitherT ema) = EitherT $ fmap swapEither ema


newtype ReaderT r m a = ReaderT { runReaderT :: r -> m a }

instance (Functor m) => Functor (ReaderT r m) where
    fmap :: (a -> b) -> ReaderT r m a -> ReaderT r m b
    fmap f (ReaderT rma) = ReaderT $ \r -> fmap f (rma r)

instance (Applicative m) => Applicative (ReaderT r m) where
    pure :: a -> ReaderT r m a
    pure a = ReaderT $ \r -> pure a
    (<*>) :: ReaderT r m (a -> b) -> ReaderT r m a -> ReaderT r m b
    (ReaderT rmab) <*> (ReaderT rma) = ReaderT $ \r -> rmab r <*> rma r

instance (Monad m) => Monad (ReaderT r m) where
    return :: a -> ReaderT r m a
    return = pure
    (>>=) :: ReaderT r m a -> (a -> ReaderT r m b) -> ReaderT r m b
    (ReaderT rma) >>= f = ReaderT $ \r -> do
                                           a <- rma r
                                           (runReaderT . f) a r