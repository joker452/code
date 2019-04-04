{-# LANGUAGE CPP #-}
{-# LANGUAGE NoRebindableSyntax #-}
{-# OPTIONS_GHC -fno-warn-missing-import-lists #-}
module Paths_hw5 (
    version,
    getBinDir, getLibDir, getDynLibDir, getDataDir, getLibexecDir,
    getDataFileName, getSysconfDir
  ) where

import qualified Control.Exception as Exception
import Data.Version (Version(..))
import System.Environment (getEnv)
import Prelude

#if defined(VERSION_base)

#if MIN_VERSION_base(4,0,0)
catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
#else
catchIO :: IO a -> (Exception.Exception -> IO a) -> IO a
#endif

#else
catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
#endif
catchIO = Exception.catch

version :: Version
version = Version [0,1,0,0] []
bindir, libdir, dynlibdir, datadir, libexecdir, sysconfdir :: FilePath

bindir     = "C:\\Users\\Deng\\AppData\\Roaming\\cabal\\bin"
libdir     = "C:\\Users\\Deng\\AppData\\Roaming\\cabal\\x86_64-windows-ghc-8.6.3\\hw5-0.1.0.0-GRtenu8KMY5FaWknbTCPMs-hw5-test"
dynlibdir  = "C:\\Users\\Deng\\AppData\\Roaming\\cabal\\x86_64-windows-ghc-8.6.3"
datadir    = "C:\\Users\\Deng\\AppData\\Roaming\\cabal\\x86_64-windows-ghc-8.6.3\\hw5-0.1.0.0"
libexecdir = "C:\\Users\\Deng\\AppData\\Roaming\\cabal\\hw5-0.1.0.0-GRtenu8KMY5FaWknbTCPMs-hw5-test\\x86_64-windows-ghc-8.6.3\\hw5-0.1.0.0"
sysconfdir = "C:\\Users\\Deng\\AppData\\Roaming\\cabal\\etc"

getBinDir, getLibDir, getDynLibDir, getDataDir, getLibexecDir, getSysconfDir :: IO FilePath
getBinDir = catchIO (getEnv "hw5_bindir") (\_ -> return bindir)
getLibDir = catchIO (getEnv "hw5_libdir") (\_ -> return libdir)
getDynLibDir = catchIO (getEnv "hw5_dynlibdir") (\_ -> return dynlibdir)
getDataDir = catchIO (getEnv "hw5_datadir") (\_ -> return datadir)
getLibexecDir = catchIO (getEnv "hw5_libexecdir") (\_ -> return libexecdir)
getSysconfDir = catchIO (getEnv "hw5_sysconfdir") (\_ -> return sysconfdir)

getDataFileName :: FilePath -> IO FilePath
getDataFileName name = do
  dir <- getDataDir
  return (dir ++ "\\" ++ name)
