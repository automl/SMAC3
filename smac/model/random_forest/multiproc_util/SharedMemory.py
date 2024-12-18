#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""https://github.com/python/cpython/issues/82300#issuecomment-2169035092
"""

from typing import Optional
import sys
import threading
from multiprocessing import resource_tracker as _mprt
from multiprocessing import shared_memory as _mpshm


if sys.version_info >= (3, 13):
    SharedMemory = _mpshm.SharedMemory
else:
    class SharedMemory(_mpshm.SharedMemory):
        __lock = threading.Lock()

        def __init__(
            self, name: Optional[str] = None, create: bool = False,
            size: int = 0, *, track: bool = True
        ) -> None:
            self._track = track

            # if tracking, normal init will suffice
            if track:
                super().__init__(name=name, create=create, size=size)
                return

            # lock so that other threads don't attempt to use the
            # register function during this time
            with self.__lock:
                # temporarily disable registration during initialization
                orig_register = _mprt.register
                _mprt.register = self.__tmp_register

                # initialize; ensure original register function is
                # re-instated
                try:
                    super().__init__(name=name, create=create, size=size)
                finally:
                    _mprt.register = orig_register

        @staticmethod
        def __tmp_register(*args, **kwargs) -> None:
            return

        def unlink(self) -> None:
            if _mpshm._USE_POSIX and self._name:
                _mpshm._posixshmem.shm_unlink(self._name)
                if self._track:
                    _mprt.unregister(self._name, "shared_memory")