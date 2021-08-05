/*
 * Copyright (C) 2010 Olivier ROUSSEL
 *
 * This file is part of runsolver.
 *
 * runsolver is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * runsolver is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with runsolver.  If not, see <http://www.gnu.org/licenses/>.
 */



#ifndef _Observer_hh_
#define _Observer_hh_
#endif

/**
 * a class to watch file descriptors and wait for data to become
 * available on one of them
 *
 */
class Observer
{
private:
  vector<int> desc;
  fd_set readable;
public:
  /**
   * add a file descriptor to watch
   */
  void add(int fd)
  {
    desc.push_back(fd);
  }

  /**
   * remove a file descriptor from the list of file descriptors to watch
   *
   * doit être appele avant de fermer la socket
   */
  void remove(int fd)
  {
    vector<int>::iterator i=find(desc.begin(),desc.end(),fd);
    if (i!=desc.end())
      desc.erase(i);
  }

  /**
   * tell if there is some file descriptor left to watch
   */
  bool empty()
  {
    return desc.empty();
  }

  /**
   * wait for data to become available on one of the file descriptor
   * that is watched
   *
   * this is a blocking method
   */
  void waitForData()
  {
    int max=0;

    FD_ZERO(&readable);
    
    for (int i=0;i<desc.size();++i)
    {
      FD_SET(desc[i],&readable);
      if (max<=desc[i])
	max=desc[i]+1;
    }

    int result=select(max,&readable,NULL,NULL,NULL);
    if (result<0)
    {
      throw runtime_error("error select"+strerror(errno));
    }
  }

  /**
   * may only be called after a call to waitForData
   *
   * returns true iff descriptor fd has data available
   */
  bool hasData(int fd)
  {
    if (find(desc.begin(),desc.end(),fd)==desc.end())
      return false;

    return FD_ISSET(fd,&readable);
  }
};

#endif
