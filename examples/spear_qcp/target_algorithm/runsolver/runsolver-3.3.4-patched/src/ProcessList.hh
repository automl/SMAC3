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



#ifndef _ProcessList_hh_
#define _ProcessList_hh_

#include <set>

using namespace std;

/**
 * a class to store a list of process IDs
 *
 * ??? should be optimized !!!
 */
class ProcessList
{
private:
  set<pid_t> s;
public:
  typedef set<pid_t>::iterator iterator;

  inline void add(pid_t pid)
  {
    s.insert(pid);
  }

  inline void remove(pid_t pid)
  {
    s.erase(pid);
  }

  inline bool contains(pid_t pid) const
  {
    return s.find(pid)!=s.end();
  }

  iterator begin() const {return s.begin();}
  iterator end() const {return s.end();}
};


#endif

// Local Variables:
// mode: C++
// End:

