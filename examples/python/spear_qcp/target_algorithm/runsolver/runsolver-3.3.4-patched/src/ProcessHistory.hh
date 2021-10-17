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



#ifndef _ProcessHistory_hh_
#define _ProcessHistory_hh_

#include <iostream>
#include "ProcessTree.hh"

using namespace std;

/**
 * maintains a history of process trees
 */
class ProcessHistory
{
private:
  int nbcell;
  vector<int> nbitem;
  vector<ProcessTree *> history;

public:
  ProcessHistory(int n)
  {
    nbcell=n;
    nbitem.resize(nbcell);
    history.resize(2*nbcell);
    for(int i=0;i<nbcell;++i)
      nbitem[i]=0;
  }

  ~ProcessHistory()
  {
    // delete every process tree we have
    for(int cell=0;cell<nbcell && nbitem[cell]>0;++cell)
      for(int i=0;i<nbitem[cell];++i)
	delete history[2*cell+i];
  }

  void push(ProcessTree *elem)
  {
    int cell=0;
    ProcessTree * tmp;
    bool move;

    do
    {
      if (nbitem[cell]<2)
      {
	history[2*cell+nbitem[cell]++]=elem;
	move=false;
      }
      else
      {
	nbitem[cell]=0;

	tmp=history[2*cell];
	drop(history[2*cell+1]);

	history[2*cell+nbitem[cell]++]=elem;
	elem=tmp;
	
	move=true;
	++cell;
      }
    }
    while(move && cell<nbcell);

    if (move)
      drop(elem);

#ifdef debug
    for(int cell=0;cell<nbcell && nbitem[cell]>0;++cell)
    {
      for(int i=0;i<nbitem[cell];++i)
	cout << history[2*cell+i] << ' ';
      cout << '|';
    }
    cout << endl;
#endif
  }

  void dumpHistory(ostream &s, float elapsedLimit)
  {
    for(int cell=nbcell-1;cell>=0;--cell)
    {
      if (nbitem[cell]==0 ||
	  history[2*cell]->getElapsedTime()<=elapsedLimit)
	continue;

      history[2*cell]->dumpProcessTree(s);
      history[2*cell]->dumpCPUTimeAndVSize(s);
    }

    if (nbitem[0]==2)
    {
      // also dump the most recent
      history[1]->dumpProcessTree(s);
      history[1]->dumpCPUTimeAndVSize(s);
    }
  }

protected:
  void drop(ProcessTree *elem)
  {
    delete elem;
  }
};

// Local Variables:
// mode: C++
// End:
#endif
