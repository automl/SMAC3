#ifndef _Cores_hh_
#define _Cores_hh_

#include <unistd.h>
#include <vector>

using namespace std;

/**
 * get the list of cores available on the system
 *
 * if physicalView is true, "list" will contain the id of existing
 * cores in increasing order (from 0 to n-1).
 *
 * if physicalView is false, "list" will contain the id of existing
 * cores ordered by the processor to which they belong. For example,
 * list could contain 0,2,4,6,1,3,5,7 if the first processor contains
 * cores 0,2,4,6 and the second processor contains cores 1,3,5,7.
 */
void getExistingCores(vector<unsigned short int> &list, bool physicalView)
{
  char fname[128];
  string buffer;
  ifstream f; 
  cpu_set_t cores;

  CPU_ZERO(&cores);

  list.clear();

  for(unsigned int cpu=0;cpu<sizeof(cpu_set_t)<<3;++cpu)
    if(!CPU_ISSET(cpu,&cores)) // if we didn't already found that core
    {
      snprintf(fname,sizeof(fname),
	       "/sys/devices/system/cpu/cpu%d/topology/core_siblings",cpu);

      f.open(fname);
      if(!f.good())
	return;

      if(physicalView)
      {
	list.push_back(cpu);
	f.close();
	continue;
      }

      // logical view
      getline(f,buffer);
      f.close();

      int len=buffer.length()-1;
      while(isspace(buffer[len]) && len>0)
	--len;

      int id=0,mask;
      for(int i=len;i>=0;--i,id+=4)
	if(buffer[i]!='0' && buffer[i]!=',')
	{
	  if(buffer[i]>='0' && buffer[i]<='9')
	    mask=buffer[i]-'0';
	  else
	    if(buffer[i]>='a' && buffer[i]<='f')
	      mask=10+buffer[i]-'a';
	    else
	      if(buffer[i]>='A' && buffer[i]<='F')
		mask=10+buffer[i]-'A';
	      else
		throw runtime_error("invalid character in cpu mask");

	  for(int j=0;j<4;++j)
	  {
	    if((mask & 1) && !CPU_ISSET(id+j,&cores))
	    {
	      list.push_back(id+j);
	      CPU_SET(id+j,&cores); // don't count it twice!
	    }

	    mask>>=1;
	  }
	}
    } // if(CPU_ISET(...))
}



/**
 * return the list of cores allocated to this process, ordered by
 * physical cpu
 */
void getAllocatedCoresByProcessorOrder(vector<unsigned short int> &allocatedCores)
{
  char fname[128];
  string buffer;
  ifstream f; 
  cpu_set_t affinityMask;

  allocatedCores.clear();

  sched_getaffinity(0,sizeof(cpu_set_t),&affinityMask);

  for(unsigned int cpu=0;cpu<sizeof(cpu_set_t)<<3;++cpu)
    if(CPU_ISSET(cpu,&affinityMask))
    {
      snprintf(fname,sizeof(fname),
	       "/sys/devices/system/cpu/cpu%d/topology/core_siblings",cpu);

      f.open(fname);
      if(!f.good())
	return;

      getline(f,buffer);

      f.close();

      int len=buffer.length()-1;
      while(isspace(buffer[len]) && len>0)
	--len;

      int id=0,mask;
      for(int i=len;i>=0;--i,id+=4)
	if(buffer[i]!='0' && buffer[i]!=',')
	{
	  if(buffer[i]>='0' && buffer[i]<='9')
	    mask=buffer[i]-'0';
	  else
	    if(buffer[i]>='a' && buffer[i]<='f')
	      mask=10+buffer[i]-'a';
	    else
	      if(buffer[i]>='A' && buffer[i]<='F')
		mask=10+buffer[i]-'A';
	      else
		throw runtime_error("invalid character in cpu mask");

	  for(int j=0;j<4;++j)
	  {
	    if((mask & 1) && CPU_ISSET(id+j,&affinityMask))
	    {
	      allocatedCores.push_back(id+j);
	      CPU_CLR(id+j,&affinityMask); // don't count it twice!
	    }

	    mask>>=1;
	  }
	}
    } // if(CPU_ISET(...))
}

/**
 * get the list of cores allocated to this process
 */
void getAllocatedCores(vector<unsigned short int> &list, pid_t pid=0)
{
  cpu_set_t mask;
  list.clear();
  
  sched_getaffinity(pid,sizeof(cpu_set_t),&mask);

  for(unsigned int i=0;i<sizeof(cpu_set_t)<<3;++i)
    if(CPU_ISSET(i,&mask))
      list.push_back(i);
}

/**
 * print the list of cores allocated to this process
 * (getAllocatedCores must be called first).
 */
void printAllocatedCores(ostream &s, const vector<unsigned short int> &list)
{
  size_t end;
  
  for(size_t beg=0;beg<list.size();beg=end)
  {
    end=beg+1;
    while(end<list.size() && list[end]==list[end-1]+1)
      ++end;

    if(beg!=0)
      s << ',';

    if(end==beg+1)
      s << list[beg];
    else
      s << list[beg] << '-' << list[end-1];
  }
}

/**
 * generate a cpu_set mask from a list of cores
 */
cpu_set_t affinityMask(const vector<unsigned short int> &cores)
{
  cpu_set_t mask;
  
  CPU_ZERO(&mask);
  for(size_t i=0;i<cores.size();++i)
    CPU_SET(cores[i],&mask);
  
  return mask;
}


#endif
