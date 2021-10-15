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



#ifndef _CircularBufferFilter_hh_
#define _CircularBufferFilter_hh_

#include <iostream>
#include <cstring>

using namespace std;


class AbstractFilter
{
public:
  virtual void write(const char *buffer, int len)=0;

  /**
   * handle partial writes and EINTR
   */
  void systemWrite(int fd, const char *buf, size_t len)
  {
    const char *p=buf;
    int n;
    
    do
    {
      n=::write(fd,p,len);
      if(n<0)
      {
	if(errno==EINTR)
	  continue;
	
	perror("write failed: ");
	break;
      }
      
      len-=n;
      p+=n;
    }
    while(len);
  }
};

class NullFilter : public AbstractFilter
{
private:
  int fd;
public:
  NullFilter(int fd): fd(fd) {}

  virtual void write(const char *buffer, int len)
  {
    systemWrite(fd,buffer,len);
  }
};

/**
 * This filter enforces a limit on an output file size. When the file
 * exceeds the limit, only the first and last bytes of output are
 * saved in the file.
 *
 * This filter transmits directly to file descriptor fd the first
 * activateSize bytes and then it only transmits the last
 * maxSize-activateSize bytes.
 *
 * TODO ???
 *
 * - we currently use a circular buffer in memory to do the job. We
     may instead work directly on the file, using lseek to rewind to
     position activateSize as soon as the file size exceeds
     maxSize. At the end, we have to reorder the bytes at the end of
     the file. This may save memory but requires to be able to lseek
     on the file.
 *
 * - the last part of the file to which we output doesn't necessarily
     start at the beginning of a line. This may cause problem in some
     applications.
 */
class CircularBufferFilter : public AbstractFilter
{
private:
  unsigned long long int total; // total number of bytes sent to this filter
  unsigned long long int activateSize,maxSize,bufferSize;

  char *data; // circular buffer
  unsigned int w; // position where to write in the circular buffer

  int fd; // file descriptor to write to

public:
  CircularBufferFilter()
  {
    data=NULL;
  }

  CircularBufferFilter(int fd, 
		       unsigned long long int activateSize, 
		       unsigned long long int maxSize)
  {
    data=NULL;
    setup(fd,activateSize,maxSize);
  }

  void setup(int fd, 
	     unsigned long long int activateSize, 
	     unsigned long long int maxSize)
  {
    this->fd=fd;
    this->activateSize=activateSize;
    this->maxSize=maxSize;
    bufferSize=maxSize-activateSize;
    total=0;
  }

  ~CircularBufferFilter()
  {
    flush();
  }

  virtual void write(const char *buffer, int len)
  {
    total+=len;

    if (total<activateSize)
      systemWrite(fd,buffer,len);
    else
    {
      unsigned int n,r=0;

      if (!data)
      {
	data=new char[bufferSize];
	w=0;
      }

      if (total-len<activateSize)
      {
	// the first activateSize bytes must be written directly

	n=activateSize-total+len;
	systemWrite(fd,buffer,n);
	len-=n;
	buffer+=n;
      }

      do
      {
	n=len;
	if (n>bufferSize-w)
	  n=bufferSize-w;

	memcpy(data+w,buffer+r,n);
	len-=n;
	r+=n;
	w+=n;
	if (w>=bufferSize)
	  w=0;	
      } while(len>0);
    }
  }

  /**
   * normally, this should only be called by the destructor.
   *
   * remember that the destructor is not called if we are an auto
   * object (local variable) and if we call exit()
   */
  void flush()
  {
    if (!data)
      return;

    char msg[512];

    if (total<=maxSize)
      systemWrite(fd,data,total-activateSize);
    else
    {
      snprintf(msg,sizeof(msg),
	       "\n"
	       "###########################################################\n"
	       "# A total of %llu bytes were output by the program.\n"
	       "# This exceeds the hard limit that is enforced.\n"
	       "# Only the %llu first bytes are saved in this file before\n"
	       "# this point and only the %llu last bytes are saved after\n"
	       "# this point. A total of %llu bytes are lost.\n"
	       "###########################################################\n",
	       total,activateSize,bufferSize,total-maxSize);
      systemWrite(fd,msg,strlen(msg));
      systemWrite(fd,data+w,bufferSize-w);
      systemWrite(fd,data,w);
    }
  }
};

// Local Variables:
// mode: C++
// End:
#endif
