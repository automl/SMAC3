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



#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>
#include <sstream>
#include <cctype>
#include <cerrno>

using namespace std;

int main(int argc, char **argv)
{
  string s,timestamp;
  bool withTimestamp;

  withTimestamp=argc==2 && strcmp(argv[1],"--timestamp")==0;

  try
  {
    while(cin.good())
    {
      if (withTimestamp)
      {
	cin >> timestamp;
	cin.get(); // skip tabulation

	if (!cin.good())
	  break;
      }

      getline(cin,s);

      if (!cin.good() && s.length()==0)
	break;

      // we're only concerned by a "v " line
      if (s.length()>=2 && s[0]=='v' && s[1]==' ')
      {
	istringstream f(s);
	string word;
	int len=0;

	f >> word; // skip "v "

	while (f.good())
	{
	  // read a literal
	  word="";
	  f >> word;
	  if (word=="")
	    break;

	  if (len==0)
	  {
	    if (withTimestamp)
	      cout << timestamp << "\t";
	    cout << "v";
	  }

	  cout << " " << word;
	  len+=word.length();

	  if (len>100)
	  {
	    cout << endl;
	    len=0;
	  }
	}

	if (len!=0)
	  cout << endl;

	s.clear();
      }
      else
      {
	// copy
	if (withTimestamp)
	  cout << timestamp << "\t";
	cout << s << endl;
      } 
    }
  }
  catch (exception &e)
  {
    cout.flush();
    cerr << "\n\tUnexpected exception :\n";
    cerr << "\t" << e.what() << endl;
    exit(1);
  }
}

