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



#include <fstream>
#include <string>
#include <map>

#include <asm/unistd.h>

using namespace std;

map<int,string> list;

int main()
{
#include "tmpSyscallList.cc"

  int max=0;
  map<int,string>::iterator it;

  for(it=list.begin();it!=list.end();++it)
    if ((*it).first>max)
      max=(*it).first;

  ofstream Hfile("SyscallNames.hh");

  Hfile << "#ifndef _SyscallNames_hh_" << endl
	<< "#define _SyscallNames_hh_" << endl
	<< endl;

  Hfile << "const int nbSyscallNames=" << max+1 << ";" << endl;

  Hfile << endl;

  Hfile << "const char *getSyscallName(int n);" << endl;

  Hfile << endl;

  Hfile << "#endif" << endl;

  Hfile.close();

  ofstream CCfile("SyscallNames.cc");

  CCfile << "#include \"SyscallNames.hh\"\n\n";

  CCfile << "const char *syscallNames[nbSyscallNames]={\n";

  for(int i=0;i<=max;++i)
  {
    string name;
    it=list.find(i);
    if (it==list.end())
      name="???";
    else
      name=(*it).second;

    CCfile << "\t\"" << name << "\"";
    if (i!=max)
      CCfile << ",";

    CCfile << "\n";
  }

  CCfile << "};\n\n";

  CCfile << "const char *getSyscallName(int n)\n"
	 << "{\n"
	 << "  if (n>0 && n<=nbSyscallNames)\n"
	 << "    return syscallNames[n];\n"
	 << "  else\n"
	 << "    return \"???\";\n"
	 << "}\n";

  CCfile.close();
}
