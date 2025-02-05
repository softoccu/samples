#include <iostream>
#include <cstdio>
#include <fstream>
#include <vector>
#include "parser.h"
#include "decode.h"
using namespace std;

int main(int argc, char * argv[]){
   string usage = "Usage: protodec ProtoPath rootmsg bufferHexString \n eg: protodec mypath/abc.proto frame 1234567890abcdef";
   if(argc < 4) { 
     cout << usage << endl;
     return 0;
   }
   string proto_file = argv[1];
   string msg_name = argv[2];
   string buff_hex_string = argv[3];
   cout << "proto file to parse: " << proto_file << endl;
   cout << "message to decode " <<  msg_name << endl;
   cout << "buffer to decode, length =   " << buff_hex_string.length()  << ": " << buff_hex_string << endl;
   // proto file set to 2K for now
   char buf[2048];
   FILE* f = std::fopen(proto_file.c_str(), "r");
   int bufsize = fread(&buf[0], sizeof(buf[0]), sizeof(buf)/sizeof(buf[0]), f);
   string proto_str (buf, bufsize);
   parser & parser_obj = *parser::get_parser_ptr();
   parser_obj.set_proto_str(proto_str);
   parser_obj.parse_proto();
   parser_obj.print_all();
   decoder dec(&parser_obj, msg_name, buff_hex_string) ;
   dec.print_buffer();
   dec.decode();
   delete &parser_obj;
}
