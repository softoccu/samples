#pragma once
#include <string>
#include "parser.h"




class decoder{
private:
   string msg_name;
   parser * parser_ptr;
   string buffer;
   int get_field_number(uint64_t tag);
   int get_wire_type(uint64_t tag);
   string get_next_string_sized(const string & buff, int & i, int size);
   string get_next_string_varint(const string & buff, int & i);
   string get_next_string_fixed64(const string & buff, int & i);
   string get_next_string_fixed32(const string & buff, int & i);
   uint64_t get_next_varint(const string & buff, int & i);
   uint64_t get_next_tag(const string & buff, int & i);
   bool decode_msg(msg_po  *const pmsg, const string & msg_buff );
   bool decode_field(field_po  *const pfield, const string & field_buff);
   string print_nice_read();
   string print_to_json();

   void set_buffer_str(const string & proto_hex_buffer){
      buffer.resize(proto_hex_buffer.length()/2);
      for(int i = 0; i < buffer.length(); ++i){
         char hex0 = proto_hex_buffer[i*2];
         char hex1 = proto_hex_buffer[i*2 + 1];
         int num0 = hex0 >= 'a' ? 10 + hex0 - 'a' : hex0 - '0';
         int num1 = hex1 >= 'a' ? 10 + hex1 - 'a' : hex1 - '0';
         buffer[i] = (num0 << 4) + num1;
      }
   }

/*
    // basic type
    bool parse_double(field_po * const pfield,  const string & str);
    bool parse_float(msg_po * const parent_msg, const string & str, int & i);
    bool parse_int32(msg_po * const parent_msg, const string & str, int & i);
    bool parse_int64(const msg_po *& const parent_msg, const string & str, int & i);
    bool parse_uint32(const msg_po *& parent_msg, const string & str, int & i);
    bool parse_uint64(const msg_po *& parent_msg, const string & str, int & i);
    bool parse_sint32(const msg_po *& parent_msg, const string & str, int & i);
    bool parse_sint64(const msg_po *& parent_msg, const string & str, int & i);
    bool parse_fixed32(const msg_po *& parent_msg, const string & str, int & i);
    bool parse_fixed64(const msg_po *& parent_msg, const string & str, int & i);
    bool parse_sfixed32(const msg_po *& parent_msg, const string & str, int & i);
    bool parse_sfixed64(const msg_po *& parent_msg, const string & str, int & i);
    bool parse_bool(const msg_po *& parent_msg, const string & str, int & i);
    bool parse_string(const msg_po *& parent_msg, const string & str, int & i);
    bool parse_bytes(const msg_po *& parent_msg, const string & str, int & i);
*/
   
public:
   decoder(parser *  const parser_pointer){
      parser_ptr = parser_pointer;
   }
   decoder(parser *  const parser_pointer, const string & message_name, const string & input_hex_string){
      parser_ptr = parser_pointer;
      msg_name = message_name;
      set_buffer_str(input_hex_string);
   }

   ~decoder(){
   }

   bool decode();
   bool decode_print(enum_print_type dt = enum_print_type::e_nice_read);
   void print_buffer(){
      cout << "print buffer as hex string: buffer length = " << buffer.length()  <<  endl;
      for(auto c : buffer){
         printf("%02x", (unsigned char)c);
      }
      cout << endl;
      cout << "--------" << endl;
   }
   void print_hex_buffer(const string & buff){
      for(auto c : buff){
         printf("%02x", (unsigned char)c);
      }
      cout << endl;
   }

};
