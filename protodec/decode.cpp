#include "decode.h"

int decoder::get_wire_type(uint64_t tag){
   int wire_type = tag & uint64_bit0_2;
   return wire_type;
}

int decoder::get_field_number(uint64_t tag){
   tag >>= wire_type_bits;
   int field_num = tag;
   return field_num;
}

string decoder::get_next_string_sized(const string & buff, int & i, int size){
   int start = i;
   i += size;
   return i >=buff.length() ? buff.substr(start) : buff.substr(start, i - start);
}

string decoder::get_next_string_varint(const string & buff, int & i){
   int start = i;
   while(buff[i] & byte_bit7)++i;
   ++i;
   return buff.substr(start, i -start);
}

string decoder::get_next_string_fixed64(const string & buff, int & i){
   int start = i;
   i+=sizeof(uint64_t);
   return buff.substr(start, i - start);
}

string decoder::get_next_string_fixed32(const string & buff, int & i){
   int start = i;
   i+=sizeof(uint32_t);
   return buff.substr(start, i - start);
}

uint64_t decoder::get_next_varint(const string & buff, int & i){
   int start = i;
   uint64_t vint = 0;
   do {
      uint8_t part = buff[i] & byte_bit0_6;
      vint <<= varint_payload_bits;
      vint += part;
   }while(buff[i++] & byte_bit7);
   return vint;
}

uint64_t decoder::get_next_tag(const string & buff, int & i){
   return get_next_varint(buff, i);
}

bool decoder::decode(){
   auto it = parser_ptr->defined_msg_map.find(msg_name);
   if(it == parser_ptr->defined_msg_map.end()) {
      cout << "decode() can't find message name " << msg_name << endl;
      return false;
   } else {
      msg_po * pmsg = it->second;
      bool ret = decode_msg(pmsg, buffer);
      CHECK_BOOL(ret);
      return ret;
   }
}

bool decoder::decode_msg(msg_po  *const pmsg, const string & msg_buff ){
      bool ret = false;
      int i = 0;
      cout << pmsg->msg_name << " message in decoding " << endl;
      print_hex_buffer(msg_buff);
      while(i < msg_buff.length()){
         uint64_t tag = get_next_tag(msg_buff, i);
         int wire_type = get_wire_type(tag);
         int filed_num = get_field_number(tag);
         auto it = pmsg->num_map.find(filed_num);
         if(it == pmsg->num_map.end()){
            cout << "unknown filed num  " << filed_num << " in message " << pmsg->get_msg_name() << endl;
         } else {
            field_po * pfield = it->second;
            // wire type 3 and 4 are not in use now
            if (wire_type == wire_type_varint){
                  // varint
                  string str_vint = get_next_string_varint(msg_buff, i);
                  ret = decode_field(pfield, str_vint);
                  CHECK_BOOL(ret);
            } else if (wire_type == wire_type_fix64) {
                  // fixed64bit
                  string str_fixed64 = get_next_string_fixed64(msg_buff, i);
                  ret = decode_field(pfield, str_fixed64);
                  CHECK_BOOL(ret);
            } else if (wire_type == wire_type_delimited) {
                  // to fetch size
                  int size = get_next_varint(msg_buff, i);
                  string sized_buffer = get_next_string_sized(msg_buff, i, size);
                  print_hex_buffer(sized_buffer);
                  cout << "size = " << size  << endl;
                  ret = decode_field(pfield, sized_buffer);
                  CHECK_BOOL(ret);
            } else if (wire_type == wire_type_fix32) {
                  // fixed32bit
                  string str_fixed32 = get_next_string_fixed32(msg_buff, i);
                  ret = decode_field(pfield, str_fixed32);
                  CHECK_BOOL(ret);
            }
         }
      }
      return true;
}

bool decoder::decode_field(field_po  *const pfield, const string & field_buff){
   cout << pfield->fieldtype << " " << pfield->fieldname << " field in decoding " << endl;
   print_hex_buffer(field_buff) ;
   pfield->has_data = true;
   pfield->data = field_buff;
   if(pfield->is_submessage()){
      if(pfield->is_global_submsg()){
         auto it = parser_ptr->defined_msg_map.find(pfield->fieldtype);
         if(it == parser_ptr->defined_msg_map.end()){
            cout << "unknown field type : " << pfield->fieldtype << endl;
            return false;
         } else {
            msg_po * pmsg = it->second;
            bool ret = decode_msg(pmsg, field_buff);
            CHECK_BOOL(ret);
         }
      } else {
         auto it = pfield->pmsg->inner_defined_msg_map.find(pfield->fieldtype);
         if(it == pfield->pmsg->inner_defined_msg_map.end()){
            cout << "unknown field type " << pfield->fieldtype;
            return false;
         } else {
            msg_po * pmsg = it->second;
            bool ret = decode_msg(pmsg, field_buff);
            CHECK_BOOL(ret);
         }
      }
   } else {
      if(pfield->fieldtype == "string") cout <<  pfield->fieldname <<  " : " << field_buff << endl;
      cout <<  pfield->fieldname <<  "as hex :" ;
      for(auto c : field_buff){
         printf("%02x", (unsigned char)c);
      }
      cout << endl;
      //// decode to real type
      if(pfield->is_repeated()){
      } else {  
      }
   }
   return true;
}

bool decoder::decode_print(enum_print_type dt){
   if(dt == enum_print_type::e_nice_read){
      string nice_str = print_nice_read();
      cout << nice_str << endl;
   } else if(dt == enum_print_type::e_to_json){
      string json_str = print_to_json();
      cout << json_str << endl;
   }
}

/*
 // basic type
bool decode::decode_double(const string & str, double & val){
   return true;
}
bool decoder::decode_float(const string & str, float & val){
   return true;
}
bool decoder::decode_int32(const string & str, int){
   return true;
}
bool decoder::decode_int64(const string & str, int64_t & val){
   return true;
}
bool decoder::decode_uint32(const string & str, uint32_t & val){
   return true;
}
bool decoder::decode_uint64(const string & str, uint64_t & val){
   return true;
}
 bool decoder::decode_sint32(const string & str, int & val){}
 bool decoder::decode_sint64(const string & str, int64_t & val){}
 bool decoder::decode_fixed32( const string & str, int & i, uint32_t mask){}
 bool decoder::decode_fixed64(const field_po * pfield, const string & str, int & i, uint32_t mask){}
 bool decoder::decode_sfixed32(const field_po * pfield, const string & str, int & i, uint32_t mask){}
 bool decoder::decode_sfixed64(const field_po * pfield, const string & str, int & i, uint32_t mask){}
 bool decoder::decode_bool(const field_po * pfield, const string & str, int & i, uint32_t mask){}
 bool decoder::decode_string(const field_po * pfield, const string & str, int & i, uint32_t mask){}
 bool decoder::decode_bytes(const field_po * pfield, const string & str, int & i, uint32_t mask){}
 
 bool decoder::decode_double(const field_po * pfield, const string & str){
 } 
 bool decoder::decode_float(const field_po * pfield, const string & str, int & i, uint32_t mask){}
 bool decoder::decode_int32(const field_po * pfield, const string & str, int & i, uint32_t mask){}
 bool decoder::decode_int64(const field_po * pfield, const string & str, int & i, uint32_t mask){}
 bool decoder::parse_uint32(const field_po * pfield, const string & str, int & i, uint32_t mask){}
 bool decoder::parse_uint64(const field_po * pfield, const string & str, int & i, uint32_t mask){}
 bool decoder::parse_sint32(const field_po * pfield, const string & str, int & i, uint32_t mask){}
 bool decoder::parse_sint64(const field_po * pfield, const string & str, int & i, uint32_t mask){}
 bool decoder::parse_fixed32(const field_po * pfield, const string & str, int & i, uint32_t mask){}
 bool decoder::parse_fixed64(const field_po * pfield, const string & str, int & i, uint32_t mask){}
 bool decoder::parse_sfixed32(const field_po * pfield, const string & str, int & i, uint32_t mask){}
 bool decoder::parse_sfixed64(const field_po * pfield, const string & str, int & i, uint32_t mask){}
 bool decoder::parse_bool(const field_po * pfield, const string & str, int & i, uint32_t mask){}
 bool decoder::parse_string(const field_po * pfield, const string & str, int & i, uint32_t mask){}
 bool decoder::parse_bytes(const field_po * pfield, const string & str, int & i, uint32_t mask){}

*/
string decoder::print_nice_read(){
   return "";
}

string decoder::print_to_json(){
   return "";
}

