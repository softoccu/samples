#pragma once
#include <iostream>
#include <string>
#include <cstdint>

const int bit_offset_required = 0;
const int bit_offset_repeated = 1;
const int bit_offset_submessage = 2;
const int bit_offset_global_submsg = 3;
const int bit_offset_inside_oneof = 4;
const int bit_offset_enum_type = 5;
const int bit_offset_reserved = 6;
const int init_mask = 0;

const uint64_t uint64_bit0_2 = 7;
const unsigned char byte_bit0_6 = 127;
const unsigned char byte_bit7 = 128;
const int varint_payload_bits = 7;
const int wire_type_bits = 3;

const int wire_type_varint = 0;
const int wire_type_fix64 = 1;
const int wire_type_delimited = 2;
const int wire_type_fix32 = 5;


    // bit 0    1 required / 0 optional  
    // bit 1    1 repeated  / 0 not repeated
    // bit 2    1 submessage / non-submessage
    // bit 3    1 global submessage / non-global submessage
    // bit 4    1 inside oneof   / 0 non inside oneof 
    // bit 5    1 enum type    /  non enum type
    // bit 6    1 reserved     / 0  not reserved
    // bit 7 ~ bit31 add later for more complex feature

const int indent_size = 4;

 enum class enum_field_type : int {
   e_invalid_type = -1,
   e_basic_type = 0,
   e_global_submsg = 1,
   e_local_submsg = 2
 };

 enum class enum_print_type : int{
   e_nice_read = 0,
   e_to_json = 1
 };

class mask_operator{
   public:
   static void set_required(bool is_required, uint32_t & mask){
      if(is_required) mask |= (1 << bit_offset_required);
      else mask &= ~(1<<bit_offset_required);
   }
   static bool get_required(const uint32_t & mask){
      return (mask & (1 << bit_offset_required))? true: false;
   }
   static void set_repeated(bool is_repeated, uint32_t & mask){
      if(is_repeated)mask |= (1<<bit_offset_repeated);
      else mask &= ~ (1<<bit_offset_repeated);
   }
   static bool get_repeated(const uint32_t & mask){
      return (mask & (1 << bit_offset_repeated))? true: false;
   }
   static void set_submessage(bool is_submsg, uint32_t & mask){
      if(is_submsg)mask |= (1<<bit_offset_submessage);
      else mask &= ~ (1<<bit_offset_submessage);
   }
   static bool get_submessage(const uint32_t & mask){
      return (mask & (1 << bit_offset_submessage))? true: false;
   }
   static void set_globalsubmsg(bool is_gblsubmsg, uint32_t & mask){
      if(is_gblsubmsg)mask |= (1<<bit_offset_global_submsg);
      else mask &= ~ (1<<bit_offset_global_submsg);
   }
   static bool get_globalsubmsg(const uint32_t & mask){
      return (mask & (1 << bit_offset_global_submsg))? true: false;
   }
   static void set_inside_oneof(bool is_inside_oneof, uint32_t & mask){
      if(is_inside_oneof)mask |= (1<<bit_offset_inside_oneof);
      else mask &= ~ (1<<bit_offset_inside_oneof);
   }
   static bool get_inside_oneof(const uint32_t & mask){
      return (mask & (1 << bit_offset_inside_oneof))? true: false;
   }
   static void set_enum_type(bool is_enum_type, uint32_t & mask){
      if(is_enum_type)mask |= (1<<bit_offset_enum_type);
      else mask &= ~ (1<<bit_offset_enum_type);
   }
   static bool get_enum_type(const uint32_t & mask){
      return (mask & (1 << bit_offset_enum_type))? true: false;
   }
   static void set_reserved(bool is_rederved, uint32_t & mask){
      if(is_rederved)mask |= (1<<bit_offset_reserved);
      else mask &= ~ (1<<bit_offset_reserved);
   }
   static bool get_reserved(const uint32_t & mask){
      return (mask & (1 << bit_offset_reserved))? true: false;
   }

};


const int invalid_field_num = 0;

#define CHECK_POINT(p)  if((p) == nullptr) {\
   printf("nullptr exception at %s, line %d \n", __FILE__, __LINE__);\
   return false;\
   }
#define CHECK_BOOL(b) if((b) == false) {\
   printf("false exception at %s, line %d \n", __FILE__, __LINE__);\
   return false;\
   }

static std::string get_next_token(const std::string & str, int & i){
     while(i < str.length() && !isalpha (str[i]) && str[i] != '_') ++i;
     if(i == str.length()) return "";
     int start = i;
     while(isalpha(str[i]) || isdigit(str[i]) || str[i] == '_')++i;
     int end = i;
     return str.substr(start, end - start);
}

static std::string get_next_token_reverse(const std::string & str, int & i){
     while(i >= 0 && !isalpha (str[i]) && !isdigit(str[i])  && str[i] != '_' ) --i;
     if(i < 0) return "";
     int start = i;
     while(isalpha(str[i]) || isdigit(str[i]) || str[i] == '_')--i;
     int end = i;
     return str.substr(end + 1, start - end);
}

static int get_num_from_string(const std::string & str){
   int i = 0;
   while(!isdigit(str[i]))++i;
   int start = i;
   while(isdigit(str[i]))++i;
   int end = i;
   int field_num = std::stoi(str.substr(start, end-start));
   return field_num;
}
