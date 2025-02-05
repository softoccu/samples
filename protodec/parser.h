#pragma once
#include <string>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <variant>
#include <vector>
#include <optional>
#include "common.h"

using namespace std;


class field_po;
class msg_po;
class parser;
class decoder;

// PO means parser object
class field_po {
private:
   string fieldtype;
   string fieldname;
    //The smallest field number you can specify is 1, and the largest is 536,870,911.   at max 29 bit
    // You also cannot use the numbers 19000 through 19999
   int fieldnum;
   uint32_t mask; // all flag save in mask 
   msg_po * pmsg;
   optional<string> oneofname;
   optional<unordered_map<int, field_po*>> oneof_element_map;
   optional<string> enumname;
   optional<unordered_map<int, string>> enum_element_map;

   // used by decode only
   bool has_data;
   string data; // bytes belong to this field
   variant<int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint64_t, string, vector<int8_t>, vector<int16_t>, vector<int32_t>, vector<int64_t>,
      vector<uint8_t>, vector<uint16_t>, vector<uint32_t>, vector<uint64_t>> var_data;
    
   void set_required(bool is_required){
      mask_operator::set_required(is_required, mask);
   }
   bool is_required(){
      return mask_operator::get_required(mask);
   }
   void set_repeated(bool is_repeated){
      mask_operator::set_repeated(is_repeated, mask);
   }
   bool is_repeated(){
      return mask_operator::get_repeated(mask);
   }
   void set_submessage(bool is_submsg){
      mask_operator::set_submessage(is_submsg, mask);
   }
   bool is_submessage(){
      return mask_operator::get_submessage(mask);
   }
   void set_global_submsg(bool is_global_submsg){
      mask_operator::set_globalsubmsg(is_global_submsg, mask);
   }
   bool is_global_submsg(){
      return mask_operator::get_globalsubmsg(mask);
   }
   void set_in_oneof(bool is_in_oneof){
      mask_operator::set_inside_oneof(is_in_oneof, mask);
   }
   bool is_in_oneof(){
      return mask_operator::get_inside_oneof(mask);
   }
   void set_enum_type(bool is_enum){
      mask_operator::set_enum_type(is_enum, mask);
   }
   bool is_enum_type(){
      return mask_operator::get_enum_type(mask);
   }
   void set_reserved(bool is_reserved){
      mask_operator::set_reserved(is_reserved, mask);
   }
   bool is_reserved(){
      return mask_operator::get_reserved(mask);
   }
public:
   field_po(msg_po * msg){
      pmsg = msg;
      mask = init_mask;
      fieldnum = invalid_field_num;
      has_data = false;
   }
    
   friend class msg_po;
   friend class parser;
   friend class decoder;
};

class msg_po{
 private:
    string msg_name;
    unordered_map<string, int> name_num_map;
    map<int, string> num_name_map;// oneof and enum define are not in map 
    unordered_map<string, field_po*> name_map;
    map<int, field_po*> num_map; // oneof and enum define are not in map 
    // local visable
    unordered_set<string> inner_defined_msg_set;
    unordered_set<string> inner_defined_enum_set;
    unordered_map<string, msg_po*> inner_defined_msg_map;
 public:
    string get_msg_name(){ 
        return msg_name;
    }
    msg_po(){
    }
    ~msg_po(){
        for(auto & p : inner_defined_msg_map){
            delete p.second;
            p.second = nullptr;
        }
        for(auto & p : name_map){
            delete p.second;
            p.second = nullptr;
        }
    }

    friend class field_po;
    friend class parser;
    friend class decoder;
};


class parser {
private:
    
    unordered_set<string> basic_type_set;
    unordered_set<string> keyword_set;
    // global visable
    unordered_set<string> defined_msg_set;
    unordered_map<string, msg_po*> defined_msg_map; 
    unordered_set<string> defined_enum_set;
    string proto_str;
    int syntax_ver;
    friend class field_po;
    friend class msg_po;
    friend class decoder;

    parser();
    parser(parser&) = delete;
    parser & operator = (const parser &) = delete;
    static parser * m_parser_ptr;

    bool parse_syntax(const string & str, int & i);
    bool parse_message(const string & str, int & i);
    bool parse_message_body( msg_po * const pmsg, const string & str);
    bool parse_reserved( msg_po * const parent_msg, const string & str, int & i);
    bool parse_oneof(msg_po * const parent_msg, const string & str, int & i);
    bool parse_oneof_body(field_po * const parent_field, const string & body_str);
    bool parse_oneof_field(field_po * const parent_field, const string & line_str);

    bool parse_basic_type(msg_po * const pmsg, const string & field_type, const string & body_str, int & i, uint32_t  mask);
    bool parse_sub_message_field(msg_po * const parent_msg, const string & field_type, const string & body_str, int & i, uint32_t mask);
    bool parse_sub_message(msg_po * const parent_msg, const string & str, int & i);
    void print_msg(const msg_po * pmsg, int indent = 0);

    enum_field_type check_type(const msg_po * const pmsg, const string & token);

    
 public:
    void set_proto_str(const string & input_proto_str){
        proto_str = input_proto_str;
    }
    bool parse_proto();
    ~parser(){
        for(auto & p : defined_msg_map){
            delete p.second;
        }
    }
    static parser * get_parser_ptr(){
        if(m_parser_ptr == nullptr){
            m_parser_ptr = new parser;
        }
        return m_parser_ptr;
    }
    void clear_defined_type(){
        defined_msg_set.clear();
        for(auto & p : defined_msg_map){
            delete p.second;
        }
        defined_msg_map.clear(); 
    }
    void print_all();
    void print_msg(const string & msg_name, int indent = 0 );
};


