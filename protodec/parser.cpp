#include "parser.h"

parser * parser::m_parser_ptr = nullptr;

parser::parser(){
   basic_type_set = {"double", "float", "int32", "int64", "uint32", "uint64", "sint32", "sint64", "fixed32","fixed64", "sfixed32	","sfixed64","bool","string","bytes"};
   keyword_set = {"syntax","message","reserved","oneof","required","optional","repeated"};  // support map in later version
}

bool parser::parse_proto(){
   int i = 0;
   while(i < proto_str.length()){
      string token = get_next_token(proto_str, i);
      auto it_keyword = keyword_set.find(token);
      if(it_keyword == keyword_set.end()){
         if(token != "") {
            cout << "unknown keyword " << token << endl;
            return false;
         }
      } else {
         if(token == "syntax"){
            bool ret = parse_syntax(proto_str, i);
         } else if (token == "message"){
            bool ret = parse_message(proto_str, i);
         } 
      }
   }
   return true;
}

bool parser::parse_syntax(const string & str, int & i){
   int start = i;
   while(str[i] != ';')++i;
   int end = i;
   ++i;
   string lexical = str.substr(start, end - start);
   if(lexical.find("proto3") != string::npos){
      syntax_ver = 3;
   } else if(lexical.find("proto2") != string::npos){
      syntax_ver = 2;
   } else {
      cout << "unknown syntax version" << endl;
      return false;
   }
   return true;
}

bool parser::parse_message(const string & str, int & i){
   string msg_name = get_next_token(str,  i);
   while(str[i] != '{')++i;
   int cnt = 1;
   ++i;
   int start = i;
   int end = 0;
   while(true){
      if(str[i] == '{'){
         ++cnt;
      } else if(str[i] == '}'){
         if(cnt == 1) {
            end = i;
            break;
         } else {
            --cnt;
         }
      }
      ++i;
   }
   msg_po * pmsg = new msg_po;
   pmsg->msg_name = msg_name;
   defined_msg_set.insert(msg_name);
   defined_msg_map[msg_name] = pmsg;
   string msg_body = str.substr(start, end -start);
   
   //cout << msg_body << endl;
   if(!parse_message_body(pmsg, msg_body)){
      cout << "parse message body failed for message " << msg_name << endl;
      return false;
   }
   return true;
}

enum_field_type parser::check_type(const msg_po * const pmsg, const string & token){
      if(basic_type_set.find(token) != basic_type_set.end()){   // basic type
         return enum_field_type::e_basic_type;
      } else if(defined_msg_set.find(token) != defined_msg_set.end()){ // global defined type
         return enum_field_type::e_global_submsg; 
      } else if(pmsg->inner_defined_msg_set.find(token) != pmsg->inner_defined_msg_set.end()){  // local defined type
         return enum_field_type::e_local_submsg;
      } else {
            cout << "unknown keyword " << token << endl;
         return enum_field_type::e_invalid_type;
      }
}

bool parser::parse_message_body( msg_po * const pmsg, const string & body_str){
   int i = 0;
   uint32_t mask = 0;
   while(i < body_str.length()){
      string token = get_next_token(body_str,  i);
      if(token == "")return true;
      auto it_keyword = keyword_set.find(token);
      //  keyword
      if(it_keyword != keyword_set.end()){
         if(token == "reserved"){
            bool ret = parse_reserved(pmsg, body_str, i);
         } else if (token == "message"){
            bool ret = parse_sub_message(pmsg, body_str, i);
         } else if (token == "oneof"){
            bool ret = parse_oneof(pmsg, body_str, i);
         } else if (token == "repeated"){
            mask_operator::set_repeated(true, mask);
         } else if (token == "required"){
            mask_operator::set_required(true, mask);
         }
      } else {
         enum_field_type ft = check_type(pmsg, token);
         if(ft == enum_field_type::e_invalid_type) return false;
         else if(ft == enum_field_type::e_basic_type){   // basic type
            bool ret = parse_basic_type(pmsg, token, body_str, i, mask);
            mask = 0;
         } else if(ft == enum_field_type::e_global_submsg){ // global defined type
            mask_operator::set_submessage(true, mask);
            mask_operator::set_globalsubmsg(true, mask);
            parse_sub_message_field(pmsg, token, body_str, i, mask);
            mask = 0;
         } else if(ft == enum_field_type::e_local_submsg){  // local defined type
            mask_operator::set_submessage(true, mask);
            parse_sub_message_field(pmsg, token, body_str, i, mask);
            mask = 0;
         } else {
            cout << "unknown keyword " << token << endl;
         }
      }
   }
}

bool parser::parse_basic_type(msg_po * const pmsg, const string & field_type, const string & body_str, int & i, uint32_t  mask){
   string field_name = get_next_token(body_str, i);
   while(i < body_str.length() && body_str[i] != '=')++i;
   ++i;
   int start = i;
   while(i < body_str.length() && body_str[i] != ';') ++i;
   int end = i;
   ++i;
   string field_num_str = body_str.substr(start, end - start);
   int field_num = get_num_from_string(field_num_str);
   field_po * pfield = new field_po(pmsg);
   pfield->fieldname = field_name;
   pfield->fieldtype = field_type;
   pfield->mask = mask;

   pmsg->name_num_map[field_name] = field_num;
   pmsg->num_name_map[field_num] =  field_name;
   pmsg->name_map.insert({field_name, pfield});
   pmsg->num_map.insert({field_num, pfield});
   return true;
}

bool parser::parse_sub_message_field(msg_po * const parent_msg, const string & field_type, const string & body_str, int & i, uint32_t mask){
   string field_name = get_next_token(body_str, i);
   while(i < body_str.length() && body_str[i] != '=')++i;
   ++i;
   int start = i;
   while(i < body_str.length() && body_str[i] != ';') ++i;
   int end = i;
   ++i;
   string field_num_str = body_str.substr(start, end - start);
   int field_num = get_num_from_string(field_num_str);
   field_po * pfield = new field_po(parent_msg);
   pfield->fieldname = field_name;
   pfield->fieldtype = field_type;
   pfield->mask = mask;

   parent_msg->name_num_map.insert({field_name, field_num});
   parent_msg->num_name_map.insert({field_num, field_name});
   parent_msg->name_map.insert({field_name, pfield});
   parent_msg->num_map.insert({field_num, pfield});
   return true;

}

bool parser::parse_reserved(msg_po * const parent_msg, const string & str, int & i){
   // simply jump over, enhance it in later version
   while(i < str.length() && str[i] != ';')++i;
   ++i;
}

bool parser::parse_oneof(msg_po * const parent_msg, const string & str, int & i){
   string oneof_name = get_next_token( str,  i);
   field_po * pfield = new field_po(parent_msg);
   CHECK_POINT(pfield);
   pfield->fieldname = oneof_name;
   pfield->oneofname = oneof_name;
   while(str[i] != '{')++i;
   int cnt = 1;
   ++i;
   int start = i;
   int end = 0;
   while(true){
      if(str[i] == '{'){
         ++cnt;
      } else if(str[i] == '}'){
         if(cnt == 1) {
            end = i;
            break;
         } else {
            --cnt;
         }
      }
      ++i;
   }
   string oneof_body = str.substr(start, end -start);
   //cout << "oneof dody str :" << oneof_body << endl;
   bool ret = parse_oneof_body(pfield, oneof_body);
   CHECK_BOOL(ret);
   return true;
}
bool parser::parse_oneof_body(field_po * const parent_field, const string & body_str){
   int i = 0;
   int start = i;
   while(i < body_str.length()){
      while(i < body_str.length() && body_str[i] != ';')++i;
      if(body_str[i] == ';') {
         string oneline = body_str.substr(start, i-start);
         bool ret = parse_oneof_field(parent_field, oneline);
         CHECK_BOOL(ret);
         ++i;
         start = i;
      }
   }
   return true;
}
bool parser::parse_oneof_field(field_po * const parent_field, const string & line_str){
   int epos = line_str.find("=");
   if(epos == string::npos) {
      cout << "error format found in oneof field :" << line_str << endl;
      return false;
   }
   string tokens = line_str.substr(0, epos);
   string field_num_str = line_str.substr(epos + 1);
   int field_num = get_num_from_string(field_num_str);
   int i = tokens.length() - 1;
   string field_name = get_next_token_reverse(tokens,  i);
   string field_type = get_next_token_reverse(tokens,  i);
   enum_field_type ft = check_type(parent_field->pmsg, field_type);
   if(ft == enum_field_type::e_invalid_type) return false;
   field_po * pfield = new field_po(parent_field->pmsg);
   CHECK_POINT(pfield);
   pfield->fieldname = field_name;
   pfield->fieldtype = field_type;
   pfield->fieldnum = field_num;
   pfield->set_in_oneof(true);
   if(ft == enum_field_type::e_global_submsg){
      pfield->set_submessage(true);
      pfield->set_global_submsg(true);
   } else if (ft == enum_field_type::e_local_submsg){
      pfield->set_submessage(true);
   }
   // as one of element is not support repeated by proto3, don't need parse the token before type 
   if(parent_field->oneof_element_map.has_value()) (*parent_field->oneof_element_map)[field_num] = pfield;
   else {
      unordered_map<int, field_po*> num_pfield_map;
      num_pfield_map[field_num] = pfield;
      parent_field->oneof_element_map = std::move(num_pfield_map);
   }
   parent_field->oneof_element_map->insert({field_num, pfield});
   parent_field->pmsg->name_num_map.insert({field_name, field_num});
   parent_field->pmsg->num_name_map.insert({field_num, field_name});
   parent_field->pmsg->name_map.insert({field_name, pfield});
   parent_field->pmsg->num_map.insert({field_num, pfield});
   
   return true;

}

bool parser::parse_sub_message(msg_po * const parent_msg, const string & str, int & i){
   string submsg_name = get_next_token( str,  i);
   while(str[i] != '{')++i;
   int cnt = 1;
   ++i;
   int start = i;
   int end = 0;
   while(true){
      if(str[i] == '{'){
         ++cnt;
      } else if(str[i] == '}'){
         if(cnt == 1) {
            end = i;
            break;
         } else {
            --cnt;
         }
      }
      ++i;
   }
   msg_po * pmsg = new msg_po;
   pmsg->msg_name = submsg_name;
   parent_msg->inner_defined_msg_set.insert(submsg_name);
   parent_msg->inner_defined_msg_map.insert({submsg_name, pmsg});
   string submsg_body = str.substr(start, end -start);
   bool ret = parse_message_body(pmsg, submsg_body);
   CHECK_BOOL(ret);
   return true;
}

void parser::print_all(){
   for(const auto & msg : defined_msg_set){
      print_msg(msg);
   }
}

void parser::print_msg(const msg_po * pmsg, int indent){
   string indent_str;
   if(indent > 0) {
      char tmp[indent];
      for(auto & v: tmp) v = ' ';
      indent_str =  {tmp, static_cast<string::size_type>(indent)};
   }
   if(indent > 0) cout << indent_str;
   cout << "message name :" << pmsg->msg_name << " begin print----- " << endl;
   for(const auto & p : pmsg->num_map){
      if(indent > 0) cout << indent_str;
      cout << "field number: " << p.first << ", field type :" << p.second->fieldtype << ", field name : " <<  p.second->fieldname; 
      if (p.second->is_required() ) cout << " required ";
      if (p.second->is_repeated()) cout << " repeated ";
      if (p.second->is_global_submsg()) cout << " global ";
      if (p.second->is_submessage()) cout << " submessage ";
      if (p.second->is_in_oneof()) cout << " inside oneof ";
      if (p.second->is_enum_type()) cout << " enum type ";   
      cout << endl;
      if(p.second->is_submessage()){
         if(p.second->is_global_submsg()){
            print_msg(p.second->fieldtype, indent + indent_size);
         } else {
            auto it = pmsg->inner_defined_msg_map.find(p.second->fieldtype);
            if(it == pmsg->inner_defined_msg_map.end()){
               cout << "can't find local submessage " << p.second->fieldtype;
               return;
            } else {
               print_msg(it->second, indent + indent_size);
            }
         }
      }
   }
   if(indent > 0) cout << indent_str;
   cout << "message name :" << pmsg->msg_name << " end print----- " << endl;
}

void parser::print_msg(const string & msg_type, int indent){
   auto it = defined_msg_map.find(msg_type);
   if(it == defined_msg_map.end()) return;
   msg_po * pmsg = it->second;
   print_msg(pmsg, indent);
}





