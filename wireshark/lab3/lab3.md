Three types of DNS server: root, TLD, and authoritative.  
Resource records: (name, value, type, ttl), the meaning of name and value depend on type.  
1. Run *nslookup* to obtain the IP address of a Web server in Asia. What is the IP address of that server?  
Use the command 'nslookup apigateway.cn-north-1.amazonaws.com.cn'.  
Two IP addresses are returned: 54.223.20.220 and 54.223.77.30.  
2. Run *nslookup* to determine the authoritative DNS servers for a university in Europe.  
Use the command 'nslookup -type=NS www.cam.ac.uk'  
The response is as follows:   
服务器:  dns-a.tsinghua.edu.cn  
Address:  166.111.8.28  
cam.ac.uk  
        primary name server = ipreg.csi.cam.ac.uk  
        responsible mail addr = hostmaster.cam.ac.uk  
        serial  = 1544405154  
        refresh = 1800 (30 mins)  
        retry   = 900 (15 mins)  
        expire  = 604800 (7 days)  
        default TTL = 3600 (1 hour) 
3. Run *nslookup* so that one of the DNS servers obtained in Question 2 is queired for the mail servers for Yahoo! mail. What is its IP address?  
The DNS server doesn't know the IP address of Yahoo! mail. The query sent just times out.   
4. Locate the DNS query and response messages. Are they sent over UDP or TCP?  
They are sent over UDP.  
5. What is the destination port for the DNS query message? What is the source port of DNS response message?  
Both are 53.  
6. To what IP address is the DNS query message sent? Use ipconfig to determine the IP address of your local DNS server. Are these two IP address the same?  
It can be seen in the Wiershark that the DNS query is sent to 166.111.8.28. According to ipconfig, there are two IP addresses for the local DNS server, i.e., 166.111.8.28 and 166.11.8.29.  
7. Examine the DNS query message. What "Type" of DNS query is it? Does the query message contain any "answers"?  
It's a type A DNS query. No, the field 'number of answer RRS' is 0.  
8. Examine the DNS response message. How many "answers" are provided? What do each of these answers contain?  
Three. 
The content is as follows:  
www.ietf.org: type CNAME, class IN, cname www.ietf.org.cdn.cloudflare.net  
www.ietf.org.cdn.cloudflare.net: type A, class IN, addr 104.20.1.85  
www.ietf.org.cdn.cloudflare.net: type A, class IN, addr 104.20.0.85  
9. Consider the subsequent TCP SYN packet sent by your host. Does the destination IP address of the SYN packet correspond to any of the IP addresses provided in the DNS response message?  
Yes, the destination IP address is 104.20.1.85.  
10. This web page contain images. Before retrieving each image, does your host issue new DNS queries?  
No.  
11. What is the destination port for the DNS query message? What is the source port of DNS response message?  
Both are 53.  
12. To what IP address is the DNS query message sent? Is this the IP address of your default local DNS server?  
166.111.8.28. Yes.  
13. Examine the DNS query message. What "Type" of DNS query is it? Does the query message contain any "answers"?  
Type A. No.  
14. Examine the DNS response message. How many "answers" are provided? What do each of these answers contain?  
3.  
www.mit.edu: type CNAME, class IN, cname www.mit.edu.edgekey.net  
www.mit.edu.edgekey.net: type CNAME, class IN, cname e9566.dscb.akamaiedge.net  
e9566.dscb.akamaiedge.net: type A, class IN, addr 23.213.32.34  
16. To what IP address is the DNS query message sent? Is this the IP address of your default local DNS server?
166.111.8.28. Yes.  
17. Examine the DNS query message. What "Type" of DNS query is it? Does the query message contain any "answers"?
Type NS. No.  
18. Examine the DNS response message. What MIT nameservers does the response message provide? Does this response message also provide the IP addresses of the MIT nameservers?  
asia2.akam.net, use2.akam.net, use5.akam.net, ns1-173.akam.net, ns1-37.akam.net, eur5.akam.net, usw2.akam.net, asia1.akam.net.  
Yes, it also provides the IPv4 addresses of these 8 nameservers in the Additional informatio section, and IPv6 addresses of 3 servers among them.  
20. To what IP address is the DNS query message sent? Is this the IP address of your default local DNS server? If not, what does the IP address correspond to?  
It first sends two DNS query messages to get the IPv4 and IPv6 address of bitsy.mit.edu, which the DNS server in this case.  
Then it sends DNS query to 18.72.0.3 (the IP address it got just now) to get the corresponding IP address of the hostname www.aiit.or.kr.  
21. Examine the DNS query message. What "Type" of DNS query is it? Does the query message contain any "answers"?
Type A. No.  
22. Examine the DNS response message. How many "answers" are provided? What do each of these answers contain?
There is no answer at all. It just times out.    
