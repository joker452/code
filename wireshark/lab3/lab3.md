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

