# NAT  
1. What is the IP address of the client?  
192.168.1.100.  
2. The client actually communicates with several different Google servers in order to implement "safe browsing". The main Google server that will serve up the main Google web page has IP address 64.233.169.104. In order to display only those frames containing HTTP messages that are sent to/from this Google server, enter the expression "http && ip.addr == 64.233.169.104" (without quotes) into the Filter: field in Wireshark.  
3. Consider now the HTTP GET sent from the client to the Google server (whose IP address is IP address 64.233.169.104) at time 7.109267. What are the source and destination IP addresses and TCP source and destination ports on the IP datagram carrying this HTTP GET?  
source: 192.168.1.100, port 4335.  
destination: 64.233.169.104, port 80.  
4. At what time is the corresponding 200 OK HTTP message received from the Google server? What are the source and destination IP addresses and TCP source and destination ports on the IP datagram carrying this HTTP 200 OK message?  
It's received at 7.158797s.  
source: 64.233.169.104, port 80.    
destination: 192.168.1.100, port 4335.  
5. Recall that before a GET command can be sent to an HTTP server, TCP must first set up a connection using the three-way SYN/ACK handshake. At what time is the client-to-server TCP SYN segment sent that sets up the connection used by the GET sent at time 7.109267? What are the source and destination IP addresses and source and destination ports for the TCP SYN segment? What are the source and destination IP addresses and souce and destination ports of the ACK sent in response to the SYN. At what time is this ACK received at the client?  
The segment which sets up the connection is sent at 7.075657.  
source: 192.168.1.00, port 4335.  
destination: 64.233.169.104, port 80.  
The ACK segment is received at 7.108986.    
source: 64.233.169.104, port 80.  
destination: 192.168.1.100, port 4335.  
6. In the NAT\_ISP\_side trace file, find the HTTP GET message was sent from the client to the Google server at time 7.109267 (where t=7.109267 is time at which this was sent as recorded in the NAT\_home\_side trace file). At what time does this message appear in the NAT\_ISP\_side trace file?  What are the source and destination IP addresses and TCP source and destination ports on the IP datagram carrying this HTTP GET (as recording in the NAT\_ISP\_side trace file)? Which of these fields are the same, and which are different, than in your answer to question 3 above?  
6.069168.  
source: 71.192.34.104, port 4335.  
destination: 64.233.169.104, port 80.  
The source IP address is different, and the rest is the same.  
7. Are any fields in the HTTP GET message changed? Which of the following fields in the IP datagram carrying the HTTP GET are changed: Version, Header Length, Flags, Checksum.  If any of these fields have changed, give a reason (in one sentence) stating why this field needed to change.  
No, the NAT router only changes the IP address and port number.  
The checksum field is changed. It has to change because the source address or the destination address field will change, and the checksum is calculated baesd on these fields.  
8. In the NAT\_ISP\_side trace file, at what time is the first 200 OK HTTP message received from the Google server?  What are the source and destination IP addresses and TCP source and destination ports on the IP datagram carrying this HTTP 200 OK message? Which of these fields are the same, and which are different than your answer to question 4 above?
6.117570.  
source: 64.233.169.104, port 80.  
destination: 71.192.34.104, port 4335.  
The destination IP address changes, and the rest is the same.  
9. In the NAT\_ISP\_side trace file, at what time were the client-to-server TCP SYN segment and the server-to-client TCP ACK segment corresponding to the segments in question 5 above captured? What are the source and destination IP addresses and source and destination ports for these two segments? Which of these fields are the same, and which are different than your answer to question 5 above?  
1\. TCP SYN:  
6.035475.  
source: 71.192.34.104, port 4335.  
destination: 64.233.169.104, port 80.  
The source IP address changes, and the rest remians the same.  
2\. TCP SYN ACK:  
6.067775.  
source: 64.233.169.104, port 80.  
destination: 71.192.34.104, port 4335.  
The destination IP address changes, and the rest remians the same.  

  

