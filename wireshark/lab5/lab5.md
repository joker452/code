# UDP
The packet used to answer the following questions is [here](./udp_packet.pdf).
1. Select one UDP packet from your trace. From this packet, determine how many fields there are in the UDP header. (You shouldn't look in the textbook! Answer these questions directly from what you observe in the packet trace.) Name these fields.  
There are four fields, i.e., source port, destination port, length and checksum.  
2. By consulting the displayed information in Wireshark's packet content field for this packet, determine the length (in bytes) of each of the UDP header fields.  
All of them are 2 bytes.  
3. The value in the Length field is the length of what? (You can consult the text for this answer). Verify your claim with your captured UDP packet.  
It's the total length of the UDP header and its payload in bytes.  
4. What is the maximum number of bytes that can be included in a UDP payload?  
The length field is 2 bytes, which means the maximum length of UDP header plus its payload is 65,536. Thus the maximum number of bytes that can be included in a UDP payload is 65,528.  
5. What is the largest possible source port number?  
65,536.  
6. What is the protocol number for UDP? Give your answer in both hexadecimal and decimal notation. To answer this question, you'll need to look into the Protocol field of the IP datagram containing this UDP segment (see Figure 4.13 in the text, and the discussion of IP header fields).  
0x11 or 17.  
7. Examine a pair of UDP packets in which your host sends the first UDP packet and the second UDP packet is a reply to this first UDP packet. Describe the relationship between the port numbers in the two packets.  
It's a pair of MDNS packets.  
Port numbers in the order of sender and receiver:  
1\. 5353, 5353.  
2\. 5353, 5353.  
They should be in reverse order, but as they are the same in this special case, there is no difference.  
 
