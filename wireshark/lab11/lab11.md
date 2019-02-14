# 802.11  
# Important Notes 
* PISF (Point Coordination Inter Frame Space): It's used by the Access Point (or Point Coordinator as called in this case), to gain access to the medium before any other station.  
* DIFS (Distributed Inter Frame Space): A station must sense the status of the wireless medium before transimitting. If it finds that the medium is continuously idle for DIFS duration, it's then permitted to transmit a frame.  
* EIFS (Extended Inter Frame Space): a longer IFS used by a station that has received a packet that could not be understood. It's needed to prevent the station who could not understand the duration for the Virtual Carrier Sense from colliding with a future packet belonging to the current dialog.  
* RTS (Request To Send): short control packet include source, destination, and the duration of the following transaction. The duration is the time in microseconds, required to transmit the next Data or Management frame, plus one CTS frame, plus one ACK frame, plus three SIFS intervals.  
* CTS (Clear To Send): include the same duration information as that of the RTS. The duration is the value obtained from the duration field of the immediately RTS frame, minus the time, in microseconds, required to transmit the CTS frame and its SIFS interval.  
* ACK: If the more fragment bit was set to 0 in the previous frame, the duration is set to 0, otherwise it's obtained from the duration field of the previous frame, minus the time, in microseconds, required to transmit the ACK frame and its SIFS interval.  
* NAV (Network Allocation Vector): the Virtual Carrier Sense indicator a station set after receiving either RTS and/or CTS. It's used together with **Physical Carrier Sense** when sensing the medium.  
Both RTS and CTS are short frames, which reduces the overhead of collisons. (This is true if the packet is significantly bigger than the RTS, so the standard allows for short packets to be transimitted without the RTS/CTS, and this is controlled per station by a parameter called **RTSThreshold**)  
Packets that have more than one destination are not acknowledged.  
* SIFS (Short Inter Frame Space): he amount of time in microseconds required for a wireless interface to process a received frame and to respond with a response frame. It is the difference in time between the first symbol of the response frame in the air and the last symbol of the received frame in the air.  
* Virtual Carrier Sense  
To reduce the probability of two stations colliding because they cannot hear each other.  
* FrameI Types  
	* Data Frames: used for data transmission  
	* Control Frames: used to control access to the medium(e.g. RTS, CTS and ACK)  
	* Management Frames: transmitted the same way as data frames to exchange management information, but are not forwarded to upper layers.  
* Address Fields  
A frame may contain up to 4 Addresses depending on the ToDS and FromDS bits defined in the Frame Control field.  
	* Address-1 is always the Recipient Address(i.e. the station on the BSS who is the immediate recipient of the packet), if ToDS is set this the Address of the AP, otherwise this is the address of the end-station.  
	* Address-2 is always the Transmitter Address(i.e. the station who is physically transmitting the packet), if FromDS is set this is the address of the AP, otherwise it is the address of the station.  
	* Address-3 is in most cases the remaining, missing address, on a frame with FromDS set to 1, then the Address-3 is the original Source Address, if the frame has the ToDS set then Address 3 is the destination address.  
	* Address-4 is used on the special case where a Wireless Distribution System is used, and the frame is being trasmitted from one Access Point to another, in this case both the ToDS and FromDS bits are set, so both the original destination and the original source address are missing.  
1. What are the SSIDs of the two access points that are issuing most of the beacon frames in this trace?  
30 Munroe St and linksys12.  
*answers from the solution: It seems that there are totally 3 SSID in the frame.The last one is linksys_ses_24086.*  
2. What are the intervals of time between the transimissions of the beacon frames the *linksys_ses_24086* access point? From the *30 Munroe St.* access point?  
Both are 0.102400 seconds.  
3. What (in hexadecimal notation) is the source MAC address on the beacon frame from 30 Munroe St? Recall from Figure 7.13 in the text that the source, destination, and BSS are three addresses used in an 802.11 frame.  For a detailed discussion of the 802.11 frame structure, see section 7 in the IEEE 802.11 standards document (cited above).  
00:16:b6:f7:1d:51.  
4. What (in hexadecimal notation) is the destination MAC address on the beacon frame from 30 Munroe St?  
ff:ff:ff:ff:ff:ff.  
5. What (in hexadecimal notation) is the MAC BSS id on the beacon frame from 30 Munroe St?  
00:16:b6:f7:1d:51.  
6. The beacon frames from the 30 Munroe St access point advertise that the access point can support four data rates and eight additional “extended supported rates.” What are these rates?  
4 data rates: 1 Mbit/sec, 2 Mbit/sec, 5.5 Mbit/sec, 11 Mbit/sec.  
8 extended supported rates: 6 Mbit/sec, 9 Mbit/sec, 12 Mbit/sec, 18 Mbit/sec, 24 Mbit/sec, 36 Mbit/sec, 48 Mbit/sec, 54 Mbit/sec.  
7. Find the 802.11 frame containing the SYN TCP segment for this first TCP session (that downloads alice.txt).  What are three MAC address fields in the 802.11 frame? Which MAC address in this frame corresponds to the wireless host (give the hexadecimal representation of the MAC address for the host)? To the access point?  To the first-hop router?  What is the IP address of the wireless host sending this TCP segment?  What is the destination IP address?  Does this destination IP address correspond to the host, access point, first-hop router, or some other network-attached device?  Explain.  
address-1 (BSSID, access point): 00:16:b6:f7:1d:51.  
address-2 (source address, wireless host): 00:13:02:d1:b6:4f.  
address-3 (destination address, first-hop router): 00:16:b6:f4:eb:a8.  
wireless host IP address: 192.168.1.109.  
destination IP address: 128.119.245.12. It corresponds to a host, because it's the destination address of the server which will offer the content the following HTTP messages request.  
8. Find the 802.11 frame containing the SYN ACK segment for this TCP session. What are three MAC address fields in the 802.11 frame? Which MAC address in this frame corresponds to the host? To the access point?  To the first-hop router? Does the sender MAC address in the frame correspond to the IP address of the device that sent the TCP segment encapsulated within this datagram? (Hint: review Figure 6.19 in the text if you are unsure of how to answer this question, or the corresponding part of the previous question.  It’s particularly important that you understand this). 
address-1 (destination address, host): 91:2a:b0:49:b6:4f.  
address-2 (BSSID, access point): 00:16:b6:f7:1d:51.  
address-3 (source address, first hop router): 00:16:b6:f4:eb:a8.  
No, because the sender MAC address is that of the access point.  
9. What two actions are taken (i.e., frames are sent) by the host in the trace just after t=49, to end the association with the 30 Munroe St AP that was initially in place when trace collection began?  (Hint: one is an IP-layer action, and one is an 802.11-layer action).  Looking at the 802.11 specification, is there another frame that you might have expected to see, but don’t see here?  
10. Examine the trace file and look for AUTHENICATION frames sent from the host to an AP and vice versa.  How many AUTHENTICATION messages are sent from the wireless host to the linksys\_ses\_24086 AP (which has a MAC address of Cisco\_Li\_f5:ba:bb) starting at around t=49? .
6.  
11. Does the host want the authentication to require a key or by open?  
open.  
12. Do you see a reply AUTHENTICATION from the *linksys_ses_24086* AP in the trace?  
No.  
13. Now let’s consider what happens as the host gives up trying to associate with the linksys\_ses\_24086 AP and now tries to associate with the 30 Munroe St AP. Look for AUTHENICATION frames sent from the host to and AP and vice versa. At what times are there an AUTHENTICATION frame from the host to the 30 Munroe St. AP, and when is there a reply AUTHENTICATION sent from that AP to the host in reply? (Note that you can use the filter expression “wlan.fc.subtype == 11and wlan.fc.type == 0 and wlan.addr == IntelCor\_d1:b6:4f” to display only the AUTHENTICATION frames in this trace for this wireless host.)  
63.168087, 63.169071.  
14. An ASSOCIATE REQUEST from host to AP, and a corresponding ASSOCIATE RESPONSE frame from AP to host are used for the host to associated with an AP. At what time is there an ASSOCIATE REQUEST from host to the 30 Munroe St AP?  When is the corresponding ASSOCIATE REPLY sent? (Note that you can use the filter expression “wlan.fc.subtype < 2 and wlan.fc.type == 0 and
wlan.addr == IntelCor\_d1:b6:4f” to display only the ASSOCIATE REQUEST and ASSOCIATE RESPONSE frames for this trace.)  
63.169910, 63.192101.  
15. What transmission rates is the host willing to use?  The AP?   To answer this question, you will need to look into the parameters fields of the 802.11 wireless LAN management frame.  
Both are 1, 2, 5.5, 6, 9, 11, 12, 18, 24, 36, 48, 54 Mbit/sec.  
16. What are the sender, receiver and BSS ID MAC addresses in these frames?  What is the purpose of these two types of frames?  
probe request: both receiver address and BSS ID is ff:ff:ff:ff:ff:ff. The sender address is 00:12:f0:1f:57:13.  
probe response: both sender address and BSS ID is 00:16:b6:f7:1d:51, and the receiver address is 00:12:f0:1f:57:13.  

