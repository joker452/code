# 802.11  
# Important Notes 
* PISF (Point Coordination Inter Frame Space): It's used by the Access Point (or Point Coordinator as called in this case), to gain access to the medium before any other station.  
* DIFS (Distributed Inter Frame Space): A station must sense the status of the wireless medium before transimitting. If it finds that the medium is continuously idle for DIFS duration, it's then permitted to transmit a frame.  
* EIFS (Extended Inter Frame Space): a longer IFS used by a station that has received a packet that could not be understood. It's needed to prevent the station who could not understand the duration for the Virtual Carrier Sense from colliding with a future packet belonging to the current dialog.  
* RTS (Request To Send): short control packet include source, destination, and the duration of the following transaction.  
* CTS (Clear To Send): include the same duration information as that of the RTS.  
* NAV (Network Allocation Vector): the Virtual Carrier Sense indicator a station set after receiving either RTS and/or CTS. It's used together with **Physical Carrier Sense** when sensing the medium.  
Both RTS and CTS are short frames, which reduces the overhead of collisons. (This is true if the packet is significantly bigger than the RTS, so the standard allows for short packets to be transimitted without the RTS/CTS, and this is controlled per station by a parameter called **RTSThreshold**)  
Packets that have more than one destination are not acknowledged.  
* SIFS (Short Inter Frame Space): he amount of time in microseconds required for a wireless interface to process a received frame and to respond with a response frame. It is the difference in time between the first symbol of the response frame in the air and the last symbol of the received frame in the air.  
* Virtual Carrier Sense  
To reduce the probability of two stations colliding because they cannot hear each other.  
* Frame Types  
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
00:16:b6:f7:1d:51 and 00:06:25:67:22:94.  
2. What are the intervals of time between the transimissions of the beacon frames the *linksys_ses_24086* access point? From the *30 Munroe St.* access point?  
