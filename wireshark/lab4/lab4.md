1. What is the IP address and TCP port number used by the client computer (source) that is transferring the file to gaia.cs.umass.edu? To answer this question, it's probably easiest to select an HTTP message and explore the details of the TCP packet used to carry this HTTP message, using the "details of the selected packet header window".  
IP address: 192.168.1.102.  
TCP port number: 1161.  
2. What is the IP address of gaia.cs.umass.edu? On what port number is it sending and receiving TCP segments for this connection?  
IP address: 128.119.245.12.  
TCP port number: 80.  
3. If you have been able to create your own trace, what is the IP address and TCP port number used by your client computer (source) to transfer the file to gaia.cs.umass.edu?  
IP address: 10.235.177.88.  
TCP port number: 62760.  
4. What is the sequence number of the TCP SYN segment that is used to initiate the TCP connection between the client computer and gaia.cs.umass.edu? What is it in the segment that identifies the segment as a SYN segment?  
Sequence number: 232,129,012. Relative sequence number: 0.  
The SYN field in the Flags is set to 1 to identifiy the segment as a SYN segment.  
5. What is the sequence number of the SYN ACK segment sent by gaia.cs.umass.edu to the client computer in reply to the SYN? What is the value of the Acknowledgement field in the SYN ACK segment? How did gaia.cs.umass.edu. determine that value? What is it in the segment that identifies the segment as a SYN ACK segment?  
Sequence number: 882,061,785. Relative sequence number: 0.  
Acknowledgement number: 232,129,013. Relative sequence number: 1. This value is determined by adding 1 to the sequence number in the SYN segment received.  
Both SYN and ACK fields in Flags are set to 1, which identifies the segment as a SYN ACK segment.  
6. What is the sequence number of the TCP segment containing the HTTP POST command? Note that in order to find the POST command, you'll need to dig into the packet content field at the bottom of the Wiershark window, looking for a segment with a "POST" within its DATA field.  
Sequence number: 231,129,013. Relative sequence number: 1.  
7. Consider the TCP segment containing the HTTP POST as the first segment in the TCP connection. What are the sequence numbers of the first six segments in the TCP connection (including the segment containing the HTTP POST)? When was the ACK for each segment received? Given the difference between when each TCP segment was sent, and when its acknowledgement was received, what is the RTT for each of the six segments? What is the EstimatedRTT value after the receipt of each ACK? Assume that the value of the EstimatedRTT is equal to the measured RTT for the first segment, and then is computed using the EstimatedRTT equation on page 242 for all subsequent segments.  
The sequence numbers are: 232,129,013, 232,129,578, 232,131,038, 232,132,498, 232,133,958, and 232,135,418.  
Ack arrive time: no, 21:44:20.624318, 21:44:20.647675, 21:44:20.694466, 21:44:20.739499, 21:44:20.787680.  
RTT: 0.570182s, 0.569960s, 0.570763s, 0.570756s, 0.570748s, 0.570725s.  
EstimatedRTT: 0.570182s, 0.57015425s, 0.57023034375s, 0.57029605078125s, 0.5703525444335937s, 0.5703991013793945s.  
8. What is the length of each of the first six TCP segments?  
565, 1460, 1460, 1460, 1460, 1460.  
9. What is the minimum amount of available buffer space adversided at the received for the entire trace? Does the lack of receiver buffer space ver throttle the sender?  
5840, no.  
10. Are there any retransmitted segments in the trace file? What did you check for (in the trace) in order to answer this question?  
No. Check the sequence number field. If there were any retrasmitted segments, the same sequence number would appear more than once.  
11. How much data does the receiver typically acknowledge in an ACK? Can you identify cases where the receiver is ACKing every other received segment.  
Typically the receiver acknowledge all the data it received since it acknowledged last time.  
Typically, after the arrival of an in-order segment, the receiver will wait up to 500ms for arrival of another in-order segment. If next in-order segment does not arrive, it will send an ACK.In segment 80, the receiver is ACKing 2352 bytes, because it receives two consecutive segments, the first of which is waiting for an ACK, so it just sends a single cumulative ACK.  
12. What is the throughput (bytes transferred per unit time) for the TCP connection? Explain how you calculated this value.  
It can be seen from [here](./throughput.pdf). It is calculated by Wireshark.  
Or it can be calculated by dividing all the data transmitted by the duration of the connection.  
13. Use the *Time-Sequence-Graph(Stevens)* plotting tool to view the sequence number versus time plot of segments being sent from the client to the gaia.cs.umass.edu server. Can you identify where TCP's slow start phase begins and ends, and where congestion avoidance takes over? Comment on ways in which the measured data differs from the idealized bahavior of TCP that we've studied in the text.  
The TCP sender is not sending data aggressively enough to push to the congestion state. The application at most sends out a data block of 8192 bytes before it receives the acknowledgement. It indicates before the end of the slow start phase, the application already stops transmission.  
The idealized behavior of TCP in the text assumes that TCP senders are aggressive in sending data. In the practice,TCP behavior also largely depends on the application. In some web applications, the web objects have very small sizes. Before the end of the slow start phase, the transmission is over; hence, the transmission of these objects suffers from the unnecessary long delay because of the slow start phase of TCP.
