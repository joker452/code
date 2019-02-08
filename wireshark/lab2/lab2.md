# HTTP  
## Important notes  
HTTP uses persistent connections by default but can be configured to use non-persistent connections.  
HTTP is stateless.  
With persistent connections, HTTP client can make to back-to-back requests without waiting for replies to pending requests.  
HTTP/2 builds on HTTP 1.1 by allowing multiple requests and replies to be interleaved in the same connection, and a mechanism for prioritizing HTTP message requests and replies within this connection.  
**The HTTP data is transmitted in plain text.**  
The first line of an HTTP request is called the request line, and the subsequent lines are called the header lines.  
The request line is composed of three fields: method, URL, and version.  
The HEAD method is similar to GET method except that the server will leave out the objects when response to it.  
The first line of an HTTP response is a status line, and the subsequent lines are header lines.  
The status line is composed of three parts: the protocol version, a status code and a corresponding status message.  
Cookie technology has four components: a cookie header line in the HTTP response message, a cookie header line in the HTTP request message, a cookie file kept on the user's end system and managed by the user's browser, and a back-end database at the Web site.  
A Web cache, also called a proxy server -- is a network entity that satisfies HTTP request on the behalf of an origin Web server.  
Typically a Web cache is purchased and installed by an ISP.  
Web cache can reduce traffic on an institution's access link to the Internet, and Web traffic in the Internet as a whole.  
An HTTP request message is a so-called conditional GET message if the request message uses the GET method and the request message includes an If-Modified-Since header line.  

1. Is your browser running HTTP version 1.0 or 1.1?  What version of HTTP is the server running?  
1.1, 1.1.  
2. What languages (if any) does your browser indicate that it can accept to the server?  
English in US style is indicated with a priority of 1.0, and English with a priority of 0.5.  
3. What is the IP address of your computer?  Of the gaia.cs.umass.edu server?  
192.168.1.102, 128.119.245.12.  
4. What is the status code returned from the server to your browser?  
200.  
5. When was the HTML file that you are retrieving last modified at the server?  
Tue, 23 Sep 2003 05:29:00 GMT.    
6. How many bytes of content are being returned to your browser?  
73.  
7. By inspecting the raw data in the packet content window, do you see any headers within the data that are not displayed in the packet-listing window?  If so, name one.  
No.   
8. Inspect the contents of the first HTTP GET request from your browser to the server. Do you see an "IF-MODIFIED-SINCE" line in the HTTP GET?  
No.  
9. Inspect the contents of the server response. Did the server explicitly return the contents of the file? How can you tell?  
Yes. The Content-Length header line is positive.  
10. Now inspect the contents of the second HTTP GET request from your browser to the server.  Do you see an “IF-MODIFIED-SINCE:” line in the HTTP GET? If so, what information follows the “IF-MODIFIED-SINCE:” header?  
Yes. It's followed by "Tue, 04 Dec 2018 06:59:01 GMT", which specifies the time of the last change of the cached file.  
11. What is the HTTP status code and phrase returned from the server in response to this second HTTP GET?  Did the server explicitly return the contents of the file? Explain.    
No. The status code is  304, which means the file cached by the browser doesn't change, so there is no need for the server to return the contents of the file.  
12. How many HTTP GET request messages did your browser send?  Which packet number in the trace contains the GET message for the Bill or Rights?  
One. 2311.  
13. Which packet number in the trace contains the status code and phrase associated with the response to the HTTP GET request?
2311.  
14. What is the status code and phrase in the response?  
200/. OK.    
15. How many data-containing TCP segments were needed to carry the single HTTP response and the text of the Bill of Rights?  
4/.    
16. How many HTTP GET request messages did your browser send? To which Internet addresses were these GET requests sent?  
3/.  128.119.245.12.    
17. Can you tell whether your browser downloaded the two images serially, or whether they were downloaded from the two web sites in parallel?  Explain.  
In parallel. The data header in the response is same for the two HTTP messages.  
18. What is the server's response (status code and phrase) in response to the initial HTTP GET message from your browser?  
401, "Unauthorized".  
19. When your browser's sends the HTTP GET message for the second time, what new field is included in the HTTP GET message?  
Authorization and Upgrade-Insecure-Requests.    
