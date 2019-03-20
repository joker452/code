# 
1. HTTP response messages may have empty body.
    * HTTP Status-Code of 200 for HEAD request can be sent without message-body
    * HTTP Status-Code of 204 and 304 MUST NOT include a message body  
 192.0.2.0/24 - This block is assigned as "TEST-NET" for use in
   documentation and example code.  It is often used in conjunction with
   domain names example.com or example.net in vendor and protocol
   documentation.  Addresses within this block should not appear on the
   public Internet  
   
# Important functions  
- getaddrinfo(const char \*node, const char \*service, const struct addrinfo \*hints, struct addrinfo \*\*res)  
比较新的一个函数。域名解析，可以得到一个链表的指针。里面包含后续用到的信息。用`freeaddrinfo`来释放链表。返回非零表示错误。使用`gai_strerror`将错误码转化为可读的字符串 。  

- socket(int domain, int type, int protocol)  
可以分别用`getaddrinfo`返回的结构体中的`ai_family`, `ai_socktype`和`ai_protocol`作为实参 。  

- bind(int sockfd, struct sockaddr \*my_addr, int addrlen)  
以前在调用之前需要手动填写`struct sockaddr_in`各个域，再将其作为参数传递。现在可以使用`getaddrinfo`返回的结构体中的`ai_addr`, `ai_addrlen`作为实参。此函数通常被称为给socket命名。如果port为0，那么会随机绑定一个端口，之后可以使用`getsockname`来获得分配的端口。  

- listen(int sockfd, int backlog)  
backlog是在established和syn_rcvd的总和。

- accept(int sockfd, struct sockaddr \*addr, socklen_t \*addrlen)  
`addr`参数通常为`struct sockaddr_storage`  

- recv(int sockfd, void \*buf, int len, int flags)  
对于stream socket，返回值为0意味着对方已经进行了顺序的关闭连接操作。某些域的datagram socket允许长度为0的datagram，收到这样的包或者请求的从socket接受的字节数为0时，返回值也会为0。  

- shutdown(int sockfd, int how)  
相比于`close`，提供了更多的控制功能。`SHUT_RD`不允许接收，`SHUT_WR`不允许发送，`SHUT_RDWR`二者都禁止。此函数不会释放文件描述符，只是改变其可用性，仍需用`close`释放。(Windows下使用`closesocket`)。 

- select(int numfds, fd_set \*readfds, fd_set \*writefds, fd_set \*exceptfds, struct timeval \*timeout)  
虽然可移植性比较强，但是很慢。`timeout`作用是当达到指定时限后，如果没有任何可用的文件描述符，函数将返回。设为0则会一直轮询，直到返回。设为`NULL`则会一直等待直到第一个文件描述符就位。有些系统会更新`timeout`为剩余的时间，有些则不会。  

- getopt(int argc, char \*const argv[], const char \*optstring)  
`argv`中以`-`开头并且不是`-`或者`--`的被称为一个选项元素。选项元素中不包含`-`的部分是选项字符。对`getopt`的连续调用会返回顺序地返回各选项元素中的选项字符。  

- getsockname(int sockfd, struct sockaddr \*addr, socklen_t \*addrlen)  
会返回当前和`sockfd`关联的地址，`addrlen`应该初始化为`addr`的大小，返回值为实际地址的大小，如果作为缓存传入的`addr`太小，那么地址会被截断。  

- pthread_create(pthread_t \*thread, const pthread_attr_t \*attr, void \*(\*start_routine) (void \*), void \*arg);
- fork(void)  
fork会创建一个新的子*进程*，子进程拥有自己的地址空间。fork在子进程中返回值为0，子进程要关闭一次socket，因为每增加一个子进程，引用数就会增加1，所以先关闭一次防止资源未被正确释放。  
- `exit` 和`_exit`(`_Exit`)  
在子进程应该使用`_exit`，因为`fork`会复制用户空间的缓冲，而`exit`会调用`atexit`注册的函数并且刷新用户空间的缓冲，这可能导致同样的数据出现两次或者使得临时文件被意外删除。应该在子进程`exec`失败或者不使用`exec`时使用`_exit`，`exec`成功时子进程会有新的缓冲。  
- `setsockopt(int s, int level, int optname, const void *optval, socklen_t optlen)`  
把socket s的optname选项设置为optval, optlen是optval的长度。  
	* Without SO_REUSEADDR, binding socketA to 0.0.0.0:21 and then binding socketB to 192.168.0.1:21 will fail (with error EADDRINUSE), since 0.0.0.0 means "any local IP address", thus all local IP addresses are considered in use by this socket and this includes 192.168.0.1, too. With SO_REUSEADDR it will succeed  
	* SO_REUSEPORT在所有当前的socket之前的socket也设置了此选项时，允许将完全相同的源地址、端口绑定到任意数目的socket。SO_REUSEADDR则只关注当前尝试bind的socket，以前bind的socket是否设置了SO_REUSEADDR无所谓。
                          
`<sys/types.h>`是`C POSIX library`的一部分，包含了C标准库以外的一些函数。根据网上查阅得到的资料，这个头文件里定义了一些和系统相关的基本数据类型。自己的理解是它并不保证跨平台的位数一致性，而是恰恰相反，一些位数不确定的数据类型，在位数不同的平台上，具体的大小会自动得到调制。  
`intptr_t`可以用于逐比特操作，但是用`uintptr_t`更好？可以用于比较指针大小？  
C语言中一系列`is_xxx(int)`函数的参数必须是`unsigned char`或者`EOF`。  
不要使用`gets`，因为不能预先知道要读入多少数据，`gets`可能会使得缓冲区溢出，应该使用`fgets`。  
`feof`不会指示文件指针现在的位置是不是位于文件尾部，它只会指明上一次尝试从文件流的读取是否超过了文件尾。  
`stat`, `fstat`, `lstat`可以用来获得文件的信息，比如大小，权限，用户、组、设备ID等。   

d7读到什么，返回什么。  
19读到什么，返回随机的协议TCP/IP中用于测试的协议  
不愿看到：有时出错，大多数时候不出错  
容忍性：不能由于一个用户出现问题而影响所有用户  
SCTP （stream control transmission protocol）：传输层协议。可靠的，面向连接的，提供拥塞控制，但是面向消息，而非字节。使用多宿主(multi-homing)和冗余路径来增加可靠性。创建方式为`socket(AF_INET, SOCK_STREAM, IPPROTO_SCTP)`  
SYN之后，发送方进入SYN SENT状态，接收方发送完SYN/ACK后进入SYN_RCVD状态，发送方接收到SYN/ACK并发送ACK后进ESTABLISHED状态，接收方收到ACK后进入ESTABLISHED状态。  
MSL: maximum segment lifetime：一个TCP端可以在互联网中存在的时间。RFC中建议为2min，但一般比这个短。
time_wait: 2倍MSL    
主动终止TCP连接的一端会进入TIME_WAIT状态
google：so_reuseport    
TCP的socket有一个send buffer，send调用成功仅仅意味着数据被成功加入到缓冲中。  
各个Linux系统中的`signal`的处理方式非常不同。  
成熟的服务器使用epoll来设置发送、接收时间限制？？？  


broken pipe由于sigpipe产生，服务器缺省处理方式为退出，这样的方式没有容忍性  
交换机一般有三种转发模式：  
* 存储转发（Store and Froward）  
交换机会将整个帧全部接收完毕，包括帧头、帧体、帧尾CRC。  
独立计算出一个CRC校验值，校验覆盖帧头、帧体部分，不包括帧尾CRC。
然后与接收到帧尾CRC进行比较：相同，校验通过，进一步转发处理；否则，校验失败，丢弃处理。  
* 剪切转发（Cut and Forward）  
交换机只接收帧的前64个字节，如果帧的长度小于64字节，视为冲突帧、无效帧，丢弃处理。  如果接收到64个字节，则立马查表转发，不进行帧的CRC校验。  
* 直接转发（Direct Forward）  
交换机只要看到帧头部的目的MAC，立马查表转发，不进行帧的CRC校验。  



