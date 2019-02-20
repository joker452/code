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
所有连接在`accept`前都会在队列中等待。  

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
                          
`<sys/types.h>`是`C POSIX library`的一部分，包含了C标准库以外的一些函数。根据网上查阅得到的资料，这个头文件里定义了一些和系统相关的基本数据类型。自己的理解是它并不保证跨平台的位数一致性，而是恰恰相反，一些位数不确定的数据类型，在位数不同的平台上，具体的大小会自动得到调制。  
`intptr_t`可以用于逐比特操作，但是用`uintptr_t`更好？可以用于比较指针大小？  
C语言中一系列`is_xxx(int)`函数的参数必须是`unsigned char`或者`EOF`。  
不要使用`gets`，因为不能预先知道要读入多少数据，`gets`可能会使得缓冲区溢出，应该使用`fgets`。  
`feof`不会指示文件指针现在的位置是不是位于文件尾部，它只会指明上一次尝试从文件流的读取是否超过了文件尾。  
`stat`, `fstat`, `lstat`可以用来获得文件的信息，比如大小，权限，用户、组、设备ID等。  