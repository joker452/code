from mininet.cli import CLI
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.topo import Topo
from mininet.log import setLogLevel
from mininet.node import OVSController


class Topology(Topo):
    def __init__(self, **opts):
        Topo.__init__(self, **opts)
        h1 = self.addHost('h1')
        h2 = self.addHost('h2')
        h3 = self.addHost('h3')
        h4 = self.addHost('h4')
        h5 = self.addHost('h5')
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')
        s3 = self.addSwitch('s3')
        s4 = self.addSwitch('s4')
        s5 = self.addSwitch('s5')
        self.addLink(s1, h1)
        self.addLink(s2, h2)
        self.addLink(s3, h3)
        self.addLink(s4, h4)
        self.addLink(s5, h5)
        self.addLink(s1, s2, bw=100, delay='0.01us')
        self.addLink(s2, s3, bw=100, delay='0.01us')
        self.addLink(s3, s4, bw=100, delay='0.01us')
        self.addLink(s4, s5, bw=100, delay='0.01us')


if __name__ == '__main__':
    setLogLevel('info')
    topo = Topology()
    net = Mininet(topo=topo, link=TCLink, autoSetMacs=True,
                  autoStaticArp=True, controller=OVSController)

    net.start()
    CLI(net)
    net.stop()
