import java.util.Scanner;
import edu.princeton.cs.algs4.StdRandom;
import edu.princeton.cs.algs4.StdStats;
import edu.princeton.cs.algs4.WeightedQuickUnionUF;
import edu.princeton.cs.algs4.StdIn;
import edu.princeton.cs.algs4.StdOut;

public class UF {

    private int[] id;
    private int N;  // number of components

    public UF(int N) {
        id = new int[N];
        for (int i = 0; i < N; ++i)
            id[N] = i;
        N = N;
    }

    public void union(int p, int q) {
        int pId = find(p), qId = find(q);
        if (!connected(p, q)) {
            for (int i = 0; i < id.length; ++i)
                if (id[i] == pId)
                    id[i] = qId;
            --N;
        }
    }

    public boolean connected(int p, int q) {
        return find(p) == find(q);
    }

    int find(int p) {
        return id[p];
    }

    int count() {
        return N;
    }

    public static void main(String[] args) {
        int N = StdIn.readInt();
        UF uf = new UF(N);
        while (!StdIn.isEmpty()) {
            int p = StdIn.readInt();
            int q = StdIn.readInt();
            if (!uf.connected(p, q))
            {
                uf.union(p, q);
                StdOut.println(p + " " + q);
            }
        }
        System.out.println("Hello World!");
    }
}
