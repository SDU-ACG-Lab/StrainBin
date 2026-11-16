import copy
def reverse_complement(dna):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    # 先替换成互补碱基，再反转字符串
    return ''.join(complement[base] for base in reversed(dna.upper()))


def get_inout_edges(sub_g,node, positive=True):
    in_egs = sub_g.in_edges(node,data=True)
    pos_in_egs = []
    neg_in_egs = []
    for u,v,e_attr in in_egs:
        if e_attr['to_orient'] == "+":
            pos_in_egs.append((u,v,e_attr))
        else:
            neg_in_egs.append((u,v,e_attr))

    out_egs = sub_g.out_edges(node,data=True)
    pos_out_egs = []
    neg_out_egs = []
    for u,v,e_attr in out_egs:
        if e_attr['from_orient'] == "+":
            pos_out_egs.append((u,v,e_attr))
        else:
            neg_out_egs.append((u,v,e_attr))
    
    if positive:
        return pos_in_egs,pos_out_egs
    else:
        return neg_in_egs,neg_out_egs


def node_merge(_sub_g):
    sub_g = copy.deepcopy(_sub_g)
    # print('merging node')
    merged_node = set()
    paths =[]

    for node,n_attr in list(sub_g.nodes(data=True)):
        if node in merged_node:
            continue
        # if node != 'edge_204':
        #     continue
        #pos_in/out_egs
        st_in_egs,st_out_egs = get_inout_edges(sub_g,node)

        # print('node:',node,st_in_egs,st_out_egs)

        if len(st_in_egs) == 1 and len(st_out_egs) == 1:
            if st_in_egs[0][0] == node or st_out_egs[0][1] == node or st_in_egs[0][0] == st_out_egs[0][1]:
                # print('node is a loop')
                continue

            nt_pt = [node]
            in_egs,out_egs = st_in_egs.copy(),st_out_egs.copy()
            while True:
                next_node = out_egs[0][1]
                next_node_pos = out_egs[0][2]['to_orient'] == '+'
                in_egs,out_egs  = get_inout_edges(sub_g,next_node,next_node_pos)
                # print('next_node:',next_node,in_egs,out_egs,'merged:',merged_node)

                if len(in_egs) > 1 :
                    break
                if next_node in merged_node:
                    break
                if len(out_egs) != 1:
                    nt_pt.append(next_node)
                    merged_node.add(next_node)
                    merged_node.add(node)
                    break
                
                
                nt_pt.append(next_node)
                merged_node.add(next_node)
                merged_node.add(node)

            ft_pt = []
            in_egs,out_egs = st_in_egs.copy(),st_out_egs.copy()
            # print('ft start',ft_pt,in_egs)
            while True:
                next_node = in_egs[0][0]
                next_node_pos = in_egs[0][2]['from_orient'] == '+'
                in_egs,out_egs  = get_inout_edges(sub_g,next_node,next_node_pos)
                # print('next_node:',next_node,in_egs,out_egs,'merged:',merged_node)

                if len(in_egs) > 1 or len(out_egs) > 1:
                    break
                if next_node in merged_node:
                    break
                if len(in_egs) != 1:
                    ft_pt.append(next_node)
                    merged_node.add(next_node)
                    merged_node.add(node)
                    break
                
                
                ft_pt.append(next_node)
                merged_node.add(next_node)
                merged_node.add(node)

            # print('ft end',ft_pt)

            pt = ft_pt[::-1] + nt_pt
            # print('ft and nt',ft_pt,nt_pt)
            if len(pt)>1:
                # merged_node.update(pt)
                paths.append(pt)
                # print(node,pt)

    # print('paths:',paths)
    for path in paths:
        # print('path:',path)
        
 
        merged_ori = []
        merged_seq = ''
        merged_dp = 0

        eg_attr = sub_g.get_edge_data(path[0],path[1]).items()
        eg_attr = list(eg_attr)
        if len(eg_attr) != 1:
            print('error occurs in finding simple path')
        eg_attr = eg_attr[0][1]
        merged_ori.append(eg_attr['from_orient'])
        for i in range(1,len(path)):
            # print(path[i])
            eg_attr = sub_g.get_edge_data(path[i-1],path[i]).items()
            eg_attr = list(eg_attr)
            # print(eg_attr)
            if len(eg_attr) != 1:
                print('error occurs in finding simple path')
            eg_attr = eg_attr[0][1]
            # print(eg_attr)
            in_egs,out_egs = get_inout_edges(sub_g,path[i])
            # print(in_egs,out_egs)
            if merged_ori[-1] != eg_attr['from_orient']:
                print('error occurs in node merge')
            merged_ori.append(eg_attr['to_orient'])
        
        for i in range(len(path)):
            _n_seq = sub_g.nodes[path[i]].get('sequence','') 
            merged_seq += _n_seq if merged_ori[i] == '+' else reverse_complement(_n_seq)
        merged_len = len(merged_seq)
        for i in range(len(path)):
            _n_dp = sub_g.nodes[path[i]].get('dp',0)    
            _n_len = sub_g.nodes[path[i]].get('length',0)  
            merged_dp += _n_len/merged_len * _n_dp

        merged_dp = int(merged_dp)     


        new_node = "#".join(path) + '#' + "".join(merged_ori)

        node_attr = {
            'sequence': merged_seq, 
            'length' : len(merged_seq),
            'dp' : merged_dp
        }
        
        sub_g.add_node(new_node, **node_attr)

        # print('merged ori',merged_ori)

        # 头尾连接的都是一个节点，解决？
        pos_in_egs,pos_out_egs = get_inout_edges(sub_g,path[0],merged_ori[0]=='+')
        # print(pos_in_egs,pos_out_egs,)
        for u,_,__ in pos_in_egs:
            if u == path[1]:
                continue
            v = path[0]
            u_n_attr = sub_g.get_edge_data(u,v).items()
            n_u_attr = sub_g.get_edge_data(v,u).items()
            for _,_attr in u_n_attr:
                _attr['to_orient'] = '+' 
                sub_g.add_edge(u,new_node,**_attr)
            for _,_attr in n_u_attr:
                _attr['from_orient'] = '-' 
                sub_g.add_edge(new_node,u,**_attr)
            # sub_g.add_edge(u,new_node,**u_n_attr)
            # sub_g.add_edge(new_node,u,**n_u_attr)
        # for _,u,__ in pos_out_egs:
        #     if u == path[1]:
        #         continue
        #     v = path[0]
        #     u_n_attr = sub_g.get_edge_data(u,v).items()
        #     n_u_attr = sub_g.get_edge_data(v,u).items()
        #     for _,_attr in u_n_attr:
        #         sub_g.add_edge(u,new_node,**_attr)
        #     for _,_attr in n_u_attr:
        #         sub_g.add_edge(new_node,u,**_attr)
        #     # sub_g.add_edge(u,new_node,**u_n_attr)
        #     # sub_g.add_edge(new_node,u,**n_u_attr)

        pos_in_egs,pos_out_egs = get_inout_edges(sub_g,path[-1],merged_ori[-1]=='+')
        # print(pos_in_egs,pos_out_egs)
        # for u,_,__ in pos_in_egs:
        #     if u == path[-2]:
        #         continue
        #     v = path[-1]

        #     u_n_attr = sub_g.get_edge_data(u,v).items()
        #     n_u_attr = sub_g.get_edge_data(v,u).items()
        #     for _,_attr in u_n_attr:
        #         sub_g.add_edge(u,new_node,**_attr)
        #     for _,_attr in n_u_attr:
        #         sub_g.add_edge(new_node,u,**_attr)
        #     # sub_g.add_edge(u,new_node,**u_n_attr)
        #     # sub_g.add_edge(new_node,u,**n_u_attr)

        for _,u,__ in pos_out_egs:
            if u == path[-2]:
                continue
            v = path[-1]
            u_n_attr = sub_g.get_edge_data(u,v).items()
            n_u_attr = sub_g.get_edge_data(v,u).items()
            for _,_attr in u_n_attr:
                _attr['to_orient'] = '-'
                sub_g.add_edge(u,new_node,**_attr)
            for _,_attr in n_u_attr:
                _attr['from_orient'] = '+'
                sub_g.add_edge(new_node,u,**_attr)
            # sub_g.add_edge(u,new_node,**u_n_attr)
            # sub_g.add_edge(new_node,u,**n_u_attr)

        for _n in path:
            sub_g.remove_node(_n)

    # print('merged_subg',sub_g.nodes())
    return sub_g
