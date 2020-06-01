import pickle
import numpy as np
from IPython import embed


class Graphviz(object):

    def __init__(self):
        self.internal_color = "lavenderblush4"
        self.colors = [
            "aquamarine", "bisque", "blue", "blueviolet", "brown", "cadetblue",
            "chartreuse", "coral", "cornflowerblue", "crimson", "darkgoldenrod",
            "darkgreen", "darkkhaki", "darkmagenta", "darkorange", "darkred",
            "darksalmon", "darkseagreen", "darkslateblue", "darkslategrey",
            "darkviolet", "deepskyblue", "dodgerblue", "firebrick",
            "forestgreen", "gainsboro", "ghostwhite", "gold", "goldenrod",
            "gray", "grey", "green", "greenyellow", "honeydew", "hotpink",
            "indianred", "indigo", "ivory", "khaki", "lavender",
            "lavenderblush", "lawngreen", "lemonchiffon", "lightblue",
            "lightcoral", "lightcyan", "lightgoldenrodyellow", "lightgray",
            "lightgreen", "lightgrey", "lightpink", "lightsalmon",
            "lightseagreen", "lightskyblue", "lightslategray", "lightslategrey",
            "lightsteelblue", "lightyellow", "limegreen", "linen", "magenta",
            "maroon", "mediumaquamarine", "mediumblue", "mediumorchid",
            "mediumpurple", "mediumseagreen", "mediumslateblue",
            "mediumturquoise", "midnightblue", "mintcream", "mistyrose",
            "moccasin", "navajowhite", "navy", "oldlace", "olive", "olivedrab",
            "orange", "orangered", "orchid", "palegoldenrod", "palegreen",
            "paleturquoise", "palevioletred", "papayawhip", "peachpuff", "peru",
            "pink", "powderblue", "purple", "red", "rosybrown", "royalblue",
            "saddlebrown", "salmon", "sandybrown", "seagreen", "seashell",
            "sienna", "silver", "skyblue", "slateblue", "slategray",
            "slategrey", "snow", "springgreen", "steelblue", "tan", "teal",
            "thistle", "tomato", "violet", "wheat", "burlywood", "chocolate"]
        self.color_map = {}
        self.color_counter = 0

    def format_id(self, ID):
        return ("id_%s" % ID).replace('-', '')\
            .replace('#', '_HASH_').replace('.', '_DOT_')

    def clean_label(self, s):
        return s.replace("[/:.]", "_")

    def get_color(self, lbl):
        if lbl in self.color_map:
            return self.color_map[lbl]
        else:
            self.color_map[lbl] = self.colors[self.color_counter]
            self.color_counter = (self.color_counter + 1) % len(self.colors)
            return self.color_map[lbl]


    def format_node(self, mentions_dict, muid):
        euid = mentions_dict[muid]['label_document_id']
        mention_text = mentions_dict[muid]['text']
        color = self.get_color(euid)
        graphviz_format = (
                '%s[shape=egg;style=filled;color=%s;'
                'label=<%s<BR/>%s<BR/>%d:%d<BR/>>];'
                    % (self.format_id(muid),
                       color,
                       euid,
                       mention_text, 
                       mentions_dict[muid]['start_index'],
                       mentions_dict[muid]['end_index'])
        )
        return graphviz_format

    def format_edge(self, muid_a, muid_b, value, dotted=False):
        return ('%s -- %s [label=%f];'
                    % (self.format_id(muid_a), self.format_id(muid_b), value))

    def graphviz_mst(self,
                     mentions_dict,
                     mention_local_indices,
                     local_indices2mention,
                     mst,
                     pruned_mst):

        mst = mst.toarray() if hasattr(mst, 'toarray') else mst
        pruned_mst = pruned_mst.toarray() if hasattr(pruned_mst, 'toarray') else pruned_mst

        # prepare data for graphviz conversion
        local_muids = [local_indices2mention[i] for i in mention_local_indices]
        head, tail = np.where(pruned_mst > 0.0)
        pred_sl_edges = zip(
                [local_indices2mention[mention_local_indices[i]]
                    for i in head.tolist()],
                [local_indices2mention[mention_local_indices[i]]
                    for i in tail.tolist()],
                pruned_mst[pruned_mst > 0.0].tolist()
        )

        cross_mask = (mst > 0.0) & (pruned_mst <= 0.0)
        head, tail = np.where(cross_mask == True)
        pred_cross_edges = zip(
                [local_indices2mention[mention_local_indices[i]]
                    for i in head.tolist()],
                [local_indices2mention[mention_local_indices[i]]
                    for i in tail.tolist()],
                mst[cross_mask].tolist()
        )

        # gather the lines of the dot file to be compiled with `neato`
        s = []
        s.append('graph MST {')

        # add the nodes
        s.extend([self.format_node(mentions_dict, muid)
                        for muid in local_muids])

        # add the edges
        s.append('edge [len=5];')
        s.extend([self.format_edge(muid_a, muid_b, value)
                        for muid_a, muid_b, value in pred_sl_edges])
        s.append('edge [style=dotted];')
        s.extend([self.format_edge(muid_a, muid_b, value)
                        for muid_a, muid_b, value in pred_cross_edges])

        s.append('}')
        return '\n'.join(s)

    @staticmethod
    def write_mst(
            filename,
            mentions_dict,
            mention_local_indices,
            local_indices2mention,
            mst,
            pruned_mst
    ):
        gv = Graphviz()
        mst = gv.graphviz_mst(
                mentions_dict,
                mention_local_indices,
                local_indices2mention,
                mst,
                pruned_mst
        )
        with open(filename, 'w') as f:
            f.write(mst)


if __name__ == '__main__':
    with open('viz_data.pkl', 'rb') as f:
        viz_data = pickle.load(f)

    mentions_dict = viz_data['mentions_dict']
    document_id = viz_data['document_id']
    true_clusters = viz_data['true_clusters']
    mention_local_indices = viz_data['mention_local_indices']
    local_indices2mention = viz_data['local_indices2mention']
    doc_mentions = viz_data['doc_mentions']
    mst = viz_data['mst']
    pruned_mst = viz_data['pruned_mst']
    pred_clusters = viz_data['pred_clusters']

    Graphviz.write_mst('.'.join([document_id, 'mst.dot']),
                       mentions_dict,
                       mention_local_indices,
                       local_indices2mention,
                       mst,
                       pruned_mst)
