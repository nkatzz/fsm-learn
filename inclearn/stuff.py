x = """
	s0 [shape="circle" label="0"];
	s1 [shape="doublecircle" label="1"];
	s2 [shape="circle" label="2"];
	s3 [shape="circle" label="3"];
	s4 [shape="circle" label="4"];
	s5 [shape="circle" label="5"];
	s6 [shape="circle" label="6"];
	s7 [shape="doublecircle" label="7"];
	s8 [shape="circle" label="8"];
	s9 [shape="circle" label="9"];
	s10 [shape="circle" label="10"];
	s11 [shape="circle" label="11"];
	s12 [shape="doublecircle" label="12"];
	s13 [shape="circle" label="13"];
	s14 [shape="circle" label="14"];
	s15 [shape="circle" label="15"];
	s16 [shape="circle" label="16"];
	s17 [shape="circle" label="17"];
	s18 [shape="circle" label="18"];
	s19 [shape="doublecircle" label="19"];
	s20 [shape="circle" label="20"];
	s21 [shape="doublecircle" label="21"];
	s22 [shape="circle" label="22"];
	s23 [shape="circle" label="23"];
	s24 [shape="circle" label="24"];
	s25 [shape="circle" label="25"];
	s26 [shape="circle" label="26"];
	s27 [shape="doublecircle" label="27"];
	s28 [shape="circle" label="28"];
	s29 [shape="doublecircle" label="29"];
	s30 [shape="circle" label="30"];
	s31 [shape="circle" label="31"];
	s32 [shape="doublecircle" label="32"];
	s33 [shape="circle" label="33"];
	s34 [shape="circle" label="34"];
	s35 [shape="circle" label="35"];
	s36 [shape="circle" label="36"];
	s0 -> s1 [label="t"];
	s1 -> s1 [label="t"];
	s2 -> s35 [label="t"];
	s3 -> s14 [label="s"];
	s3 -> s22 [label="t"];
	s4 -> s10 [label="r"];
	s4 -> s24 [label="s"];
	s4 -> s17 [label="t"];
	s5 -> s5 [label="q"];
	s5 -> s10 [label="r"];
	s5 -> s10 [label="s"];
	s5 -> s10 [label="t"];
	s6 -> s6 [label="c"];
	s6 -> s6 [label="d"];
	s6 -> s6 [label="e"];
	s6 -> s6 [label="f"];
	s6 -> s6 [label="g"];
	s6 -> s6 [label="h"];
	s6 -> s6 [label="i"];
	s6 -> s6 [label="j"];
	s6 -> s6 [label="k"];
	s6 -> s6 [label="l"];
	s6 -> s6 [label="m"];
	s6 -> s6 [label="n"];
	s6 -> s6 [label="o"];
	s6 -> s6 [label="p"];
	s6 -> s5 [label="q"];
	s6 -> s5 [label="r"];
	s6 -> s6 [label="s"];
	s6 -> s5 [label="t"];
	s7 -> s0 [label="t"];
	s8 -> s30 [label="s"];
	s8 -> s9 [label="t"];
	s9 -> s34 [label="t"];
	s10 -> s4 [label="r"];
	s10 -> s4 [label="s"];
	s10 -> s25 [label="t"];
	s11 -> s3 [label="s"];
	s11 -> s4 [label="t"];
	s12 -> s21 [label="t"];
	s13 -> s7 [label="t"];
	s14 -> s25 [label="s"];
	s14 -> s30 [label="t"];
	s15 -> s30 [label="s"];
	s15 -> s14 [label="t"];
	s16 -> s28 [label="t"];
	s17 -> s11 [label="s"];
	s17 -> s15 [label="t"];
	s18 -> s22 [label="s"];
	s18 -> s11 [label="t"];
	s19 -> s29 [label="t"];
	s20 -> s12 [label="t"];
	s21 -> s13 [label="t"];
	s22 -> s15 [label="s"];
	s22 -> s8 [label="t"];
	s23 -> s33 [label="t"];
	s24 -> s26 [label="s"];
	s24 -> s18 [label="t"];
	s25 -> s18 [label="s"];
	s25 -> s26 [label="t"];
	s26 -> s17 [label="s"];
	s26 -> s3 [label="t"];
	s27 -> s19 [label="t"];
	s28 -> s20 [label="t"];
	s29 -> s7 [label="t"];
	s30 -> s8 [label="s"];
	s30 -> s36 [label="t"];
	s31 -> s23 [label="t"];
	s32 -> s27 [label="t"];
	s33 -> s2 [label="t"];
	s34 -> s31 [label="t"];
	s35 -> s32 [label="t"];
	s36 -> s16 [label="t"];
"""

lines = x.split("\n")
final_state_lines = [x for x in lines if 'shape="doublecircle"' in x]
final_states = [x.split(" ")[0] for x in final_state_lines]
final_state_preds = map(lambda x: 'final({0}).'.format(x), final_states)

transition_lines = [x for x in lines if '->' in x]
print(transition_lines)

transitions_lists, labels = [x.split('[')[0].split('->') for x in transition_lines], [
    x.split("=")[1].split("]")[0].split('"')[1] for x in transition_lines]
zipped = zip(transitions_lists, labels)
transition_preds = map(lambda x: 'transition({0},{1},{2}).'.format(x[0][0].strip(), x[1].strip(), x[0][1].strip()),
                       zipped)
#all = transition_preds + final_states
print(' '.join(transition_preds + final_state_preds))
