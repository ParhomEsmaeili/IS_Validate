#Intended for the abstract functions utilised in generating scribble prompts.

#TODO when implementing this: whatever you build here must also populate
#interactions_metadata['scribbles_axes'] (see src/data/interaction_state_construct.py) with
#one shape-[1] int tensor per scribble, same order as scribbles_labels -- the axis the scribble
#is constant along. This is a settled, already-agreed API contract with CLoPA (not a new
#decision to make here) -- do not put it back in interactions_labels['scribbles_slice_axes'],
#that placement was deliberately moved out because it broke an invariant other code relies on.

raise NotImplementedError('Scribbles can likely only be used with existing heuristic functions once we have a component level abstraction in separating prompt sampling regions.')