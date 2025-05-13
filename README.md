# 107 Midterm CCT Project
Introduction:
This project looked at Romney, Weller, and Batchelder's Cultural Consesnus Theory and creates a basic model to analyze a dataset about plant knowledge. The CCT model attempts to look at and infer compentence and culturally shared consesuses. In both real life and in this example, the model looks at observed responses (X) and compared how similar they are with not observed correct answers (Z) that come from the group's consensus. There is also a competence score (D), with high competence scores making it more likely for observed responses to match the group's consensus. Within my model, I used the Bayesian hierarchial model approach to CCT. I used the example from the assignment as a guideline. Ultimately, I ended up with this:

My Model:
D = pm.Uniform("D", lower=0.5, upper=1.0, shape=N)
Z = pm.Bernoulli("Z", p=0.5, shape=M)

D_reshaped = D[:, None]  
Z_reshaped = Z[None, :]

p = Z_reshaped * D_reshaped + (1 - Z_reshaped) * (1 - D _reshaped)

For D, i decided to make the lower bound 0.5 because anything worse than that would be worse than guessing which would create incorrect results. I included shape = N to make sure it did it each time for each individual. For Z, I had a very similar approach with making the lower bound 0.5 and made the shape = M to make sure it went throught each PQ. 

I saw the D_reshaped and decided to do it for Z as well. This was just to make sure that the D values had one value per row, and the Z Had one value per column.

Finally, My p value, or the probability of a correct response was the exact same as the example but i used the Z_reshaped instead of just Z. 

My results:

My competence levels were not consistent, with P5 and P6 consistently getting the highest Competence score (usually around 0.9). On the other hand, P9 was the least competence, with it recieving a D of .58. 

My Z values, or consensus answers, came out as:
[0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1] for each participant.

AI Usage:
Chat GPT was used in this assignment to help me bring the models referenced into code. It also helped with formatting this README into a professional looking structure. 

