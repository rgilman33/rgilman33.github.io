---
layout: post
title:  "Consistency training I: Intro"
description: "An intuitive guide to consistency training (i.e. contrastive learning, energy-based modelling, invariance training)"
date:   2021-04-14
categories: consistency-training
---

Consistency training is a slippery concept. All the subfields of ML use it, but they each call it something different. Researchers have been doing it since the 19th century, yet they still rediscover it on a yearly basis. Once you recognize it you’ll see it everywhere.

It may be hard to pin down when folks are doing consistency training, but it’s worth getting the concept right: it’s powering the current [revolution in self-supervised learning](https://towardsdatascience.com/the-quiet-semi-supervised-revolution-edec1e9ad8c){:target="_blank"} for computer vision. It’s how [OpenAI’s CLIP](https://openai.com/blog/clip/){:target="_blank"} learns a zero-shot classifier. It’s the best way we know of doing domain-adaptation. It’s what makes dropout and data augmentation work.

Self-supervised learning (SSL) has been doing it in the form of “contrastive learning”. But “contrastive” can mean a lot of things---any softmax is contrastive. Now it’s popping up as “non-contrastive SSL”, but that could include pretext tasks. I’ve also seen it called “invariant representation learning” or “energy-based modelling”. Some people just describe it in terms of what it’s doing: “smoothing the space around a point”, “attaining low density separation”, “propagating labels”, or “pushing away decision boundaries”. 

The folks in semi-supervised learning call it “consistency training”. This is descriptive and accurate. It’s general enough to cover all its instantiations but narrow enough to actually have a meaning. It’s what we’ll use here.

<br/>

------------------------------------------
<br/>

### What is consistency training? 

You’re the technical lead of the Oscar Meyer data science team. The CEO comes to your office with an urgent assignment: they need a model that recognizes hotdogs in natural images. Binary classifier, hotdog or not-hotdog. 

Your team’s first classifier performs well but it isn’t perfect. It’s overfitting to the small dataset. You decide to implement data augmentation. 

What set of augmentations do you choose? You’ve got the menu of standard transformations at your disposal: crop, color jitter, rotation, blur, solarization, affine transform---whatever you want. You’ll want to be as aggressive as possible to counteract the small size of your dataset. Pause here, think about what augmentations you’d do. 

You probably imagined different levels of augmentation that look something like this:

![Hotdogs at different levels of data augmentation](/assets/img/hotdogs.png)

<span class="img_text"> Different options for data augmentation. We want enough to fill out the dataset but not so much that even we as domain experts can't recognize it. The original image is on the left, augmentation increases as we go right. The rightmost image is too augmented---even we can't tell it's a hotdog.</span>

This augmentation allows you to train a model that’s more robust. How is that? 

**Obfuscating unhelpful variability.** The model will become blind to whichever axes you randomize. Jitter the hue and blur the image? The model becomes blind to color and texture. Crop and rotate? The model becomes blind to global structure. It's like a game of twenty-questions: every constraint you impose is another hint, narrowing down the set of valid hypotheses. Telling the model that *"whatever solution you come up with shouldn't depend on global structure, texture or color"* rules out a GIGANTIC portion of the search space. 

**Highlighting the signal.** Not only does data augmentation collapse unhelpful variation, it also highlights the useful stuff. You're giving the model another hint: *"the solution should depend on whatever is constant throughout these data augmentations"*.

You can see why consistency training is often framed in terms of mutual information: given a single image with multiple augmentations (like the hotdog above), the solution to the task must be in the venn-diagram overlap of two zones: (1) solutions that *don't* rely on whatever axes we're randomizing (color, texture, global structure) and (2) solutions that *do* rely on whatever is constant throughout the augmented images (the hotdog). 

This isn't some niche ML thing, humans also learn best through examples. To teach a child the concept of an "apple" you don't describe it in some abstract way and hope they recognize one when they see it, you just show them a few examples---granny smith, honeycrisp, golden delicious---and trust that their brain will sift through the noise to find the thing that's constant across examples. 

<br/>

--------------------------------------------------------------

<br/>

You’re about to deliver the final model when you get an urgent email from the CEO: now they need a hotdog *brand* detector! Multiclass classifier. Needs to differentiate between the twelve most popular brands of hotdog by *hotdog alone*---can’t even rely on packaging! That changes things. 

The model architecture can mostly stay the same.  What about data augmentation? Does this change the set of perturbations you’re comfortable doing? Think about it for a moment.

Oscar Meyer didn’t make you head of data science for nothing! You quickly realize that you’ll have to be much more conservative with the data perturbations. You can’t willy-nilly collapse all the intra-hotdog variability like you did before. Those subtle differences in color, shape and texture are exactly what will help the model distinguish between brands. They weren’t important when your job was just to identify the presence of a hotdog, but now they’re critical differentiators---even though the data itself is the same. **The validity of a set of augmentations depends on the task.**

Let’s appreciate what you just did. You combined your human priors about the natural world with your domain expertise in hotdogs, then you refined that knowledge for the *specific task* and injected it into the algorithm in the form of thoughtfully-crafted data augmentation. **Data augmentation is a form of supervision.**

**But that form of supervision has *nothing to do with labels*. The fact that you as domain-expert are able to say that brand identification is invariant to a particular set of perturbation is an item of knowledge that applies to *all of your records*. This "learning by example", mutual-information channel of supervision will benefit even from an increase in *unlabeled samples*.**

And you have a ton of those. Oscar Meyer has terabytes of unlabeled hotdog footage just sitting in the basement! Surely there’s a wealth of structure just waiting to be uncovered! You can use the intuition you had above about class consistency across views---*even when you don’t know the class*.

But how do you operationalize that intuition? How can you actually use the natural structure of all that unlabeled data? How can you spread that supervisory signal across your entire dataset, imbuing the right invariances even in your unlabeled records?

-----------------------------------------------------------------------------

[Consistency training II: How does it work?]({% post_url 2021-04-13-consistency-III %}) is a walkthrough of what’s happening under the hood.

[Consistency training III: An engineer's guide to the literature]({% post_url 2021-04-13-consistency-II %}) reviews the landscape of existing work on consistency training.

[Consistency training IV: Beyond invariance]({% post_url 2021-04-13-consistency-IV %}) shows that invariances are just one type of consistency. There’s a whole world of constraints we can enforce to reduce our hypothesis space enough to solve simple tasks with *no labels at all*.

----------------------------------------------------------
