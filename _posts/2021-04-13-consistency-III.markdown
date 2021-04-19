---
layout: post
title:  "Consistency training III: An Engineer's Guide to the Literature"
date:   2021-04-14
categories: consistency-training
---



You may have first seen consistency training presented alongside pretext tasks as a new and improved way of doing self-supervised learning (SSL) for computer vision. That’s just an accident of history. Pretext tasks have nothing to do with consistency training---they’re totally different.{% include note.html content="You can shoehorn the concepts together. Many do. Confusing! "%} Pretext task SSL is just supervised learning where your labels just happen to be derived from the data itself. In consistency training there are no labels, not even made-up ones. We’re not telling the model *how* to map inputs to outputs, we’re just telling it to map *in the same way across different views*. 

Pretext-tasks are straightforward: make up a task for your model to do, train it in a totally supervised fashion, then throw the task head away and hope you’ve created useful features. After you've figured out how to derive your labels, *this is just supervised learning*. Folks in olden times used to do this primarily with autoencoders, but the default pretext task nowadays is Imagenet classification.{% include note.html content="Pretraining on imagenet is still just a pretext task, albeit a supervised one. Sometimes you’ll see things like “SSL in computer vision can now surpass a supervised baseline” which actually means “self-supervised pretraining can sometimes beat supervised pretraining”. This is great, but it shouldn’t be that surprising---classification on a perfectly balanced dataset of strongly curated images may not be a relevant pretext task depending on the downstream application." %} Most computer vision research in SSL was pretext-task based until 2019.

![Pretext task SSL basic setup](/assets/img/rudy_machine_learning_diagrams_02_500x.png)

<span class="img_text"> Basic pretext task phenotype. Train in a supervised fashion on actual labels or ones you made up, throw the pretext-task head away and hope you've created useful features. Loss happens at the targets like in normal supervised learning. Pretext task head can be doing Imagenet classification, autoencoding, solving a jigsaw task or whatever. </span>

There are a bunch of pretext tasks to choose from. You can scramble frames in a video and put them back together, or try to generate the audio channel from the video one. You can change an image to black and white and try to recolor it. You can rotate the image and predict the rotation, or you can cut the image up into jigsaw pieces and make the model put the puzzle back together. You can make the model infill some patches you’ve blanked out or you can go all the way with an autoencoder and make the model reconstruct the entire input. {% include note.html content="Not including any references here. There are hundreds of papers out there that differ only in their choice of pretext task. Google them if you’re interested." %}

All of those pretext tasks will make your model react *differently* to different views of the *same* data. Different rotations, jigsaw configurations, crops, etc, of the same image will give you different features. But in reality---especially if you’re doing an object recognition task like classification---you probably want the model to respond to different views in the *same way*. This is the **primary motivation of invariance-based consistency training: we want our model to respond in the exactly the same way to different perturbations of the same input data**. If it's not clear why this produces useful features, check out our last post: [Consistency training III: How does it work?]({% post_url 2021-04-13-consistency-III %})

Not only does it feel intuitively correct that different views of the same image should produce similar features, we also see empirically that this just works better than pretext tasks. {% include ref.html key="pirl" text="Misra et al"%} and {% include ref.html key="cmc" text="Tian et al"%} both showed explicitly that consistency training beats pretext task training in a fair bake-off. It’s also just plain to see in the literature: the best SSL methods have been consistency-based since 2019. 

But some pretext tasks *also* do consistency training. The original work here is {% include ref.html key="exemplar" text="Exemplar." %} They use the pretext task of “instance discrimination”, where you treat each image in your original dataset as its own class. The model takes in differently augmented views of the data and has to map them to the original "exemplar" seed image. This makes the softmax layer as big as the original dataset itself. 

![Consistency training basic setup](/assets/img/exemplar.png)

This works. Mapping different views to the same seed image creates the invariances that we want. But imagine that monster softmax! It has an output neuron for *every image* in the dataset. Clearly not scalable.

We can just cut out the middleman. All we actually want is for different views of an image to produce the same features. That’s something we can optimize directly. We can implement our loss term *directly on the features.*

![Consistency training basic setup](/assets/img/rudy_machine_learning_diagrams_01_500x.png)

<span class="img_text"> Basic consistency training phenotype. Run differently augmented views of the same data through the model and train it to output the same features for all versions. </span>

Much more direct. We’re training exactly what we want, no pretext task apparatus to mess around with. Simple to understand, simple to code. And it actually works. This is the basic formulation of consistency training. All consistency methods follow this general format.

You can choose any loss function you want to nudge your features together. The simplest is `MSE`, used by {% include ref.html key="sajjadi" text="Transformation-Stability (TS)" %} and {% include ref.html key="laine_aila" text="Π-model (Horshoe)," %} among many others. {% include ref.html key="sajjadi" text="TS" %} is as simple as it gets: a single model, run two views of a single image together through the exact same model and enforce distance loss on the outputs. {% include ref.html key="laine_aila" text="Horseshoe" %}  adds a bit of asymmetry: instead of comparing features at the same point in time, they compare features from *this* epoch to features from the *previous* epoch. Same model, slightly different weights.

Instead of a previous model, you can take a weighted average of previous models. {% include note.html content="This is like the temporal ensembling we do for stability in supervised learning, or like what we do to smooth the bootstrapped updates from the Bellman equation in RL." %} Laine and Aila's {% include ref.html key="laine_aila" text="Temporal Ensembling model" %} takes a rolling average of predictions over time. The canonical work that we’ll see pop up most often, however, is {% include ref.html key="mean_teacher" text="Mean Teacher," %} which takes an exponentially-moving average (EMA) of *model weights* instead. You’ll see variants of this model everywhere: {% include ref.html key="french" text="French et al" %} would use it to win the VisDA competition in domain adapaptation. Xie et al would use a version of it in {% include ref.html key="noisy_student" text="Noisy Student" %} to hold SOTA on Imagenet for a few a while.

Most of the methods we’ll see in the rest of this post are variants of the two main families above: symmetrical like {% include ref.html key="sajjadi" text="TS"%} and {% include ref.html key="laine_aila" text="Horseshoe"%} where both lanes are identical, or asymmetric like {% include ref.html key="mean_teacher" text="Mean Teacher" %} and {% include ref.html key="laine_aila" text="Temporal Ensembling." %} These are two types of {% include ref.html key="bromley_1994" text="siamese network." %} The symmetrical ones are fully weight-tied, whereas {% include ref.html key="mean_teacher" text="Mean Teacher" %} and other EMA-variants are “loosely weight-tied” in the sense that their weights are similar but not identical

{% include ref.html key="sajjadi" text="TS," %} {% include ref.html key="laine_aila" text="Horshoe" %} and {% include ref.html key="mean_teacher" text="Mean Teacher" %} do `MSE` directly on the features. But there’s a small catch with this simple `MSE` loss: we’re exerting a “pull” with no corresponding “push”. There’s nothing stopping the features from collapsing into a single constant feature. Indeed, that would perfectly satisfy the `MSE` objective! {% include ref.html key="exemplar" text="Exemplar" %} solved this with the softmax, which pushes and pulls in equal measures.

<br/>

-----------------------------------------------------------

<br/>

Let's pause for a second and set the stage for the rest of this post.

We’ve already introduced the two main architectural themes used throughout consistency training: *symmetric* vs *asymmetric*. That’s not going to change. Consistency methods aren’t differentiated much by their setup. They’re all just siamese networks. 

What *does* differentiate the work in consistency training is how they address the two major design challenges below. We’ve already surfaced both of these questions:

**1) How do you avoid feature collapse in a scalable, memory-efficient way?** *E.g. Exemplar did this with a softmax.*

**2) How do you define different views of the data?** *E.g. Exemplar used crops, flips and color jitters.*

Most papers in consistency training can be summed up as a response to one of those two questions. Sometimes both. It seems like SSL has been a firehose of activity recently, but in actuality some of those papers---I’m just gonna say it---could have simply been tweets...

The majority of recent work addresses the first question of stability, which is mostly just a logistical problem (albeit a challenging one). The second question of defining views, on the other hand, is how we instill invariances in the model---that's the heart of consistency training.

<br/>

-----------------------------------------------------------

<br/>

## Avoiding feature collapse

Stable training may be just a means to an end, but that doesn’t make it easy. Like most of deep learning, it rarely "just works". But there are a few tricks you can use to make collapse or divergence less likely. 

<br/>

### Task signal

Most of the papers we introduced above are actually from the subfield of semi-supervised learning. They’re training a task objective at the same time as the `MSE` consistency loss. That makes the whole process a lot more stable. Features are less likely to collapse if they’re also being used for a supervised task.

But they can still collapse. I haven’t seen any work in semi-supervised learning that just jumps right into pure consistency training right off the bat. Most of them schedule it in only *after* training on the task objective for a while. This means they’ve already got a set of reasonable features before even bringing in the unlabeled data.

{% include ref.html key="laine_aila" text="Horeshoe," %} {% include ref.html key="laine_aila" text="Temporal Ensembling," %} {% include ref.html key="mean_teacher" text="Mean Teacher," %} {% include ref.html key="uda" text="Unsupervised Data Augmentation (UDA)" %} and others schedule the consistency training in gradually. {% include ref.html key="noisy_student" text="Noisy Student" %} and {% include ref.html key="meta_psuedo_labels" text="Meta Psuedo Label" %} go all the way, waiting until the teacher network is fully trained before doing any consistency training at all.

Another way these semi-supervised methods ease in the consistency signal is by only enforcing it on confident predictions. This forms a sort of curriculum. As the model gets better, it can enforce consistency on more points. As it enforces consistency on more points, the model gets better.

You may notice this sounds similar to psuedolabeling, where confident predictions are added back into the original dataset and treated as normal training data. Psuedolabelling is just consistency training where you round the softmax (i.e. do argmax to get the hard labels) and compare views across epochs like in {% include ref.html key="laine_aila" text="Horeshoe." %} A handful of methods do this, including {% include ref.html key="psuedolabel" text="Psuedolabel," %} {% include ref.html key="noisy_student" text="Noisy Student" %} and {% include ref.html key="meta_psuedo_labels" text="Meta Psuedo Label," %} but most methods use soft labels unless explicitly noted. Psuedolabelling / consistency training on the soft labels is the same as {% include ref.html key="knowledge_distillation" text="knowledge distillation," %} though used for a different purpose. 

{% include ref.html key="entmin" text="Entropy minimization (entmin)" %} operates on the same principle---low density separation---as consistency training / psuedolabelling but uses a slightly different mechanism. I mention it because it shows up as an additional performance enhancement in a lot of the semi-supervised work. When doing entmin you punish entropy on the unlabeled points, encouraging them to coalesce into groups, which hopefully pulls them back from the “front line” of the decision boundary.

<br/>

### Dispersal term

Let’s move into purely unsupervised territory. The main family of methods here addresses feature collapse by adding a dispersal term to its objective function. This acts as a countervailing “push” to the “pull” of the `MSE` on positives we saw above. Now we’ve got two components to our objective: something like a minimization of `MSE` that attracts positives together, and another term to push negatives away from each other. 

<br/>

#### Noise Contrastive Estimation

The first work in consistency training for deep learning came out of {% include ref.html key="bromley_1994" text="LeCun's" %} and {% include ref.html key="becker_1992" text="Hinton's"%} labs in the early nineties. They were both simple siamese setups---{% include ref.html key="bromley_1994" text="LeCun's work" %} introduced the term itself. They both used what we can consider to be contrastive losses, but I’ve never seen those forms used in modern work. {% include note.html content="The authors from Barlow Twins (more below) tried to get Hinton's to work, but to no avail." %} Most modern work in unsupervised consistency training that uses a contrastive loss uses some variant of the one from Radsel and Chopra in {% include ref.html key="chopra_2005" text="2005"%} and {% include ref.html key="hadsell_2006" text="2006," %} which is usually called something like **“noise contrastive estimation"** (`NCE`). The `NCE` loss is essentially `similarity_of_all_positive_pairs / similarity_of_all_negative_pairs`. Here's a slightly more detailed (but still highly stylized) psuedocode description of `NCE`.

{% highlight python %}
def NCE_loss(features_1, features_2)
    """ features_1 and features_2 are differently augmented views 
    of the same batch. Indices are aligned, e.g. features_1[0] 
    and features_2[0] are from the same original image. """

    batch_size, feature_dimension_size = features_1.shape

    # For all pairs of points
    numerator = 0
    denominator = 0
    for i, f1 in enumerate(features_1):
        for ii, f2 in enumerate(features_2):
            if i == ii: 
                # MAXIMIZE similarity of views from same img
                numerator += cos_similarity(f1, f2)
            else:       
                # MINIMIZE similarity of views from different imgs
                denominator += cos_similarity(f1, f2)
    
    return (numerator / denominator)
{% endhighlight %}

Note the two components of the objective. The numerator pulls different views from the same image closer. The denominator pushes views from different images apart. 

In our psuedocode above we looped through all our pairs of images. In reality, we'd compute a similarity matrix. We try to maximize the diagonal and minimize the rest.

![Consistency training basic setup](/assets/img/rudy_machine_learning_diagrams_03_500x.png)
<span class="img_text"> The NCE loss computes a similarity matrix between two differently-augmented views of the same underlying seed batch. We want views from the same seed image---"positives"---to be treated as identical, and views from different seed images---"negatives"---to be pushed apart.</span>

The tough question when implementing a `NCE`-like loss is where you get your negatives. The dispersal term in the denominator requires a lot of negatives examples to work well. I mentioned a firehose of research---a big portion of that is just focused on how to handle the negative examples!

To address {% include ref.html key="exemplar" text="Exemplar's," %} massive softmax, {% include ref.html key="wu_" text="Wu et al "%} keep the precomputed features from all of the images in an external memory-bank, drawing their negatives from there. {% include ref.html key="moco" text="MoCo"%} does the same thing but using an EMA-teacher (which they call a “momentum encoder”) rather than a symmetrical siamese setup. 

{% include ref.html key="simclr" text="SimCLR"%} got rid of the queue entirely, computing the `NCE` loss across a monster batch instead of a monster softmax. This work simplified the earlier `NCE`-based approaches, getting rid of the EMA-style asymmetry, the external queue, and the overengineered cropping of another earlier method, {% include ref.html key="cpc2" text="CPC." %} {% include ref.html key="simclr" text="SimCLR"%} is the canonical modern work in the SSL-`NCE` family.

The contrastive loss isn’t limited to computer vision, but it *is* especially helpful for high-dimensional data like images. Consider NLP as a counterexample of a {% include ref.html key="robot_brains_lecun" text="lower-dimension, easier domain." %} Masked LMs can simply do a vanilla softmax over the entire vocabulary. The “generative” in GPT-3 isn’t generative at all---they’re just predicting from a predefined set of words. `NCE` is computer vision’s attempt to take the "contrastive-ness" benefits of the softmax and bring them into the high-dimensional realm of images.

<br/>

#### Contrastive third party

In {% include ref.html key="exemplar" text="Exemplar"%} we were able to maintain stability by doing a contrastive loss across the whole dataset. We had a softmax layer with an output activation for each image in the seed dataset and we trained the model so that different augmentations of the same seed image would light up the same activation. The `NCE` family solved this with by replacing the static contrastive loss of the softmax with the dynamic contrastive loss of `NCE`. There’s another way we can do it that allows us to keep the original softmax. 

We don’t actually need one output activation for every seed image---all we care about is that views light up the *same* activation. Multiple seed images can share a single output activation. If we had a seed dataset of length 100k, for example, and we used a softmax of 1k, then each output activation would be shared by 100 seed images (assuming everything is balanced). These 100 seed images would all be using the same single activation simply as a fixed, third-party target against which to enforce consistency.

There is an intuitive way of looking at the reduced activation space: cluster assignments. You can imagine the seed images that share an output activation also share some sort of semantically meaningful content. This is a pretext task like {% include ref.html key="exemplar" text="Exemplar"%} but instead of *instance* discrimination we’re doing *cluster* discrimination.

![Consistency training basic setup](/assets/img/rudy_machine_learning_diagrams_04_500x.png)
![Consistency training basic setup](/assets/img/rudy_machine_learning_diagrams_05_500x.png)

{% include ref.html key="deepcluster" text="Deepcluster"%} from Caron et al is the canonical example here. They alternate between a clustering phase where they use k-means to assign a cluster id to each record in the dataset, and a training phase where the model learns to predict the cluster ids. They compare views across time like {% include ref.html key="laine_aila" text="Horseshoe," %} predicting *previous* cluster ids from *current* views.

{% include ref.html key="clusterfit" text="Clusterfit"%} and {% include ref.html key="noroozi_2018" text="Noroozi et al"%} do the same thing as {% include ref.html key="deepcluster" text="Deepcluster," %} but train on a pretext task before starting the clustering phase. This gives k-means more basis on which to cluster. {% include ref.html key="deepcluster" text="Deepcluster"%} had to bootstrap itself from scratch---the first round of clustering would have been on the features of a randomly initialized model. After pretraining on jigsaw or rotation, {% include ref.html key="clusterfit" text="Clusterfit"%} and {% include ref.html key="noroozi_2018" text="Noroozi et al"%} are essentially just {% include ref.html key="deepcluster" text="Deepcluster"%}

These three methods have the same problem as {% include ref.html key="laine_aila" text="Horseshoe" %} above: you have to wait an entire epoch to get your views for comparison. If your dataset is large, that means the model will have changed substantially. We need diversity of views for consistency training to work, but those views could be WAY different by the time the model gets around to the next iteration. We also don’t want our gradient update cycle to be coupled directly to the size of the dataset---how would that work with an *actually* massive dataset?

That’s the motivation for {% include ref.html key="swav" text="SwAV," %} the algorithm powering Facebook's recent {% include ref.html key="seer" text="SEER model," %} a 1.3 ***billion*** parameter model trained on one billion uncurated images from Instagram. {% include ref.html key="swav" text="SwAV" %} combines the clustering approach of {% include ref.html key="deepcluster" text="Deepcluster"%} with the batch-level updates of the `NCE`-like methods above. {% include ref.html key="swav" text="SwAV" %} is just {% include ref.html key="deepcluster" text="Deepcluster"%} where you compare differently augmented views at the same time (rather than across time) and at the batch level (rather than the dataset level). 

But get this: *the clusters don’t even have to be learned*. {% include ref.html key="swav" text="SwAV" %} works almost as well using random cluster centroids. They’re just using those “centroids” as a fixed intermediary yardstick against which to measure consistency. The whole point is just to ease the contrastive challenge by avoiding the pairwise comparisons between individual points---the locations of the cluster centroids don’t matter! {% include note.html content="The earlier clustering work never tests this, but the results from swav suggest that they would have gotten similar results by removing the k-means phase entirely and replacing the learned centroids with random targets." %}

{% include ref.html key="nat" text="Noise as Targets (NAT)"%} makes this explicit. They generate a set of targets randomly and use those as the goalposts for measuring consistency. If you set the number of targets to be the same as the size of the original dataset, that’s {% include ref.html key="exemplar" text="Exemplar." %} If you set it to less than that, it’s the clustering above but without the unnecessary centroid learning. 

<br/>

#### Explicit dispersal

The contrastive losses above are able to prevent collapse, but their mechanism of dispersal requires a comparison between *every pair of points*, or an extra third-party goalpost. All we want is to spread things out, so let’s just do that explicitly. 

{% include ref.html key="barlow" text="Barlow Twins"%} does batch-level consistency training like `NCE` but gets dispersion by trying to maximize the correlation between positives and minimize the correlation between negatives. Put another way, they encourage the model to turn the cross correlation matrix between two sets of views into the identity matrix (ones along the diagonal and zeros everywhere else). This is just like the `NCE` methods, but instead of a similarity matrix we've got a correlation matrix.

![Consistency training basic setup](/assets/img/rudy_machine_learning_diagrams_03_500x.png)
<span class="img_text"> Barlow Twins tries to maximize the correlation between positives and minimize it between negatives. Like NCE but correlation instead of similarity---I even reused the same sketch!</span>

TODO(this is still a lot of pairwise calculations. Do we understand this right?)

{% include ref.html key="barlow" text="Barlow Twins"%} encourages this decorrelation of negatives as a soft constraint, but you can also just do the decorrelation explicitly and manually like {% include ref.html key="whitening" text="Ermolov et al." %} This is called “whitening”. It’s like a supercharged batchnorm: instead of just normalizing along the batch dimension you also *decorrelate* along the batch dimension. This does explicitly what {% include ref.html key="barlow" text="Barlow Twins"%} trains indirectly. They pair this "negatives-scattering" effect with a `MSE` loss on the positives. 

Both of these methods are nice and simple. No need for massive batches. Gets right to the point. 

<br/>

### Stopgrad, asymmetry, and hidden dispersal

When we covered the semi-supervised methods above we largely presented it as “task signal prevents collapse”. The full story is more complicated. Things will get hand-wavy here because I don’t understand the details myself. 

The first confounding factor is that all of the semi-supervised methods are doing stopgrad. One of the siamese twins outputs fixed “targets” and another one optimizes towards those targets. They’re not passing gradients down both lanes at the same time like the `NCE` methods above. The student-teacher methods frame this explicitly as one network providing targets for another.

Implementing stopgrad is easy. Here's the psuedocode.
{% highlight python %}
# Grab two random augs of the same input batch
aug_1 = random_aug(batch)
aug_2 = random_aug(batch)

# Don't calculate gradients for one of them
with torch.no_grad():
    features_1 = model(aug_1)

# DO calculate gradients for the other
features_2 = model(aug_2)

# One-way stopgrad loss. 
# We could also do it the other way and add them together for a two-way loss.
loss = mse(features_1, features_2)
{% endhighlight %}

The second candidate is asymmetry at the model level. The EMA family, for example, has one model where the weights are a moving average of the other. The asymmetry itself seems helpful, but these models also have no choice but to also do stopgrad. The offline, EMA “teacher” lane never gets gradient updates, making it hard to disentangle the effects of stopgrad and model asymmetry. 

Three papers came out recently that are mostly just our familiar friends from above: {% include ref.html key="simsiam" text="SimSiam" %} and {% include ref.html key="directpred" text="DirectPred" %} are {% include ref.html key="sajjadi" text="TS" %} from the symmetric family. {% include ref.html key="byol" text="BYOL"%} is {% include ref.html key="mean_teacher" text="Mean Teacher." %} But they’re not *exactly* like those previous methods. They also add a bit of asymmetry in the form of an extra MLP head on whichever model is getting the gradient update. 

Those three papers seemed to show that the extra MLP and stopgrad are enough to stabilize training. No need for supervisory task signal. No need for an EMA teacher.{% include note.html content="BYOL initially claimed that the EMA lane was necessary because of how it stabilized the targets---same rationale as we’ve heard from the teacher student methods---but they later showed that it wasn’t actually critical, just helpful." %} But the MLP needs to have a higher learning rate. Why? I'm not sure. 

To add to the confusion, EMA on one of the lanes may not be necessary, but it seems to be helpful. The three papers above all found it to give a small but reliable boost. {% include ref.html key="byol" text="BYOL"%} added it to {% include ref.html key="simclr" text="SimCLR"%} and boosted their original performance. And there’s all the previous work in the mean teacher family that found the EMA twin to be important for SOTA performance. Even {% include ref.html key="moco" text="MoCo"%} probably gets a benefit from it, though they created their EMA teacher for a different reason. 

But why does changing one of the lanes to an EMA of the other help? Is it because the teacher is giving better, more stable targets as portrayed by the student-teacher methods? Or is it just because they’re *different*, which adds asymmetry in the same way as the mysterious pred head?

These questions also spill into the work on clustering above. You can’t pass gradients through the clustering phase---all of those papers are doing stopgrad implicitly. Is clustering just a complicated way of stopping gradients? {% include ref.html key="deepcluster" text="Deepcluster"%} was also getting extra view diversity by waiting an entire epoch to compare views. Was it silently benefiting from this added view asymmetry in the same way as {% include ref.html key="moco" text="MoCo"%} might have been?

The plot thickens further…

Everyone does batch norm, which as we mentioned above is dispersal in disguise! Does that matter? There’s no question that batch norm acts as a batch-level dispersal term (that’s literally what it does), but it’s an {% include ref.html key="untitled_ai" text="open question" %} whether or not that’s what’s providing the critical beam of support. It will probably turn out that normalization in general is important, but that the dispersal effect of batch-level normalization {% include ref.html key="byol2" text="may not be strictly necessary." %} Also note that {% include ref.html key="whitening" text="Ermolov et al"%} found batchnorm alone to be insufficient for preventing collapse, they had to also add decorrelation.

These questions may be keeping researchers up at night, but for us engineers things look rosy. We may not understand exactly why these simpler methods are more stable, but it’s nice that the best-performing methods are also the easiest ones to implement. {% include ref.html key="byol" text="BYOL," %} {% include ref.html key="simsiam" text="SimSiam," %} {% include ref.html key="directpred" text="DirectPred," %} {% include ref.html key="barlow" text="Barlow Twins" %} and {% include ref.html key="whitening" text="Ermolov et al" %} are all easier to code and train, *and* appear to be just as powerful as the more complicated methods above. 

<br/>

---------------------------------------------------------------

<br/>
<br/>

## Defining views

Everything we’ve done so far assumes we have enough distance between our views. If we fed the same exact image to the same network twice in a row we’d have no signal for consistency training. We’ve already seen that asymmetric model architecture can create diversity in our views, but it gets more interesting than that. 

There are two broad ways we can widen the distance between views even further: **(1)** we can perturb the model itself, like by using dropout, or **(2)** we can apply data augmentations to the inputs before they even hit the model. In reality, pretty much everyone uses dropout. All the action is in intentionally defining views through data preparation. 

<br/>

### Model noise

Even the architecturally-symmetrical siamese models above become *asymmetric* when we use dropout. A few “early-modern” consistency methods rely *solely* on this form of asymmetry to create different views.

The first in this line was {% include ref.html key="bachman_" text="Psuedo-ensemble," %} which phrases it as “taking different child networks from the main parent network” but appears to be essentially doing dropout. Laine and Aila's {% include ref.html key="laine_aila" text="Horseshoe and Temporal Ensembling models"%} use only dropout, as well.

{% include ref.html key="saito_2017" text="Adversarial Dropout Regularization" %} applies dropout-based consistency training to domain-adaptation in the same way as above, but instead of enforcing a statistical metric like `MSE` or `NCE`, they train an adversary to maximize the difference between views under different dropouts. They train this in the traditional GAN fashion, alternating between an encoder that tries to create dropout-invariant features and a discriminator that tries to differentiate between the views. {% include note.html content="This seems over-engineered, but then again folks in domain adaptation have found that learned distances (i.e. adversarial training) can be more powerful than statistical metrics (e.g. MMD and CORAL) for aligning domains, so maybe it works for views as well? Saito only benchmarks against other work within the narrow subfield of domain adaptation, so we don’t know how it performs against most of the other techniques we’ve reviewed so far. What we do know is that it was defeated by the simpler approach of French et al, which was just Tarvainen's Mean Teacher applied to domain adaptation" %}

<br/>

### Data augmentation

As we talked about in [the intro]({% post_url 2021-04-13-consistency-I %}), our entire motivation for consistency training is that we as domain-experts have a specific set of invariances we want to *intentionally* impart on the model. Defining our views through data augmentation or other transformations is precisely how we inject this supervision into the algorithm. We’re sketching out zones of invariance, and our pencil is data augmentation. Everything we did in the stability section above was just a vehicle for delivering the payload we care about: *consistency across views.*

#### Standard set of augmentations

Some authors do a good job of thinking through which priors they’re imposing on the model. But a lot of the literature glosses over the specific choice of augmentation as just an afterthought.

They generally implement some variant of the augs used by {% include ref.html key="simclr" text="SimCLR:" %} Take a random patch of the image, resize it to your target dimensions, do a random horizontal flip, do a color distortion phase composed of random combination of brightness, contrast, hue, saturation and maybe grayscaling, then finish up with random gaussian blur and maybe solarization. 

![Augmentations proposed by SimCLR](/assets/img/simclr_augs.jpg)

<span class="img_text">The set of augmentations tested by SimCLR. They found crop + color jitter to be the most effective.</span>

Some methods mix this up a bit but still essentially just do standard data augmentation. {% include ref.html key="swav" text="SwAV" %} takes five TODO crops at multiple resolutions rather than two at the same resolution, imparting a nice scale invariance to the model. {% include ref.html key="uda" text="UDA" %} does RandAugment. TODO {% include ref.html key="mixmatch" text="MixMatch"%} adds mixup. {% include ref.html key="fixmatch" text="FixMatch" %} does a weak augmentation (flip and shift) as a target, then trains towards that using strong augmentation (full set of augmentations), mimicking the benefits of a teacher network as a provider of stable targets. {% include ref.html key="pirl" text="PIRL" %} shuffles the image jigsaw-style to do a fair bake-off with old-school pretext task training. The original work in consistency training from {% include ref.html key="becker_1992" text="Becker and Hinton" %} simply takes neighboring crops.

Interestingly, {% include ref.html key="byol" text="BYOL" %} even brags that their setup is “robust to data aug”, which is essentially saying they don’t have fine-grained control over how their data augmentations create invariances in their features. 

The literature is able to treat the specific choice of data-aug as an afterthought because they’re mostly all focusing on the same set of object-recognition tasks. This isn’t a bad thing---it’s helpful to have common benchmarks---but it does mean that we need to take their perspectives with a grain of salt. Invariance across crops, for example, collapses all information relating to the global structure of an image. This particular invariance is great for Imagenet classification but it may not be what you want in your actual, real-life project. Like we saw with the hotdog example in [the intro]({% post_url 2021-04-13-consistency-I %}), you have to be very careful about what axes you choose to randomize---your model will become blind to them.

#### Getting creative

There are a few interesting exceptions to the vanilla data-aug crowd 

{% include ref.html key="vat" text="Virtual Adversarial Training (VAT)"%} learns the perturbation adversarially. Like normal adversarial training, it learns an augmentation that *most messes with* your objective, whereas vanilla data aug exerts the same “smoothing” pressure in all directions. This is a cool idea, but it essentially outsources the opportunity for extra supervision to an adversary. What’s preventing the adversary from making the image unrecognizable even to us as domain experts, instilling invariances we don’t want? Indeed, you have to explicitly reign in the amount of augmentation or the adversary will completely obscure the most relevant features of the input. 

![Consistency training basic setup](/assets/img/vat.jpg)

{% include ref.html key="uda" text="UDA" %} would later show that the standard suite of data augmentations works better than cleverly trying to learn the augmentations.

Another interesting example of view creation is {% include ref.html key="uda" text="UDA" %} applied to NLP. They do back translation to get a different view of the same underlying sentence. They outperformed previous work by a large margin using consistency training rather than a masked LM (which is contrastive but not consistency). GPT-3 would of course come in and dominate NLP with a traditional masked LM, but one wonders how a consistency approach like {% include ref.html key="uda" text="UDA" %} would perform if it too had 175 billion parameters?

{% include ref.html key="cpc" text="CPC" %} and its {% include ref.html key="cpc2" text="follow-up"%} proposed a more complicated version of cropping. They divide each image into a grid of patches, run them through the model separately, and predict patches *below* only using the context from *above*. {% include ref.html key="simclr" text="SimCLR" %} thankfully then showed that this was too clever by half---random cropping on the input image itself works better!

<br/>

### Different versions

Sometimes your dataset already has the multiple views you need. You don’t have to do augmentation here because you’ve already got natural groupings in the data. 

Face detection is a good example here. A positive set of views is all the pictures of the same person from different angles, in different lighting and in different settings. You can imagine Facebook does something like this to navigate their gigantic database of photos. There are more recent efforts on this front but they aren't conceptually different from the classic work out of{% include ref.html key="chopra_2005" text="LeCun's" %} and {% include ref.html key="nca" text="Hinton's" %} labs.

The {% include ref.html key="bromley_1994" text="original siamese work" %} does the same thing for signature verification. They trained an adorable model of a few thousand parameters on two hundred signatures from their colleagues at Bell Labs. This may have been a seminal work in consistency training but apparently folks weren’t taking it too seriously—”Mickey Mouse” showed up in the signatures at least once.

In face detection and signature verification, the invariant features *are* the deliverables themselves. You can use them directly to identify faces or to check signature fraud. No downstream task, just a nice set of reusable features that you can use for doing lookups. Like we mentioned above, the form of the contrastive loss means you can add new signatures or faces to your database as they come in, no fixed softmax. 

<br/>

### Different modalities

Instead of perturbing the same set of input properties in multiple ways, you could split your input properties into disjoint sets of features. This could be color channels in an image, different columns in tabular data, or entirely different modalities like audio, text or images. These are identical to the unimodal examples above, the only difference is in how they grab their views. Most use the `NCE` loss or something like it.

{% include ref.html key="cmc" text="CMC"%} is of the same lineage as the {% include ref.html key="simclr" text="SimCLR" %} crowd but uses different color channels for their different views. This is a fun idea, but it makes you wonder: what invariance were they hoping to impart? In a real project, for example, if you decide that color isn’t important for your task then you just collapse your image to grayscale as a preprocessing step. No need to go through all the trouble of setting up invariance training if you can just remove the variability manually. (TODO think about this more. Is there benefit?)

There’s a lot of work going under the banner audio-visual correspondence (AVC) that does cross-modal consistency training using the *audio* and *visual* channels of video. This includes {% include ref.html key="alwassel_2020" text="XDC," %} which is essentially a multimodal {% include ref.html key="deepcluster" text="Deepcluster." %} In addition to the consistency training, {% include ref.html key="objects_that_sound" text="Objects that Sound"%} splits the image into a grid of small patches and scores them separately. This allows them to predict which items in a given image are making a sound.

![Consistency training basic setup](/assets/img/objects_that_sound.png)

{% include ref.html key="clip" text="OpenAI's CLIP" %} is the most splashy and impressive from a long line of work doing multimodal consistency training between text and image. Like the student teacher methods above, they frame this as "text giving supervisory signal to images", but in reality they propagate gradients down both models. No stopgrad. Once trained, this lookup between text and images gives them the ability to create what they call a “zero-shot classifier”, which is just the dynamic lookup that a `NCE`-style contrastive loss gives you. Exactly as we'd hope, {% include ref.html key="clip" text="CLIP" %} develops [multi-modal neurons](https://distill.pub/2021/multimodal-neurons/), which, for example, light up in the say way for both the text *"Lady Gaga"* and for *images* of Lady Gaga herself. 


