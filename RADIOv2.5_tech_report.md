# RADIOv2.5 Tech Report

This is a tech report for the early-access release of the RADIOv2.5 model family. We plan on publishing full papers on the techniques applied for this release in upcoming conferences, but wanted to share the latest models with the community as soon as possible.

On 7.22.24 we are releasing ViT-B/16 and ViT-L/16 pretrained models. Under the hood, we've made a bunch of improvements to the training algorithms to produce these models. Fortunately, the API remains exactly the same!

Update: On 10.2.24 we are also releasing a ViT-H/16 model, which is now our best offering!

## Usage

```Python
torch.hub.load('NVlabs/RADIO', 'radio_model',
    version='radio_v2.5-h',  # Can also be 'radio_v2.5-b' for the ViT-B version, 'radio_v2.5-l' for the ViT-L version
    force_reload=True,  # Make sure you set this to True the first time you're requesting either of these two models
)
```

## What's New?

### Smaller Models

First off, our previous releases have been ViT-H models. While ViT-H is a very powerful model architecture, we've heard from the community that there is a need for smaller VFMs (Visual Foundation Model). With this release, we're releasing ViT-B/16 and ViT-L/16 models which still achieve very strong quality, while being much smaller and faster than ViT-H. In fact, we're so confident in our ViT-L/16 model (RADIOv2.5-L) that we think you should use it instead of RADIOv2.

### Mode Switching

A major issue we identified in the paper is that RADIO was "mode switching" based on the input resolution of the image. In effect, when the resolution was approximately less than 704, it was running in "CLIP + DINOv2" mode where the features were very relevant to those two teachers, but completely irrelevant to SAM. At >720px, RADIO was switching modes to produce features that were relevant for SAM, but suddenly incapable of modeling CLIP or DINOv2.

This would show up in strange ways, for example, trying to do zero shot classification at high resolution would degrade to random guessing (0.1% for ImageNet-1k). It also meant that our results at hi-res when integrated into a VLLM (e.g. LLaVA 1.5 / Vicuna 7B) were similarly poor. Starting with RADIOv2.5, we've solved the mode switching problem, and now these models are truly capable of processing any input resolution without surprising changes in behavior. In fact, RADIOv2.5 _loves_ high resolution, where our best classification and VLLM results are coming from >= 768px resolutions.

<div align="center">
    <img src="assets/radio_v2.5/mode_switching_dv2.png", width="512"/>
</div>

Similar to the paper, we plot the MSE between the DINOv2-g-reg features and the RADIO model at various resolutions. While RADIOv2 (owing to the ViT-H) is able to achieve lower MSE at lower resolutions, you can see how at 720px, there's a huge spike in error and never recovers. This is how we quantified the mode switch. We can also visualize this phenomenon:

<div align="center">
    <img src="assets/radio_v2.5/mode_switching/radio-v2.1_vis-24.jpg", width="512"/>
    <img src="assets/radio_v2.5/mode_switching/radio-v2.5-l_vis-24.jpg", width="512"/>
    <img src="assets/radio_v2.5/mode_switching/radio-v2.5-h_vis-24.jpg", width="512"/>
</div>

You can see how the 720px RADIOv2 (left) image abruptly changes representions, whereas DINOv2 and the RADIOv2.5 models (middle, right) remain consistent and instead produce increasingly fine-grained details. We can also see how RADIOv2 is working in reverse with the SAM head, where the low-resolution inputs don't produce features that are SAM-like at all. At 1024px, RADIOv2 starts to produce reasonable SAM features. On the contrary, RADIOv2.5-L produces SAM-like features at any resolution, and arguably does a better job of extrapolating to 2048px resolution.

<div align="center">
    <img src="assets/radio_v2.5/sam_mode_switching/radio-v2_vis-23.jpg", width="768"/>
    <img src="assets/radio_v2.5/sam_mode_switching/radio-v2.5-h_vis-23.jpg", width="768"/>
</div>

Similarly to mode switching being directly observable in the spatial features, it was also causing issues with the summary features, which can be seen looking at zero shot classification:

<div align="center" style="display: flex;">

<div align="right" style="flex: 1; padding-right: 20px;">
    <img src="assets/radio_v2.5/classification_mode_switch.png", width="512"/>
</div>

<div align="left" style="flex: 1;">

| Resolution       | RADIOv2.1 | RADIOv2.5-B | RADIOv2.5-L | RADIOv2.5-H |
|------------------|-----------|-------------|-------------|-------------|
| 224              | 78.892    | 62.344      | 74.852      | 79.158      |
| 256              | 80.780    | 68.892      | 78.220      | 80.156      |
| 336              | 82.320    | 72.626      | 80.004      | 81.426      |
| 432              | 82.800    | 73.628      | 80.460      | 81.944      |
| 512              | 82.882    | 73.894      | 80.542      | 82.162      |
| 768              | 1.292     | 74.386      | 80.804      | 82.088      |
| 1024             | 0.204     | 74.280      | 80.886      | 82.304      |

</div>

<div>

| Resolution       | RADIOv2.1 | RADIOv2.5-B | RADIOv2.5-L | RADIOv2.5-H |
|------------------|-----------|-------------|-------------|-------------|
| 512 - ViTDet 16  | 82.370    | 70.488      | 78.102      | 80.058      |
| 1024 - ViTDet 16 | 0.192     | 72.182      | 79.878      | 81.834      |

</div>

</div>


Not only do the RADIOv2.5 models allow classification at any resolution, they also allow for using ViTDet mode with only a small drop in accuracy.

There is an important implication to fixing mode switching, which is that it's now possible to ask for both the CLIP and SAM features for a given hi-res image simultaneously, and the results will be meaningful for both. Or, you might want to get the hi-res DINOv2 spatial features as well as the summary token (for classification) for the same image. This wasn't possible with the RADIOv2 model because it wasn't able to simultaneously represent CLIP (or DINO) and SAM at the same time, but is now fixed with the v2.5 models.

#### LLaVA 1.5 + Vicuna 7B

Last but not least, we tested out the models at various resolutions within LLaVA 1.5 + Vicuna 7B:

<div align="center">
<table>
    <tr>
        <td rowspan="2" align="center">Model</td>
        <td rowspan="2" align="center">Resolution</td>
        <td colspan="2" align="center">GQA</td>
        <td colspan="2" align="center">TextVQA*</td>
        <td rowspan="2" align="center">POPE</td>
        <td rowspan="2" align="center">VQAv2</td>
    </tr>
    <tr>
        <td>Val</td>
        <td>TestDev</td>
        <td>Tokens</td>
        <td>No Tokens</td>
    </tr>
    <tr>
        <td>RADIOv2.1</td><td>432</td><td>71.70</td><td>63.01</td><td>56.32</td><td>42.03</td><td>86.20</td><td>79.28</td>
    </tr>
    <tr/>
    <tr>
        <td rowspan="3">RADIOv2.5-B</td><td>432</td><td>70.49</td><td>62.09</td><td>52.13</td><td>32.43</td><td>85.87</td><td>77.24</td>
    </tr>
    <tr>
                                        <td>512</td><td>71.08</td><td>62.70</td><td>54.36</td><td>36.39</td><td>86.59</td><td>78.03</td>
    </tr>
    <tr>
                                        <td>768</td><td>71.99</td><td>63.31</td><td>56.93</td><td>43.96</td><td>87.54</td><td>79.22</td>
    </tr>
    <tr/>
    <tr>
        <td rowspan="3">RADIOv2.5-L</td><td>432</td><td>71.57</td><td>62.89</td><td>56.71</td><td>42.34</td><td>86.13</td><td>79.44</td>
    </tr>
    <tr>
                                        <td>512</td><td>72.04</td><td>63.58</td><td>58.52</td><td>46.50</td><td>86.66</td><td>80.04</td>
    </tr>
    <tr>
                                        <td>768</td><td>72.91</td><td>64.13</td><td>61.93</td><td>53.95</td><td><b>87.68</b></td><td>81.02</td>
    </tr>
    <tr>
        <td rowspan="3">RADIOv2.5-H</td><td>432</td><td>73.22</td><td>64.91</td><td>58.66</td><td>47.61</td><td>85.95</td><td>80.49</td>
    </tr>
    <tr>
                                        <td>512</td><td>73.60</td><td>64.98</td><td>60.03</td><td>51.99</td><td>86.73</td><td>80.96</td>
    </tr>
    <tr>
        <td>768</td><td><b>74.04</b></td><td><b>65.03</b></td><td><b>62.39</b></td><td><b>56.93</b></td><td>87.36</td><td><b>81.56</b></td>
    </tr>
</table>

*By default, TextVQA adds detected OCR tokens into the context of the LLM. Because we're interested in how well the vision encoder itself is able to represent text, we study TextVQA both with (Tokens) and without (No Tokens) these tokens.

</div>


### SigLIP

SigLIP is an extraordinary ViT-L model, and we've added it as a teacher in the latest release. If you'd like to use RADIO's adaptor for it, you can get it using the `'siglip'` adaptor name. For example, in the `examples/zero_shot_imagenet.py` script, you'd pass `--adaptor-name siglip` as an argument to use SigLIP instead of the default DFN CLIP.

The specific SigLIP version we're using is `ViT-SO400M-14-SigLIP-384` found in the [OpenCLIP](https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv#L5) library.

<div align="center">

| Resolution | RADIOv2.5-B | RADIOv2.5-L | RADIOv2.5-H |
|------------|-------------|-------------|-------------|
| 224        | 58.670      | 72.492      | 76.796      |
| 256        | 65.190      | 75.962      | 77.984      |
| 336        | 69.110      | 77.830      | 79.400      |
| 432        | 70.276      | 78.582      | 79.990      |
| 512        | 70.694      | 78.828      | 80.258      |
| 768        | 71.102      | 78.930      | 80.172      |
| 1024       | 70.900      | 78.922      | 80.476      |

</div>

As can be seen, the classification results using the SigLIP head are slightly worse than those of DFN CLIP, so we'd suggest defaulting to DFN CLIP unless you're specifically looking for compatibility.


### Videos!

While RADIOv2 may have a visually pleasing 1024px resolution video, you can clearly see how it switches modes between low and high resolution. All models exhibit strong temporal stability.

<details>
<summary>RADIOv2</summary>

#### RADIOv2 512px

https://github.com/user-attachments/assets/fbde8f2d-0f89-499a-b460-62810595ba55

#### RADIOv2 1024px

https://github.com/user-attachments/assets/4abdd040-c837-4e7a-87ac-e9bc7d40751b

</details>

<details>
<summary>RADIOv2.5-B</summary>

#### RADIOv2.5-B 512px

https://github.com/user-attachments/assets/c9871a93-7260-4386-b334-12bf7790a8e4

#### RADIOv2.5-B 1024px

https://github.com/user-attachments/assets/14c415d4-0962-4937-a087-bf01e1511881

</details>

<details>
<summary>RADIOv2.5-L</summary>

#### RADIOv2.5-L 512px

https://github.com/user-attachments/assets/024a23ca-ec27-44dc-8cb2-47bdd1b15635

#### RADIOv2.5-L 1024px

https://github.com/user-attachments/assets/1dd4e2ef-a47a-4121-8462-d90e087fdab6

</details>

<details>
<summary>RADIOv2.5-H</summary>

#### RADIOv2.5-H 512px


#### RADIOv2.5-H 1024px


</details>
