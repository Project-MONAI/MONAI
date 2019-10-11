**Project MONAI** (**M**edical **O**pen **N**etwork for **AI**)

_AI Toolkit for Healthcare Imaging_

_Contact: monai.miccai2019@gmail.com_

This document identifies key concepts of project MONAI at a high level, the goal is to facilitate further technical discussions of requirements,roadmap, feasibility and trade-offs.


# Vision
   *   Develop a community of academic, industrial and clinical researchers collaborating and working on a common foundation of standardized tools.
   *   Create a state-of-the-art, end-to-end training toolkit for healthcare imaging.
   *   Provide academic and industrial researchers with the optimized and standardized way to create and evaluate models

# Targeted users
   *   Primarily focused on the healthcare researchers who develop DL models for medical imaging

# Goals
   *   Deliver domain-specific workflow capabilities 
   *   Address the end-end “Pain points” when creating medical imaging deep learning workflows.
   *   Provide a robust foundation with a performance optimized system software stack that allows researchers to focus on the research and not worry about software development principles.

# Guiding principles
1. Modularity
   *   Pythonic -- object oriented components
   *   Compositional -- can combine components to create workflows
   *   Extensible -- easy to create new components and extend existing components
   *   Easy to debug -- loosely coupled, easy to follow code (e.g. in eager or graph mode)
   *   Flexible -- interfaces for easy integration of external modules
2. User friendly
   *   Portable -- use components/workflows via Python “import”
   *   Run well-known baseline workflows in a few commands
   *   Access to the well-known public datasets in a few lines of code
3. Standardisation
   *   Unified/consistent component APIs with documentation specifications
   *   Unified/consistent data and model formats, compatible with other existing standards
4. High quality
   *   Consistent coding style - extensive documentation - tutorials - contributors’ guidelines
   *   Reproducibility -- e.g. system-specific deterministic training
5. Future proof
   *   Task scalability -- both in datasets and computational resources
   *   Support for advanced data structures -- e.g. graphs/structured text documents
6. Leverage existing high-quality software packages whenever possible
   *   E.g. low-level medical image format reader, image preprocessing with external packages
   *   Rigorous risk analysis of choice of foundational software dependencies
7. Compatible with external software
   *   E.g. data visualisation, experiments tracking, management, orchestration

# Key capabilities

<table>
  <tr>
   <td>
<strong><em>Basic features</em></strong>
   </td>
   <td colspan="2" ><em>Example</em>
   </td>
   <td><em>Notes</em>
   </td>
  </tr>
  <tr>
   <td>Ready-to-use workflows
   </td>
   <td colspan="2" >Volumetric image segmentation
   </td>
   <td>“Bring your own dataset”
   </td>
  </tr>
  <tr>
   <td>Baseline/reference network architectures
   </td>
   <td colspan="2" >Provide an option to use “U-Net”
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Intuitive command-line interfaces
   </td>
   <td colspan="2" >
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Multi-gpu training
   </td>
   <td colspan="2" >Configure the workflow to run data parallel training
   </td>
   <td>
   </td>
  </tr>
</table>



<table>
  <tr>
   <td><strong><em>Customisable Python interfaces</em></strong>
   </td>
   <td colspan="2" ><em>Example</em>
   </td>
   <td><em>Notes</em>
   </td>
  </tr>
  <tr>
   <td>Training/validation strategies
   </td>
   <td colspan="2" >Schedule a strategy of alternating between generator and discriminator model training
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Network architectures
   </td>
   <td colspan="2" >Define new networks w/ the recent “Squeeze-and-Excitation” blocks
   </td>
   <td>“Bring your own model”
   </td>
  </tr>
  <tr>
   <td>Data preprocessors
   </td>
   <td colspan="2" >Define a new reader to read training data from a database system
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Adaptive training schedule
   </td>
   <td colspan="2" >Stop training when the loss becomes “NaN”
   </td>
   <td>“Callbacks”
   </td>
  </tr>
  <tr>
   <td>Configuration-driven workflow assembly
   </td>
   <td colspan="2" >Making workflow instances from configuration file 
   </td>
   <td>Convenient for managing hyperparameters
   </td>
  </tr>
</table>



<table>
  <tr>
   <td><strong><em>Model sharing & transfer learning</em></strong>
   </td>
   <td colspan="2" ><em>Example</em>
   </td>
   <td><em>Notes</em>
   </td>
  </tr>
  <tr>
   <td>Sharing model parameters, hyperparameter configurations
   </td>
   <td colspan="2" >Standardisation of model archiving format
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Model optimisation for deployment
   </td>
   <td colspan="2" >
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Fine-tuning from pre-trained models
   </td>
   <td colspan="2" >Model compression, TensorRT
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Model interpretability
   </td>
   <td colspan="2" >Visualising feature maps of a trained model
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Experiment tracking & management
   </td>
   <td colspan="2" >
   </td>
   <td><a href="https://polyaxon.com/">https://polyaxon.com/</a>
   </td>
  </tr>
</table>



<table>
  <tr>
   <td><strong><em>Advanced features</em></strong>
   </td>
   <td colspan="2" ><em>Example</em>
   </td>
   <td><em>Notes</em>
   </td>
  </tr>
  <tr>
   <td>Compatibility with external toolkits 
   </td>
   <td colspan="2" >XNAT as data source, ITK as preprocessor
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Advanced learning strategies
   </td>
   <td colspan="2" >Semi-supervised, active learning
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>High performance preprocessors
   </td>
   <td colspan="2" >Smart caching, multi-process
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Multi-node distributed training
   </td>
   <td colspan="2" >
   </td>
   <td>
   </td>
  </tr>
</table>



*   Project licensing: Apache License, Version 2.0
