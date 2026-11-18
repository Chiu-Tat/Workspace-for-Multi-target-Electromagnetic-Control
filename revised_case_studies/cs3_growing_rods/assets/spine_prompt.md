# Spine illustration provenance

Mode: built-in image_gen tool, transparent background.
Purpose: illustrative anatomical context in Figure 20(a), not patient imaging, a scanned mesh, or experimental geometry. The spine is not used by the magnetic or mechanical model.

## Initial generation prompt

Create a single anatomically credible 3D medical illustration asset of a human thoracolumbar spine in a direct posterior view, isolated on a genuinely transparent background. Show twelve thoracic vertebrae and five lumbar vertebrae, with the upper sacrum at the base, recognizable posterior vertebral arches, central spinous processes, paired transverse processes, and subtle grey intervertebral discs glimpsed between bones. The lumbar vertebrae should be larger than the thoracic vertebrae. No ribs, pelvis, skull, soft tissue, implants, rods, instruments, arrows, text, labels, grid or floor. Alignment is approximately straight in the posterior projection with natural anatomical detail, not scoliosis. Ivory bone with fine realistic cortical surface detail and subtle neutral-grey shadows, clear depth and soft studio lighting. A polished anatomical teaching-model rendering, not a patient photograph or scan. Tall narrow full-length composition with no cropped processes, centred, minimal transparent margins. Orthographic camera looking at the back, sacrum inferior, upper thoracic end superior. High visual clarity at small journal-figure size. Produce only the isolated spine; the two growing rods and labels will be added separately by reproducible plotting code.

## Final edit prompt

Edit this anatomical spine illustration asset. Preserve the visible ivory bones, discs, posterior viewpoint, arrangement and fine bone detail exactly. Remove all of the surrounding beige/grey glow, haze, vignette and cast shadow. Every pixel outside the anatomical silhouette must be completely transparent, including gaps between the lateral processes. Keep soft antialiasing only on the bone silhouette edges. Make a clean narrow cutout with minimal transparent margins, maintaining the natural width-to-height ratio of the spine. Do not introduce a background, labels, rods, implants, grid, floor or text. This asset will be composed over white by Python for a journal figure; no visible halo should remain.

The generated RGBA asset is preserved without bitmap modification. Matplotlib places its visible anatomical bounds into the panel, preserving aspect ratio. All rods, actuator boxes, annotations, and quantitative results remain source-authored.
