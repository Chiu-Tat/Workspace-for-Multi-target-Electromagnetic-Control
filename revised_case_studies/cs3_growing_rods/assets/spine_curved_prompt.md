# Curved spine illustration provenance

Mode: built-in image_gen tool, transparent-background edit.
Reference: spine_posterior_illustration.png, whose original generation prompts are saved in spine_prompt.md.
Final asset: spine_posterior_curved.png.
Purpose: lateral S-shaped anatomical context in Figure 20(a). The spine is not patient imaging or numerical model geometry. Parallel actuator centres and all simulation data remain unchanged.

## Final edit prompt

Edit the supplied posterior anatomical spine illustration. Change the previously straight vertebral column into a clearly visible but gentle lateral S-shaped curve appropriate for a conceptual scoliosis research illustration. Maintain a direct posterior orthographic view. Articulate the individual vertebrae smoothly so they follow the curve; do not merely tilt the entire spine. The thoracic midsection bends to image-right and the lumbar midsection compensates toward image-left, with the upper end and sacrum approximately centred. The maximum lateral deviation should be about one vertebral body's width: enough to be apparent in a small printed panel, not an extreme deformity. Preserve the realistic ivory cortical bone surface, posterior arches, spinous and transverse processes, subtle grey discs, and sacrum. Keep individual bones anatomically plausible, continuous and connected, with no stretched or melted vertebrae. Preserve the reference's polished 3D teaching-model lighting and detail. Show the entire spine with no cropped processes. Output only this isolated curved spine on a genuinely transparent background with clean alpha around the silhouette and between processes. No rods, implants, ribs, pelvis, text, arrows, annotation, labels, vignette, glow, floor or cast shadow. The asset will be placed between two separately drawn parallel growing-rod actuators by Python; it is illustrative context, not patient data.

The RGBA bitmap is stored without image modification. Matplotlib selects its visible bounds for composition, preserves its alpha and aspect ratio, and draws rods, anchors, actuator components, text and quantitative plots separately.
