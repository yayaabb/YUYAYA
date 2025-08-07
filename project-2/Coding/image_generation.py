from diffusers import AutoPipelineForText2Image
import torch
import os
from huggingface_hub import login

login(token="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")

# Hugging Face Token
os.environ["HUGGINGFACE_TOKEN"] = "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"

# Load FLUX.1-dev main model
pipe = AutoPipelineForText2Image.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

# Loading LoRA: Ghibli Style
pipe.load_lora_weights(
    "aleksa-codes/flux-ghibsky-illustration", 
    weight_name="lora.safetensors"
)

# Use prompt
prompts = [
    "A Ghibli-style watercolor fantasy illustration of a couple of elderly men aged 60+ on a well-planned 3–5 day food journey through Croatia. They sit at a rustic wooden table inside a quiet seaside café in Dubrovnik, sharing a seafood platter while flipping through an art gallery brochure. One wears round glasses and a linen blazer, the other points toward a nearby exhibition poster hanging on a stone wall. Through the open window, pastel-painted buildings and the sparkling Adriatic Sea form a calming backdrop. A map and tablet rest beside a glass of chilled white wine. Painterly textures, warm coastal light, detailed interior shadows, and a peaceful, refined storybook mood full of culture, taste, and companionship.",
    "A Ghibli-style watercolor fantasy illustration of a small group of women aged 25–34 on a well-planned 6–14 day self-driving cultural journey through rural France. They quietly observe a group of wild deer at the edge of a forest clearing, their car parked nearby on a dirt path. One uses binoculars, another sketches in a field notebook, while the others sit on a picnic blanket with warm tea and a wildlife guidebook. Their clothing is practical yet stylish—windbreakers, hiking boots, and soft scarves. In the background, mist rises over golden meadows, and the outline of a distant stone abbey appears through the trees. Painterly textures, muted morning light, soft natural shadows, and a calm, immersive storybook mood full of curiosity, freedom, and subtle adventure.",
    "A Ghibli-style watercolor fantasy illustration of a large group of elderly men aged 45–60 on a go-with-the-flow 3–5 day luxury adventure through Switzerland. They gather with professional cameras on a high alpine ridge overlooking the Matterhorn, capturing photos as golden sunlight breaks through morning mist. Some kneel beside tripods adjusting lenses, others point toward the distance while laughing together in windproof jackets and woolen scarves. A sleek black van is parked nearby on a gravel road, and open map pages flutter on the dashboard. Behind them, snow-capped peaks, winding trails, and Swiss flags ripple in the breeze. Painterly textures, crisp mountain air, layered sunlight and shadow, and a joyful, cinematic storybook mood full of exploration, camaraderie, and iconic scenery.",
    "A Ghibli-style watercolor fantasy illustration of a small group of elderly travelers aged 60+ on a well-planned weekend self-driving food trip through the French countryside. They arrive at a rustic stone manor converted into a gourmet inn, parking their vintage car beneath tall sycamore trees. One holds a basket of local cheeses and wine, while the others admire the carved details of the historic façade. Ivy climbs the old walls, and sunlight filters through lace curtains in the arched windows. In the foreground, a table is set with pastries, cured meats, and fresh herbs. Painterly textures, golden afternoon glow, delicate architecture, and a serene, storybook mood full of heritage, flavor, and shared appreciation for beauty.",
    "A Ghibli-style watercolor fantasy illustration of a large group of women aged 35–44 on a spontaneous 6–14 day food journey across Spain. They gather at a bustling local market in Valencia, surrounded by colorful stalls of spices, citrus, olives, and fresh seafood. Some sample tapas at a wooden counter, others laugh while holding paper cones of churros dipped in chocolate. One jots down notes in a food diary, while another takes a selfie with a street chef flipping paella behind her. Their light dresses, sunhats, and sling bags catch the golden Mediterranean light. Painterly textures, layered shadows beneath canvas awnings, vibrant market atmosphere, and a joyful, flavorful storybook mood full of discovery, friendship, and taste."
]

# Save image
start_index = 66
for idx, prompt in enumerate(prompts):
    image = pipe(
        prompt=prompt,
        num_inference_steps=35,
        guidance_scale=8.0,
        generator=torch.manual_seed(100 + idx),
        width=352,
        height=528
    ).images[0]

    image.save(f"Mobileoutput/Mobileoutput_{start_index + idx}.png")
    print(f"Image {start_index + idx} generated successfully")
