
import streamlit as st
from google.api_core.client_options import ClientOptions
from google.cloud import aiplatform
import vertexai
import vertexai.generative_models # for Gemini Models

# Configuration
PROJECT_ID = "mg-ce-demos"
REGION = "us-central1"

# Gemini Models
#gemini15_multimodal = vertexai.generative_models.GenerativeModel("gemini-1.5-pro-002")
gemini_flash = vertexai.generative_models.GenerativeModel("gemini-2.0-flash-001")

# Page config
st.set_page_config(
    page_title="Text-to-Video Query Rewriter",
    page_icon="🎬",
    layout="wide"
)

# Header
st.markdown("<h1 class='title'>Text-to-Video Query Rewriter</h1>", unsafe_allow_html=True)

def generate_response_stream(input_text):
    rewriter_preamble = """Generate a high quality rewrite of USER_QUERY for a text-to-video service. The rewrite adds details to greatly improve the visual quality and motion of the video, but does not change the user's intent.

    Refrain from adding children or minors to the rewrite if not necessary to satisfy the USER_QUERY.
    
    Consider extra details to enhance creativity. Consider adding visual details IF it would support the user query:
    - camera angle and composition: wide angle, drone camera, low angle view, closeup, macro, view from below looking up, centered, fisheye
    - lighting: silhouette, backlit, dim ambient lighting, long shadows, natural light, sunrise / sunset, daylight
    - camera settings and motion: depth of field, in focus, long exposure, tracking shot, POV
    - general quality identifiers: professional, award winning, high-quality
    - styles: cinematic shot, street photography, fashion photography, architectural photography, dramatic, vintage, retro
    - background: blurred background, bokeh, pink background, solid light blue background
    - color scheme: high contrast, cold muted tones, muted orange warm tones, dark tones, pastel colors
    - subject actions: walking, running, turning head
    
    Feel free to repeat the most important parts of the description! If you can't interpret the query as a plausible video, consider it as text and specify the details how and where it is written.
    
    Remember, it is important to include every word or a synonym from the USER_QUERY. Never remove any details from the USER QUERY, including mediums and styles.
    
    If USER_QUERY is long and detailed, either 1) add minor details in the variations, or 2) copy the USER_QUERY and only correct typos or misspellings.
    
    Absolutely make sure that EVERY detail of the USER_QUERY is well captured in each variation.
    Consider emphasizing the features of the USER_QUERY so that the video is rendered faithfully to the USER_QUERY.
    
    Please follow this style of text prompt, each line is a different prompt example:
    
    This close-up shot follows a happy queen as she ascends the steps of a candlelit throne room. The warm glow of the candlelight illuminates her regal bearing and the intricate details of her jeweled crown, the light dancing on the jewels as she moves. She turns her head, the happiness in her eyes becoming more prominent. The background blurs as she continues her ascent, the tapestries and gilded furniture a testament to her power and authority.
    
    Close-up portrait of a Black woman dancing in a vibrant carnival in Trinidad and Tobago. The energetic scene captures the infectious rhythm of the music and the exuberant spirit of the celebration. Colorful lights illuminate her face, highlighting her joyful expression and the graceful movement of her body. Her eyes, a sparkling brown, radiate pure happiness and the unbridled passion of Caribbean culture.
    
    Cinematic shot of a Caucasian man dressed in a weathered green trench coat, bathed in the eerie glow of a green neon sign. He leans against a gritty brick wall with a payphone, clutching a black rotary phone to his ear, his face etched with a mixture of urgency and desperation. The shallow depth of field focuses sharply on his furrowed brow and the tension in his jaw, while the background street scene blurs into a sea of neon colors and indistinct shadows.
    
    This underwater film scene features a close-up of a man in a dark business suit swimming through murky water. The video is captured in motion blur, with the man's limbs and suit jacket trailing behind him in swirling eddies. His expression is one of intense focus, eyes wide and mouth slightly open as he navigates the depths. The muted light filtering through the water casts eerie shadows and highlights the texture of his suit fabric. The overall mood is one of suspense and urgency, as if the man is on a desperate mission with time running out.
    
    Close-up shot of a quick cat briskly walking in the park, it’s crafted entirely of glass, illuminated by dramatic lighting. Each facet of its form glints and reflects, from the delicate whiskers to the curve of its tail. Its paws, though seemingly fragile, press firmly against the surface with each stride. The cat's translucent body allows the light to pass through, creating an ethereal glow that highlights its elegance and poise. The background is a deep, rich color, allowing the cat to stand out as the main focal point of the video.
    
    Cinematic shot of a lone surfer's silhouette, walking on a vast beach with surfboard in hand. The dramatic sunset paints the sky in vibrant hues of purple and red, casting long shadows across the sand. The sun dips below the horizon, leaving a fiery glow that illuminates the figure and the crashing waves. The wide shot captures the vastness of the scene, emphasizing the surfer's solitude and the awe-inspiring beauty of nature.
    
    Extreme close-up of a woman's eyes, bathed in the vibrant glow of neon lights. The camera focuses on the intricate details of her iris, a mesmerizing blend of blues, greens, and golds. Her long, dark lashes cast delicate shadows on her skin, and a single tear glistens at the corner of her eye. The woman's gaze is both alluring and mysterious, inviting the viewer to explore the depths of her emotions. The neon lights reflect in her pupils, creating a kaleidoscope of colors that dance and shimmer with each blink. The overall effect is one of intense beauty and raw vulnerability, capturing the essence of the human spirit in a single, captivating frame.
    
    A close-up shot of a man made entirely of glass riding the New York City subway. Sunlight refracts through his translucent form, casting a rainbow of colors on the nearby seats. His expression is serene, his eyes fixed on the passing cityscape reflected in the subway window. The other passengers, a mix of ages and ethnicities, sit perfectly still, their eyes wide with a mixture of fascination and fear. The carriage is silent, the only sound is the rhythmic clickety-clack of the train on the tracks.
    
    Close-up cinematic shot of an Indian man in a crisp white suit, bathed in the warm glow of an orange neon sign. He sits at a dimly lit bar, swirling a glass of amber liquid, his face a mask of quiet contemplation and hidden sorrow. The shallow depth of field draws attention to the weariness in his eyes and the lines etched around his mouth, while the bar's interior fades into a soft bokeh of orange neon and polished wood.
    
    A cinematic close-up frames the face of a young Asian woman in the heart of Tokyo's Shibuya Crossing. The neon glow of the cityscape illuminates her delicate features, highlighting the soft blush on her cheeks. Gentle lighting accentuates her bright, inquisitive eyes, reflecting the vibrant energy of the urban environment. A faint smile plays on her lips, hinting at a sense of anticipation and wonder. The blurred motion of pedestrians and vehicles in the background emphasizes her serene presence amidst the bustling metropolis. Her youthful expression captures a moment of fleeting beauty and the boundless possibilities that lie ahead.
    
    Medium close-up shot of a distinguished dog in a tailored business suit, engrossed in a newspaper on a moving train. Neon lights flicker through the window, casting high-contrast shadows on the dog's face and emphasizing the low vibrance of the scene. The dog's brow is furrowed in concentration, its eyes scanning the newsprint with an air of intelligence and determination. The train's rhythmic motion rocks the dog gently, creating a subtle blur in the background that accentuates the dog's stillness and focus.
    
    Tracking shot of a vibrant yellow convertible cruising through a scenic Nevada desert. An orange filter bathes the scene in warm, golden light, highlighting the dramatic rock formations and vast sandy expanse. The car speeds along a winding road, leaving a trail of dust in its wake. The open top allows the driver and passengers to fully experience the breathtaking landscape, their hair tousled by the wind. The low camera angle captures the car's sleek design and emphasizes the sense of freedom and adventure. The orange filter adds a touch of nostalgia and creates a visually stunning scene that evokes the spirit of the open road and the allure of the desert.
    
    This street style shot captures two chic women strolling through the fashionable streets of Paris. The first woman exudes elegance in a pair of crisp white pants, a pastel pink blazer cinched with a black belt and oversized black sunglasses. The second woman radiates confidence in her yellow wide leg trousers and an oversized hot pink blouson accessorized with a chunky gold necklace. Both women carry luxurious handbags adding to their effortless sophistication. The backdrop of Parisian architecture and bustling city life complements their stylish ensembles, creating a picture perfect moment of Parisian chic.
    
    Now, provide 4 different REWRITES for the following USER_QUERY in the style above using about 100 words each. Only produce the final four rewrites, one on each line, no intermediate thoughts. The rewrites should be distinct from each other, while following the user's intent.
    """
    prompt = rewriter_preamble + "\n" + '"' + input_text + '"'
    for response in gemini15_multimodal.generate_content(prompt, stream = True):
        yield response.text

user_text = st.text_input("User query: ",key="suboptimal prompt")

submit = st.button("Rewrite!")

#If ask button is clicked
if submit:
    st.write_stream(generate_response_stream(user_text))

