# Image Steganography Mobile App

## About

<p align="justify">
Steganography is the technique of hiding information within digital media to ensure secure communication. Traditional steganographic methods, such as Least Significant Bit (LSB) substitution, are vulnerable to attacks and can be easily detected by steganalysis tools. To address these limitations, this project proposes Generative Adversarial Networks (GAN) for Semantic Steganography System that embeds messages into the semantic feature space of images rather than directly modifying pixel values. The system consists of an <b>Encoder</b> that hides the message in deep feature maps, a GAN-based <b>Generator</b> that enhances security and a <b>Decoder</b> that accurately extracts the hidden message. This approach increases message retention, security, and robustness against detection while preserving the visual quality of the image. The project aims to optimize real-time performance and explore applications in AI security, medical imaging, and 5G communication.
</p>

## Output
### Encoding Phase
First we'll upload the image that we want the message to be hidden. Then, we'll put the secret key and the message and do the encoding process. After this, the image and the key will be shared with another person.

<img width="3537" height="2307" alt="collage (1)" src="https://github.com/user-attachments/assets/98b21e24-efae-4eca-9d5f-7267c9ee9147" />

### Decoding Phase
In this phase, the user will upload the encoded image. Next, they will use the secret key shared by the sender to extract the hidden message. Finally, the message will be extracted.

<img width="3527" height="2303" alt="collage (2)" src="https://github.com/user-attachments/assets/a6ba70a6-aa15-4a7b-b5ee-af4c63cf1591" />
