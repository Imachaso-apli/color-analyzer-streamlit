import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
from colorsys import rgb_to_hls, hls_to_rgb
import matplotlib.pyplot as plt
from io import BytesIO
import base64

st.set_page_config(page_title="Color Analyzer 60-30-10", layout="centered")
st.title("\U0001F3A8 Image Color Analyzer for 60-30-10 Rule")

# --- Helper Functions ---
def rgb_to_hex(color):
    return '#{:02X}{:02X}{:02X}'.format(*color)

def extract_dominant_colors(image, k=5):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(img)
    colors = np.array(kmeans.cluster_centers_, dtype='uint8')
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    ratios = counts / counts.sum()
    return list(zip(colors, ratios))

def simplify_colors(color_info, threshold=0.0):
    simplified = []
    other_ratio = 0.0
    other_rgb = np.array([0.0, 0.0, 0.0])
    for color, ratio in color_info:
        if ratio * 100 < threshold:
            other_ratio += ratio
            other_rgb += color * ratio
        else:
            simplified.append((color, ratio))
    if other_ratio > 0:
        other_rgb = (other_rgb / other_ratio).astype('uint8')
        simplified.append((other_rgb, other_ratio))
    return simplified

def plot_color_pie_chart(color_info):
    labels = []
    sizes = []
    color_values = []
    for color, ratio in color_info:
        hex_code = rgb_to_hex(color)
        labels.append(f"{hex_code} - {ratio*100:.1f}%")
        sizes.append(ratio)
        color_values.append(tuple(c / 255 for c in color))

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=color_values, startangle=90, counterclock=False)
    ax.axis('equal')
    return fig

def suggest_accent_colors(base_colors):
    base_color = np.mean(base_colors, axis=0).astype('uint8')
    r, g, b = base_color
    h, l, s = rgb_to_hls(r/255, g/255, b/255)

    accent_colors = []
    for dl, ds in [(0.2, 0.0), (0.0, 0.3), (-0.2, 0.0)]:
        new_h = (h + 0.5) % 1.0
        new_l = min(max(l + dl, 0), 1)
        new_s = min(max(s + ds, 0), 1)
        rgb = (np.array(hls_to_rgb(new_h, new_l, new_s)) * 255).astype('uint8')
        accent_colors.append(rgb_to_hex(rgb))

    tone_colors = []
    for dl, ds in [(-0.15, 0.0), (0.0, -0.3), (0.15, 0.0)]:
        new_l = min(max(l + dl, 0), 1)
        new_s = min(max(s + ds, 0), 1)
        rgb = (np.array(hls_to_rgb(h, new_l, new_s)) * 255).astype('uint8')
        tone_colors.append(rgb_to_hex(rgb))

    return accent_colors, tone_colors

# --- Main Interface ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
clusters = st.slider("Number of color clusters", 3, 10, 5)
threshold = st.slider("Group colors below this % into 'Other'", 0, 20, 5)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

    color_info = extract_dominant_colors(image, k=clusters)
    color_info.sort(key=lambda x: -x[1])
    simplified = simplify_colors(color_info, threshold)
    simplified.sort(key=lambda x: -x[1])

    st.subheader("\U0001F58C Dominant Colors")
    for color, ratio in simplified:
        hex_code = rgb_to_hex(color)
        st.markdown(f"- `{hex_code}` — **{ratio*100:.2f}%**")

    fig = plot_color_pie_chart(simplified)
    st.pyplot(fig)

    if len(simplified) >= 2:
        base_colors = [simplified[0][0], simplified[1][0]]
        accents, tones = suggest_accent_colors(base_colors)

        st.subheader("\U0001F3A8 Suggested Accent Colors")
        st.markdown("**Contrast Accents (Complementary):**")
        st.markdown(", ".join([f"`{c}`" for c in accents]))
        st.markdown("**Tone-on-Tone Variants:**")
        st.markdown(", ".join([f"`{c}`" for c in tones]))
