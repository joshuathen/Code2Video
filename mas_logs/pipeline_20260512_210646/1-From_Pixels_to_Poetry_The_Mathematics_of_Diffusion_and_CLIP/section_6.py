from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        lecture_lines = [
            "CLIP's text embeddings guide the U-Net's denoising path.", 
            "Cross-attention layers focus the model on specific text cues.", 
            'The prompt "Astronaut on a Horse" directs pixel changes.', 
            "Global semantics now control the emergence of local details."
        ]
        self.setup_layout("Conditioning: Guiding the Denoising with CLIP", lecture_lines)

        # Colors
        CLIP_COLOR = "#FFFF00"
        UNET_COLOR = "#BB88FF"
        ATTENTION_COLOR = "#00FFFF"
        HEATMAP_COLOR = "#FF4400"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(CLIP_COLOR)
        
        # U-Net representation (a block with internal layers)
        unet_block = Rectangle(width=2.5, height=3.5, color=UNET_COLOR, fill_opacity=0.2)
        unet_label = Text("U-Net", font_size=24, color=UNET_COLOR)
        # Issue 48/59: Reposition unet_block to B2-E4
        self.place_in_area(unet_block, "B2", "E4")
        unet_label.next_to(unet_block, UP, buff=0.2)
        
        # CLIP Embedding (represented as a feature vector)
        clip_vector = VGroup(*[Square(side_length=0.3, color=CLIP_COLOR, fill_opacity=0.8) for _ in range(4)])
        clip_vector.arrange(RIGHT, buff=0.05)
        clip_label = Text("CLIP Text Embedding", font_size=18, color=CLIP_COLOR)
        self.place_at_grid(clip_vector, "B1")
        clip_label.next_to(clip_vector, UP, buff=0.2)
        
        self.play(FadeIn(unet_block), FadeIn(unet_label), FadeIn(clip_vector), FadeIn(clip_label))
        # Move vector towards U-Net
        self.play(
            clip_vector.animate.move_to(self.grid["B2"] + RIGHT * 0.5), 
            clip_label.animate.move_to(self.grid["A2"] + RIGHT * 0.5)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(ATTENTION_COLOR)
        
        # Cross-Attention Layers (Visualized as lines connecting Embedding to U-Net)
        attention_lines = VGroup()
        target_points = [
            unet_block.get_right() + UP * 1.0,
            unet_block.get_right(),
            unet_block.get_right() + DOWN * 1.0
        ]
        
        for p in target_points:
            line = Line(clip_vector.get_right(), p, color=ATTENTION_COLOR, stroke_width=4)
            attention_lines.add(line)
        
        attn_label = Text("Cross-Attention", font_size=20, color=ATTENTION_COLOR)
        self.place_at_grid(attn_label, "D3", scale_factor=0.8)
        
        self.play(Create(attention_lines), Write(attn_label))
        # Adding a glowing effect to the lines
        self.play(attention_lines.animate.set_stroke(width=8), run_time=0.5)
        self.play(attention_lines.animate.set_stroke(width=4), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        
        # Prompt Visualization
        prompt_text = Text('"Astronaut on a Horse"', font_size=20, color=WHITE, slant=ITALIC)
        # Issue 49/59: Reposition prompt_text to A5
        self.place_at_grid(prompt_text, "A5")
        
        # Heat Map / Noise Canvas
        noise_grid = VGroup()
        rows, cols = 8, 8
        for r in range(rows):
            for c in range(cols):
                dot = Square(side_length=0.2, fill_opacity=0.5, stroke_width=1)
                dot.set_fill(color=GREY, opacity=np.random.random())
                noise_grid.add(dot)
        noise_grid.arrange_in_grid(rows=rows, cols=cols, buff=0.02)
        # Issue 50/59: Reposition noise_grid to D5-F6
        self.place_in_area(noise_grid, "D5", "F6", scale_factor=0.8)
        
        # Heat map glow (representing the 'Astronaut' being identified)
        # Issue 33/59: Integrate Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/astronaut.svg
        astronaut_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/astronaut.svg")
        astronaut_icon.set_color(HEATMAP_COLOR).set_fill(HEATMAP_COLOR, opacity=0.6)
        # Place icon on the noise grid area
        astronaut_icon.scale(0.5).move_to(noise_grid.get_center())
        
        glow_label = Text("Astronaut Guidance", font_size=16, color=HEATMAP_COLOR)
        glow_label.next_to(noise_grid, DOWN, buff=0.1)

        self.play(FadeIn(prompt_text), FadeIn(noise_grid))
        self.play(FadeIn(astronaut_icon), Write(glow_label))
        self.play(astronaut_icon.animate.scale(1.2).set_fill(opacity=0.9), run_time=1, rate_func=there_and_back)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(WHITE)
        
        # Denoising progress: Noise grid becomes more structured
        structured_grid = noise_grid.copy()
        for i, box in enumerate(structured_grid):
            # Simulate pixels coming together - some becoming "blue" for sky or "brown" for horse
            if i % 3 == 0:
                box.set_fill(BLUE_D, opacity=0.9)
            elif i % 2 == 0:
                box.set_fill(WHITE, opacity=0.9)
            else:
                # DARK_BROWN is not a standard manim color, using a hex approximation
                box.set_fill("#5D4037", opacity=0.9) 
        
        self.play(
            ReplacementTransform(noise_grid, structured_grid),
            astronaut_icon.animate.set_fill(opacity=0.3),
            run_time=2
        )
        self.wait(2)
