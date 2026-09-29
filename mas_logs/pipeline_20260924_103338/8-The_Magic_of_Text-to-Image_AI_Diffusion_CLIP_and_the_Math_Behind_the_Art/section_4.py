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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Guidance: Bridging Text and Noise", [
            "We use text prompts to guide the denoising process.",
            "Guidance shifts the model towards the desired output image.",
            "This bridge connects abstract text to visual noise."
        ])

        # Colors
        COLOR_TEXT = "#FF69B4" # Text Prompt color
        COLOR_NOISE = "#808080" # Noise color
        COLOR_GUIDANCE = "#00FFFF"

        # === Animation for Lecture Line 1 ===
        # Using SVG assets per B018 and instruction 19
        text_prompt_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/keyboard.svg", color=COLOR_TEXT)
        noise_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microphone.svg", color=COLOR_NOISE)
        
        text_prompt_label = Text("Text Prompt", font_size=20, color=COLOR_TEXT)
        noise_label = Text("Noise", font_size=20, color=COLOR_NOISE)

        # Positioning per VideoCritic (Issue 30, 31, 32) + B002 (Cols 4-6)
        # Using Grid B4 for prompt, D5 for noise
        self.place_at_grid(text_prompt_icon, "B4", scale_factor=0.5)
        self.place_at_grid(text_prompt_label, "C4", scale_factor=0.6) # Label below icon
        
        self.place_at_grid(noise_icon, "D5", scale_factor=0.5)
        self.place_at_grid(noise_label, "E5", scale_factor=0.6) # Label below icon

        self.play(FadeIn(text_prompt_icon), Write(text_prompt_label), FadeIn(noise_icon), Write(noise_label))
        self.lecture[0].set_color(COLOR_TEXT)

        # === Animation for Lecture Line 2 ===
        arrow = Arrow(start=noise_icon.get_center(), end=text_prompt_icon.get_center(), color=COLOR_GUIDANCE)
        guidance_text = Text("Guidance Signal", font_size=18, color=COLOR_GUIDANCE)
        self.place_at_grid(guidance_text, "D6", scale_factor=0.7)

        self.play(Create(arrow), Write(guidance_text))
        self.lecture[1].set_color(COLOR_GUIDANCE)

        # === Animation for Lecture Line 3 ===
        glow = Dot(noise_icon.get_center(), color=WHITE, radius=0.1)
        self.play(glow.animate.move_to(text_prompt_icon.get_center()), run_time=2)
        self.play(FadeOut(glow))
        
        self.lecture[2].set_color(WHITE)
        self.wait(1)
