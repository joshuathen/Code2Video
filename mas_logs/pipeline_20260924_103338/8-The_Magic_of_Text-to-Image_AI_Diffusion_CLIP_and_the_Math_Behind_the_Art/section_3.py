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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Diffusion adds noise to an image, destroying its structure.",
            "A neural network learns to predict this added noise.",
            "Reversing this process recovers the image from noise."
        ]
        self.setup_layout("The Diffusion Process: Math as an Artist", lecture_lines)
        
        # Load Assets
        canvas = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/canvas.svg")
        noise = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/noise.svg")
        
        # Applying requested fixes
        self.place_at_grid(canvas, 'D4', scale_factor=1.1)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_at_grid(noise, 'B5', scale_factor=0.6)
        noise.set_color("#808080")
        self.play(FadeIn(canvas), FadeIn(noise))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        net_text = Text("Neural Network", font_size=20, color=BLUE)
        self.place_at_grid(net_text, 'B4', scale_factor=1.0)
        self.play(Write(net_text))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        
        # Sequential transformation
        step_frames = 5
        target = Circle(radius=0.3, color=GREEN).set_fill(GREEN, opacity=0.8)
        self.place_at_grid(target, 'D4', scale_factor=0.8)
        
        # Animate noise to target (simulated reversal)
        self.play(
            Transform(noise, target, run_time=3),
            FadeOut(net_text)
        )
        
        # Glowing effect
        glow = Circle(radius=0.4, color=YELLOW).set_stroke(width=4)
        self.place_at_grid(glow, 'D4', scale_factor=0.8)
        self.play(Create(glow))
        
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
