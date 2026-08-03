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
        # Define lecture lines based on storyboard
        lecture_lines = [
            "- Colliding blocks act as a mechanical computer for Pi.",
            "- Geometry is deeply embedded in the laws of motion.",
            "- Simple physics reveals the most fundamental mathematical constants."
        ]
        self.setup_layout("Conclusion: The Unity of Math and Nature", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Highlight current lecture line in BLUE_B
        self.lecture[0].set_color(BLUE_B)
        
        # Wall at the left of the interaction area
        wall = Rectangle(width=0.1, height=5, color=GREY, fill_opacity=0.5)
        self.place_in_area(wall, 'A1', 'F1')
        
        # Load and place blocks asset [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg]
        # Resolve Issue 21: Integrating the provided asset
        blocks = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg")
        blocks.set_color(BLUE_B) # Apply matching color
        self.place_at_grid(blocks, 'D2', scale_factor=1.2)
        
        self.play(FadeIn(wall), FadeIn(blocks))
        
        # Animate blocks sliding to the right, away from the wall
        self.play(
            blocks.animate.move_to(self.grid['D5']),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Reset color and highlight next line
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(WHITE) # Digits are white as per storyboard
        
        pi_digits = Text("3.14159...", font_size=48, color=WHITE)
        # Resolve Issue 33: Adjust position and scale to avoid overlap
        self.place_in_area(pi_digits, 'B2', 'B5', scale_factor=0.8)
        
        self.play(Write(pi_digits))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Reset color and highlight next line
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW) # Use YELLOW for the final constant
        
        pi_symbol = MathTex(r"\pi", font_size=144, color=YELLOW)
        # Resolve Issue 32: Adjust position and scale to avoid overlap
        self.place_in_area(pi_symbol, 'C4', 'E5', scale_factor=0.8)
        
        # Fade out everything except the Pi symbol as per storyboard "Fade all elements to black..."
        self.play(
            FadeOut(wall),
            FadeOut(blocks),
            FadeOut(pi_digits),
            FadeIn(pi_symbol)
        )
        
        # Glowing/Highlighting effect for the Pi symbol
        self.play(
            pi_symbol.animate.scale(1.1).set_color(WHITE),
            rate_func=there_and_back,
            run_time=2
        )
        
        self.wait(3)
