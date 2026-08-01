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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup the layout with section title and lecture lines
        self.setup_layout(
            "The Mathematical Engine: The Convolution Formula",
            [
                "Here is the formal convolution integral for the sum.",
                "The 'z minus x' term represents the sliding motion.",
                "We integrate to find the total area of overlap."
            ]
        )
        
        # Colors defined in the storyboard and issues
        BLUE_HIGHLIGHT = "#0000FF"
        RED_HIGHLIGHT = "#FF0000"
        YELLOW_HIGHLIGHT = "#FFFF00"
        WHITE_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Display the formula (f * g)(z) in white text
        formula = MathTex(
            r"(f * g)(z)", r"=", r"\int_{-\infty}^{\infty}", r"f(x)", r"g(z - x)", r"dx",
            font_size=42, color=WHITE_COLOR
        )
        # Position fixed based on VideoCritic feedback (Issue 32 & 33)
        self.place_in_area(formula, "B1", "D6", scale_factor=0.8)
        
        self.play(
            self.lecture[0].animate.set_color(YELLOW),
            Write(formula),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight f(x) in blue and g(z - x) in red
        # Match lecture line color to the red 'z - x' term for consistency
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(RED_HIGHLIGHT),
            formula[3].animate.set_color(BLUE_HIGHLIGHT), # f(x)
            formula[4].animate.set_color(RED_HIGHLIGHT),  # g(z - x)
            run_time=2
        )
        # Visual emphasis on the 'z - x' term mentioned in the lecture
        self.play(Indicate(formula[4], color=RED_HIGHLIGHT), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight the integral sign in glowing yellow
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW_HIGHLIGHT),
            formula[2].animate.set_color(YELLOW_HIGHLIGHT), # Integral sign
            run_time=1.5
        )
        
        # Add a subtle glow/emphasis to the integral sign
        # Pre-build mobject to avoid expensive creation in updater if we used one
        glow = SurroundingRectangle(formula[2], color=YELLOW_HIGHLIGHT, buff=0.1, stroke_width=2).set_opacity(0.3)
        self.play(FadeIn(glow), run_time=0.5)
        
        # Flash the final result (f * g)(z) to signify completion
        self.play(
            Flash(formula[0], color=WHITE_COLOR, line_length=0.4, num_lines=15),
            formula[0].animate.set_color(WHITE_COLOR).scale(1.2),
            run_time=1
        )
        self.play(
            formula[0].animate.scale(1/1.2), 
            FadeOut(glow), 
            run_time=0.5
        )
        
        self.wait(2)
