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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup the scene with title and lecture script
        title_text = "The Prior: Setting the Initial Map"
        lecture_lines = [
            "Our prior belief is the probability of a cat.",
            "Robo-Guard estimates a ten percent chance of a cat.",
            "We shade the bottom ten percent for this prior."
        ]
        self.setup_layout(title_text, lecture_lines)

        # --- Object Definitions ---
        
        # 1. Main Square (The initial 'territory' of belief)
        # Using a 3x3 grid unit size (Area B1 to E4 per issue 30)
        main_square = Square(side_length=3.0, color="#FFFFFF", stroke_width=4)
        self.place_in_area(main_square, "B1", "E4")
        
        # 2. Prior shading (representing P(Cat) = 0.1)
        # 10% of height (3.0 * 0.1 = 0.3)
        prior_shading = Rectangle(
            width=3.0, 
            height=0.3, 
            fill_color="#3498DB", 
            fill_opacity=0.8, 
            stroke_width=0
        )
        # Position using grid area to match square center, then align to bottom
        self.place_in_area(prior_shading, "B1", "E4")
        prior_shading.align_to(main_square, DOWN)
        
        # 3. Probability Labels in Gray (#7F8C8D)
        label_cat = Text("P(Cat) = 0.1", font_size=18, color="#7F8C8D")
        label_no_cat = Text("P(No Cat) = 0.9", font_size=18, color="#7F8C8D")
        
        # Position labels using the grid (moved from col 6 to col 5 per issues 31, 32)
        self.place_at_grid(label_cat, "E5", scale_factor=0.8) 
        self.place_at_grid(label_no_cat, "C5", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # Line: "Our prior belief is the probability of a cat."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            Create(main_square),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line: "Robo-Guard estimates a ten percent chance of a cat."
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color("#3498DB"),
            FadeIn(prior_shading),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line: "We shade the bottom ten percent for this prior."
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color("#7F8C8D"),
            Write(label_cat),
            Write(label_no_cat),
            run_time=2
        )
        self.wait(3)
