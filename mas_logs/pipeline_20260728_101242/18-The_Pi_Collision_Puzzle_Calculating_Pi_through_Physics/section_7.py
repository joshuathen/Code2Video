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

class Section7Scene(TeachingScene):
    def construct(self):
        # Initializing layout
        title_text = "Summary and Takeaway"
        lecture_lines = [
            "We calculated Pi using only blocks and collisions.",
            "Conservation laws reveal deep, hidden geometric symmetries.",
            "Mathematics and physics are fundamentally and beautifully linked."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight Line 1 in Blue
        self.play(self.lecture[0].animate.set_color(BLUE))

        # Blocks on the right side of the grid to avoid lecture text
        block_small = Square(side_length=0.7, fill_opacity=1, color=BLUE)
        block_large = Square(side_length=1.2, fill_opacity=1, color=GREEN)
        
        self.place_at_grid(block_small, "C4")
        self.place_at_grid(block_large, "C5")
        
        self.play(FadeIn(block_small), FadeIn(block_large))
        
        # Collision animation
        self.play(
            block_small.animate.shift(RIGHT * 0.3),
            block_large.animate.shift(LEFT * 0.1),
            rate_func=there_and_back,
            run_time=0.6
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Switch highlight to Red for Pi digits
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FF4D4F")
        )

        # Pi digits on the far right side of the grid
        pi_digits = Text("3.1415...", font_size=42, color="#FF4D4F")
        self.place_in_area(pi_digits, "B6", "D6")
        
        self.play(Write(pi_digits))
        self.play(Indicate(pi_digits, color="#FF4D4F"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Switch highlight to Yellow for the connection arc
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FADB14")
        )

        # Yellow circular arc connecting blocks area to digits area
        # Center points for start (blocks) and end (digits)
        start_point = self.grid["C4"] + UP * 0.8
        end_point = self.grid["C6"] + UP * 0.8
        
        connection_arc = ArcBetweenPoints(
            start=start_point,
            end=end_point,
            angle=-TAU/4,
            color="#FADB14"
        )
        
        self.play(Create(connection_arc))
        self.play(connection_arc.animate.set_stroke(width=10), rate_func=there_and_back)
        self.wait(2)

        # Final state
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
