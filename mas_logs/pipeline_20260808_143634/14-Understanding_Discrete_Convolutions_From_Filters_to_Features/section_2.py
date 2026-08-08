from manim import *

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
        lecture_lines = [
            "Discrete convolution defines how two sequences interact.",
            "First, we flip the filter sequence.",
            "Next, slide it over the input data.",
            "Then, multiply overlapping elements together.",
            "Finally, sum them for the output result."
        ]
        self.setup_layout("Mathematical Core: The Discrete Convolution", lecture_lines)
        
        # Elements
        formula = MathTex(r"(f * g)[n] = \sum_{m} f[m] \cdot g[n-m]", font_size=32, color=WHITE)
        seq_f = VGroup(*[Square(side_length=0.6).set_fill(BLUE, opacity=0.5) for _ in range(3)]).arrange(RIGHT)
        seq_g = VGroup(*[Square(side_length=0.6).set_fill(RED, opacity=0.5) for _ in range(4)]).arrange(RIGHT)
        
        filter_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/filter.svg")
        window_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/window.svg")

        # Fix issues 24, 25, 26
        self.place_in_area(formula, 'A3', 'B5', scale_factor=1.2)
        self.place_at_grid(seq_f, 'C3', scale_factor=0.9)
        self.place_at_grid(seq_g, 'E3', scale_factor=0.9)
        
        # Place assets
        self.place_at_grid(filter_icon, 'A2', scale_factor=0.5)
        self.place_at_grid(window_icon, 'F3', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        self.play(Write(formula), FadeIn(filter_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color("#FF5733")
        self.play(seq_g.animate.rotate(PI))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color(BLUE)
        self.play(seq_g.animate.shift(UP * 1.8), FadeIn(window_icon))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(GRAY)
        self.lecture[3].set_color("#00FF00")
        # Visualizing overlap multiplication
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(GRAY)
        self.lecture[4].set_color("#FFFF00")
        self.wait(1)
