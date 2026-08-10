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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Scalar multiplication stretches or flips a vector's length.",
            "A scalar of two doubles the vector length.",
            "Negative scalars flip the vector's direction completely.",
            "Example: Jump twice as high with scalar two.",
            "Multiply by negative one to reverse the jump."
        ]
        self.setup_layout("Scalar Multiplication", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        v = Arrow(start=ORIGIN, end=RIGHT*1.5, color="#FFFF00", buff=0)
        v_label = MathTex(r"\vec{v}", color="#FFFF00").next_to(v, UP, buff=0.1)
        vector_group = VGroup(v, v_label)
        self.place_at_grid(vector_group, 'C2', scale_factor=0.7)
        self.play(Create(vector_group))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        v2 = Arrow(start=ORIGIN, end=RIGHT*3.0, color="#00FF00", buff=0)
        v2_label = MathTex(r"2\vec{v}", color="#00FF00").next_to(v2, UP, buff=0.1)
        vector2_group = VGroup(v2, v2_label)
        self.place_at_grid(vector2_group, 'C4', scale_factor=0.7)
        self.play(Create(vector2_group))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        v_neg = Arrow(start=ORIGIN, end=LEFT*1.5, color="#FF00FF", buff=0)
        v_neg_label = MathTex(r"-\vec{v}", color="#FF00FF").next_to(v_neg, UP, buff=0.1)
        vector_neg_group = VGroup(v_neg, v_neg_label)
        self.place_at_grid(vector_neg_group, 'D3', scale_factor=0.7)
        self.play(Create(vector_neg_group))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF00FF")
        self.wait(1)
