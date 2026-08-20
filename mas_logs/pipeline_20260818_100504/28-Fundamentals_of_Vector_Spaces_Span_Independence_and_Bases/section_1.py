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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Vectors are displacements, like (x, y).",
            "Basis vectors i-hat and j-hat build space.",
            "Any point is a combination of these."
        ]
        self.setup_layout("Prerequisite Review: Vectors as Arrows", lecture_lines)
        
        # Assets (Using placeholders/standard mobjects as assets aren't files)
        # Using SVGMobject placeholder for asset requirements
        VectorArrow = Arrow(ORIGIN, [1.5, 1, 0], buff=0, color=GOLD)
        BasisVectors = VGroup()
        
        # Create axes and place at recommended D4
        axes = Axes(x_range=[-1, 4], y_range=[-1, 3], axis_config={"include_tip": True}).scale(0.5)
        self.place_at_grid(axes, 'D4', scale_factor=0.6)
        
        # Place assets based on instructions
        self.place_in_area(VectorArrow, 'E1', 'F3', scale_factor=0.5)
        # Placeholder for basis vectors
        i_hat = Arrow(ORIGIN, RIGHT, color=BLUE)
        j_hat = Arrow(ORIGIN, UP, color=RED)
        BasisVectors.add(i_hat, j_hat)
        self.place_at_grid(BasisVectors, 'F5', scale_factor=0.7)
        
        label_v = MathTex(r"\vec{v}", color=WHITE).next_to(VectorArrow.get_end(), UP)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(GOLD))
        self.play(Create(VectorArrow), Write(label_v))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GOLD))
        i_label = MathTex(r"\hat{i}", color=BLUE).next_to(i_hat.get_end(), DOWN)
        j_label = MathTex(r"\hat{j}", color=RED).next_to(j_hat.get_end(), LEFT)
        
        self.play(Create(i_hat), Create(j_hat), Write(i_label), Write(j_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GOLD))
        point = Dot(axes.c2p(2, 1.5), color=WHITE)
        self.play(FadeIn(point))
        self.wait(2)
