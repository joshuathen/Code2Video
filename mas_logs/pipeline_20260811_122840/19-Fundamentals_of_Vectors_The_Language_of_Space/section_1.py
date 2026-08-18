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
        self.setup_layout("What is a Vector?", ["Scalars have only magnitude.", "Vectors add direction to magnitude.", "Vectors act like arrows in space."])
        
        # Define colors for lecture lines
        COLOR_SCALAR = "#FFFFFF"
        COLOR_VECTOR_V = "#FF00FF"
        COLOR_VECTOR_U = "#00FFFF"

        # === Animation for Lecture Line 1 ===
        # Use a scalar representation (e.g., number or dot) for magnitude
        scalar_val = MathTex("s = 5.0", color=COLOR_SCALAR)
        self.place_at_grid(scalar_val, 'B4', scale_factor=0.8)
        self.play(Write(scalar_val))
        self.lecture[0].set_color(COLOR_SCALAR)

        # === Animation for Lecture Line 2 ===
        # Show arrow representing vector v.
        arrow1 = Arrow(start=LEFT*1, end=RIGHT*1, color=COLOR_VECTOR_V)
        label1 = MathTex(r"\vec{v}", color=COLOR_VECTOR_V).next_to(arrow1, UP, buff=0.1).scale(0.8)
        v_group1 = VGroup(arrow1, label1)
        self.place_at_grid(v_group1, 'B3', scale_factor=0.9)
        self.play(Create(v_group1))
        self.lecture[1].set_color(COLOR_VECTOR_V)

        # === Animation for Lecture Line 3 ===
        # Draw both scalar s and vector v alongside
        arrow2 = Arrow(start=LEFT*1, end=RIGHT*1, color=COLOR_VECTOR_U)
        label2 = MathTex(r"\vec{u}", color=COLOR_VECTOR_U).next_to(arrow2, UP, buff=0.1).scale(0.8)
        v_group2 = VGroup(arrow2, label2)
        v_group_combined = VGroup(v_group1, v_group2).arrange(DOWN, buff=0.5)
        
        self.place_in_area(v_group_combined, 'D3', 'E4', scale_factor=0.85)
        self.play(FadeIn(v_group2), v_group1.animate.move_to(v_group_combined[0].get_center()))
        self.lecture[2].set_color(COLOR_VECTOR_U)
        self.wait(1)
