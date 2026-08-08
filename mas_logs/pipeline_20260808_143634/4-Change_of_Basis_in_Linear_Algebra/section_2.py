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
        lecture_lines = ["A basis spans the vector space.", "Standard basis vectors i and j.", "Tilted alternative vectors u and v."]
        self.setup_layout("Prerequisite Review: Basis as Building Blocks", lecture_lines)
        
        # Vectors
        i_vec = Arrow(ORIGIN, RIGHT, color=BLUE)
        j_vec = Arrow(ORIGIN, UP, color=BLUE)
        u_vec = Arrow(ORIGIN, RIGHT + UP, color=YELLOW)
        v_vec = Arrow(ORIGIN, -1*RIGHT + UP, color=YELLOW)
        
        # Labels
        i_label = MathTex("i", color=BLUE)
        j_label = MathTex("j", color=BLUE)
        u_label = MathTex("u", color=YELLOW)
        v_label = MathTex("v", color=YELLOW)

        # Asset placeholding
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]
        # In a real scenario, this would load the icon. Since it's 'none', we proceed with the requested vectors.

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF4500")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF4500")
        
        # Addressing issue 24 & 37: Fixing overlaps and positions
        self.place_at_grid(i_vec, 'C3')
        self.place_at_grid(j_vec, 'C3')
        self.place_at_grid(i_label, 'C4')
        self.place_at_grid(j_label, 'B3', scale_factor=0.8)
        
        self.play(FadeIn(i_vec), FadeIn(j_vec), Write(i_label), Write(j_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF4500")
        
        # Addressing issue 25 & 37: Tilted vectors position
        self.place_at_grid(u_vec, 'E4')
        self.place_at_grid(v_vec, 'E4')
        self.place_at_grid(u_label, 'E5')
        self.place_at_grid(v_label, 'D4', scale_factor=0.8)
        
        # Addressing issue 26 & 37: Grouped move for better separation
        group_basis = VGroup(u_vec, v_vec, u_label, v_label)
        self.place_in_area(group_basis, 'B2', 'E5', scale_factor=0.9)
        
        self.play(
            FadeOut(i_vec), FadeOut(j_vec), FadeOut(i_label), FadeOut(j_label),
            FadeIn(u_vec), FadeIn(v_vec), Write(u_label), Write(v_label)
        )
        self.wait(2)
