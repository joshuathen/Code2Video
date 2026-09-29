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
        self.setup_layout("Defining the 3D Cross Product", [
            "3D cross product is perpendicular.",
            "Use 3x3 determinant.",
            "Result is a new vector.",
            "It defines rotation axis.",
            "Follows the hinge orientation."
        ])
        
        # Objects
        vec_a = Arrow(start=ORIGIN, end=UP*1.5, color="#33FF57")
        vec_b = Arrow(start=ORIGIN, end=RIGHT*1.5, color="#33FF57")
        vec_c = Arrow(start=ORIGIN, end=OUT*1.5, color="#FF5733")
        
        hinge_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hinge.svg", color="#FFFF00")
        
        vectors = VGroup(vec_a, vec_b, vec_c)
        all_animations = VGroup(vectors, hinge_icon)
        
        # Place animations per critic recommendations
        self.place_in_area(all_animations, 'B4', 'E6', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(vec_a), FadeIn(vec_b), self.lecture[0].animate.set_color("#33FF57"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        det_text = MathTex(r"\det \begin{pmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ a_x & a_y & a_z \\ b_x & b_y & b_z \end{pmatrix}", font_size=24)
        self.place_in_area(det_text, 'D4', 'F6', scale_factor=0.8)
        self.play(Write(det_text), self.lecture[1].animate.set_color("#FFD700"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(GrowArrow(vec_c), self.lecture[2].animate.set_color("#FF5733"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(Rotate(vectors, angle=PI/4, axis=UP), self.lecture[3].animate.set_color("#33FFFF"))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(FadeIn(hinge_icon), self.lecture[4].animate.set_color("#FFFF00"))
        self.wait(1)
