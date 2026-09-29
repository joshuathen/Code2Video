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
        lecture_lines = ["Any vector is a linear combination.", "Summing basis vectors reaches any point.", "Coordinates are weights of the basis."]
        self.setup_layout("Prerequisite Review: Basis and Linear Combinations", lecture_lines)
        
        # Define objects
        # Improvement: Axes object resizing to prevent encroachment
        axes = Axes(x_range=[0, 4], y_range=[0, 3], axis_config={"include_tip": True})
        self.place_in_area(axes, 'B3', 'E5', scale_factor=0.6)
        
        # Load asset - although '/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg' is just a placeholder, we acknowledge it as per instructions
        # icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        
        i_hat = Vector(axes.c2p(1, 0) - axes.c2p(0, 0), color="#F1C40F").shift(axes.c2p(0,0))
        j_hat = Vector(axes.c2p(0, 1) - axes.c2p(0, 0), color="#F1C40F").shift(axes.c2p(0,0))
        
        v = Vector(axes.c2p(3, 2) - axes.c2p(0, 0), color=WHITE).shift(axes.c2p(0,0))
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#F1C40F"))
        self.play(Create(axes))
        self.play(GrowArrow(i_hat), GrowArrow(j_hat))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#F1C40F"))
        
        # Vector summation visuals using coordinates
        v_tail = axes.c2p(0, 0)
        # 3 i-hats
        i1 = Vector(axes.c2p(1, 0) - axes.c2p(0, 0), color="#F1C40F").shift(axes.c2p(0,0))
        i2 = Vector(axes.c2p(1, 0) - axes.c2p(0, 0), color="#F1C40F").shift(axes.c2p(1,0))
        i3 = Vector(axes.c2p(1, 0) - axes.c2p(0, 0), color="#F1C40F").shift(axes.c2p(2,0))
        # 2 j-hats
        j1 = Vector(axes.c2p(0, 1) - axes.c2p(0, 0), color="#F1C40F").shift(axes.c2p(3,0))
        j2 = Vector(axes.c2p(0, 1) - axes.c2p(0, 0), color="#F1C40F").shift(axes.c2p(3,1))
        
        self.play(Create(i1), Create(i2), Create(i3), Create(j1), Create(j2))
        self.play(Create(v))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#2ECC71"))
        
        # Correctly anchor and scale formula_v
        formula_v = MathTex(r"v = 3\hat{i} + 2\hat{j}", color="#2ECC71")
        self.place_in_area(formula_v, 'D3', 'D5', scale_factor=0.65)
        
        self.play(Write(formula_v))
        self.wait(2)
