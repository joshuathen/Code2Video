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
        self.setup_layout("Framing the Linear System", [
            "We represent Ax=b as a vector equation.", 
            "Scales x and y reach a target point b.", 
            "Think of reaching a crumb using two paths."
        ])
        
        # Load assets
        crumb = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/crumb.svg")
        
        # Objects
        axes = Axes(x_range=[0, 6], y_range=[0, 6], axis_config={"include_tip": True})
        self.place_in_area(axes, 'B2', 'F6', scale_factor=0.5)
        
        # Origin crumb
        crumb.scale(0.3).move_to(axes.c2p(0, 0))
        
        vec_col1 = Arrow(axes.c2p(0,0), axes.c2p(1,2), color=BLUE)
        vec_col2 = Arrow(axes.c2p(0,0), axes.c2p(3,1), color=YELLOW)
        vec_b = Dot(axes.c2p(5,5), color=RED)
        
        eq = MathTex("x", r"\begin{pmatrix} 1 \\ 2 \end{pmatrix}", "+", "y", r"\begin{pmatrix} 3 \\ 1 \end{pmatrix}", "=", r"\begin{pmatrix} 5 \\ 5 \end{pmatrix}")
        eq.set_color_by_tex("x", "#00FF00")
        eq.set_color_by_tex("y", "#00FF00")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"), Create(axes), FadeIn(crumb))
        self.play(GrowArrow(vec_col1), GrowArrow(vec_col2), FadeIn(vec_b))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.place_in_area(eq, 'D2', 'F5', scale_factor=0.6)
        self.play(self.lecture[1].animate.set_color("#00FFFF"), Write(eq))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.play(Indicate(eq.get_part_by_tex("x")), Indicate(eq.get_part_by_tex("y")))
        self.wait(2)
