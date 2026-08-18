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
        self.setup_layout("The Product Rule: The 'Teamwork' Formula", [
            "Product rule finds total change.", 
            "Functions f(x) and g(x) multiply.", 
            "Formula: f'g plus fg'."
        ])
        
        uv = MathTex("u", "v", color=WHITE)
        uv[0].set_color("#FFFF00")
        uv[1].set_color("#00FF00")
        
        formula = MathTex("u'", "v", "+", "u", "v'", color=WHITE)
        formula[0].set_color("#FFFF00")
        formula[1].set_color("#00FF00")
        formula[3].set_color("#FFFF00")
        formula[4].set_color("#00FF00")
        
        u_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        v_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(uv, "B5", scale_factor=1.2)
        self.play(FadeIn(uv))
        self.lecture[0].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(u_icon, "C3", scale_factor=0.5)
        self.place_at_grid(v_icon, "C4", scale_factor=0.5)
        self.play(FadeIn(u_icon), FadeIn(v_icon))
        self.play(Indicate(uv[0]), Indicate(uv[1]))
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(FadeOut(uv), FadeOut(u_icon), FadeOut(v_icon))
        # self.place_in_area(formula, 'C4', 'D6', scale_factor=1.0)
        self.place_at_grid(formula, "D4", scale_factor=1.1)
        self.play(Write(formula))
        self.play(formula[2].animate.set_color("#FF0000"))
        
        # Keep u and v labels
        u_label = MathTex("u", color="#FFFF00").scale(1.2).move_to(self.grid["B3"])
        v_label = MathTex("v", color="#00FF00").scale(1.2).move_to(self.grid["B4"])
        self.play(FadeIn(u_label), FadeIn(v_label))
        
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
        self.play(FadeOut(formula), FadeOut(u_label), FadeOut(v_label))
