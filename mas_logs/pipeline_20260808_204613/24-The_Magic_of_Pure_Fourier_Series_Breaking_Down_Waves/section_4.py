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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "We extract coefficients using the principle of orthogonality.",
            "Signal passes through specific frequency gates.",
            "The integral acts as a mathematical prism."
        ]
        self.setup_layout("Calculation via Orthogonality", lecture_lines)
        
        # Define elements
        vec_s = Arrow(ORIGIN, RIGHT, color=WHITE)
        vec_c = Arrow(ORIGIN, UP, color=WHITE)
        vgroup_ortho = VGroup(vec_s, vec_c)
        
        prism_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg")
        filter_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/filter.svg")
        
        inner_prod_text = MathTex(r"\\langle s, c \\rangle = 0", color=RED)
        
        signal = FunctionGraph(lambda x: np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * x), x_range=[-1, 1], color=TEAL)
        projection = Line(ORIGIN, RIGHT * 0.5, color=TEAL)
        
        c_n_label = MathTex(r"c_n", color=YELLOW)
        
        integral_simpl = MathTex(r"\\int f(x) e^{-inx} dx = c_n", color=GREEN)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.place_in_area(vgroup_ortho, "A1", "C2", scale_factor=0.8)
        self.place_at_grid(prism_img, "B2", scale_factor=0.5)
        self.play(Create(vec_s), Create(vec_c), FadeIn(prism_img))
        self.wait(1)
        
        self.place_at_grid(inner_prod_text, "D2")
        self.play(Write(inner_prod_text))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        self.place_in_area(signal, "A3", "B6", scale_factor=0.4)
        self.place_at_grid(projection, "C4", scale_factor=0.7)
        self.play(Create(signal), Create(projection))
        self.wait(1)
        
        self.place_at_grid(c_n_label, "E4")
        self.play(Write(c_n_label))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(BLUE)
        self.place_in_area(integral_simpl, "D1", "E6", scale_factor=0.5)
        self.place_at_grid(filter_img, "F3", scale_factor=0.6)
        self.play(Write(integral_simpl), FadeIn(filter_img))
        self.wait(2)
