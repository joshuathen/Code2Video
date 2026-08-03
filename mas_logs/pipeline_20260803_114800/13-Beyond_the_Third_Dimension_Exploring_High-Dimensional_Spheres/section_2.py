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
        # Data from storyboard and issue 44
        lecture_lines = [
            "Start with the equation for a circle or sphere.",
            "Extend the variables to describe an n-dimensional hypersphere.",
            "We distinguish between the surface and the solid ball."
        ]
        self.setup_layout("Generalizing the Equation", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Chalkboard displays x^2 + y^2 = r^2 #FFFFFF.
        self.play(self.lecture[0].animate.set_color(WHITE))
        eq_base = MathTex("x^2 + y^2 = r^2", color=WHITE)
        self.place_in_area(eq_base, "C2", "C5", scale_factor=1.2)
        self.play(Write(eq_base))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Formula transforms to x_1^2 + x_2^2 +... + x_n^2 = r^2 #00FFFF.
        # Issue 31: Use C1-C6 and scale 1.1 for eqnd
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        eq_nd = MathTex("x_1^2 + x_2^2 + \\dots + x_n^2 = r^2", color="#00FFFF")
        self.place_in_area(eq_nd, "C1", "C6", scale_factor=1.1)
        
        # Issue 32: Highlight 'n' at C6
        n_highlight = Circle(radius=0.4, color=WHITE).set_stroke(width=3)
        self.place_at_grid(n_highlight, "C6")
        
        self.play(Transform(eq_base, eq_nd))
        self.play(Create(n_highlight))
        self.play(FadeOut(n_highlight))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Symbol S^n appears. Sphere surface flashes #00FF00.
        # Symbol B^n appears. Solid ball interior flashes #FF0000.
        # Issue 33: scale 1.0 for labels at B4/D4
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        sn_label = MathTex("S^n", color="#00FF00")
        self.place_at_grid(sn_label, "B4", scale_factor=1.0)
        
        bn_label = MathTex("B^n", color="#FF0000")
        self.place_at_grid(bn_label, "D4", scale_factor=1.0)
        
        surface_ring = Circle(radius=1.5, color="#00FF00").set_stroke(width=10)
        self.place_in_area(surface_ring, "A1", "F6")
        
        solid_ball = Circle(radius=1.5, color="#FF0000", fill_opacity=0.3).set_stroke(width=0)
        self.place_in_area(solid_ball, "A1", "F6")

        # Sequence of flashes
        self.play(FadeIn(sn_label))
        self.play(Create(surface_ring), run_time=0.5)
        self.play(FadeOut(surface_ring), run_time=0.5)
        
        self.play(FadeIn(bn_label))
        self.play(FadeIn(solid_ball), run_time=0.5)
        self.play(FadeOut(solid_ball), run_time=0.5)
        self.wait(1)

        # Text 'Logic' #FFFFFF appears as formulas fade.
        logic_text = Text("Logic", color=WHITE)
        self.place_in_area(logic_text, "C2", "C5", scale_factor=1.5)
        
        self.play(
            FadeOut(eq_base), # eq_base is the one on screen (transformed to eq_nd)
            FadeOut(sn_label),
            FadeOut(bn_label),
            FadeIn(logic_text)
        )
        self.wait(2)
