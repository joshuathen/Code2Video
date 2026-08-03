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

class Section7Scene(TeachingScene):
    def construct(self):
        # Fetch data from storyboard
        title_text = "Summary & Synthesis"
        lecture_lines = [
            "Differentiation breaks the area down into heights.",
            "Integration sums heights to reconstruct the area.",
            "They are two sides of one mathematical coin."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Colors as per requirements
        GOLD_COL = "#FFD700"
        SILVER_COL = "#C0C0C0"

        # === Animation for Lecture Line 1 ===
        # Storyboard: Show a circular flow: f(x) -> Integrator -> F(x) in gold (#FFD700).
        self.play(self.lecture[0].animate.set_color(GOLD_COL))
        
        fx = MathTex("f(x)", color=GOLD_COL)
        self.place_at_grid(fx, "C3", scale_factor=1.0)
        
        integrator_box = RoundedRectangle(corner_radius=0.1, height=0.6, width=1.6, color=GOLD_COL)
        integrator_text = Text("Integrator", font_size=18, color=GOLD_COL)
        integrator = VGroup(integrator_box, integrator_text)
        self.place_at_grid(integrator, "B4", scale_factor=0.8)
        
        Fx = MathTex("F(x)", color=GOLD_COL)
        self.place_at_grid(Fx, "C5", scale_factor=1.0)
        
        # Connection for integration path
        arrow1 = Arrow(fx.get_top(), integrator.get_left(), color=GOLD_COL, buff=0.1)
        arrow2 = Arrow(integrator.get_right(), Fx.get_top(), color=GOLD_COL, buff=0.1)
        
        self.play(FadeIn(fx))
        self.play(Create(arrow1), FadeIn(integrator))
        self.play(Create(arrow2), FadeIn(Fx))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Storyboard: Show the reverse: F(x) -> Differentiator -> f(x) in silver (#C0C0C0).
        self.play(self.lecture[1].animate.set_color(SILVER_COL))
        
        diff_box = RoundedRectangle(corner_radius=0.1, height=0.6, width=1.8, color=SILVER_COL)
        diff_text = Text("Differentiator", font_size=18, color=SILVER_COL)
        differentiator = VGroup(diff_box, diff_text)
        self.place_at_grid(differentiator, "D4", scale_factor=0.8)
        
        # Connection for differentiation path
        arrow3 = Arrow(Fx.get_bottom(), differentiator.get_right(), color=SILVER_COL, buff=0.1)
        arrow4 = Arrow(differentiator.get_left(), fx.get_bottom(), color=SILVER_COL, buff=0.1)
        
        self.play(Create(arrow3), FadeIn(differentiator))
        self.play(Create(arrow4))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Storyboard: Rotate the entire cycle to represent the inverse relationship.
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        full_cycle = VGroup(fx, Fx, integrator, differentiator, arrow1, arrow2, arrow3, arrow4)
        # Center of the diamond cycle is at C4
        center_point = self.grid["C4"]
        
        self.play(Rotate(full_cycle, angle=2*PI, about_point=center_point, run_time=5))
        self.wait(2)
