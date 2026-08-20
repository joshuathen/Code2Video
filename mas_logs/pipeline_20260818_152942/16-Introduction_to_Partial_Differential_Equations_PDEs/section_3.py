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
        lecture_lines = [
            "Heat equation is the quintessential PDE.",
            "Rate of change equals spatial curvature.",
            "Heat flows from hot to cold regions.",
            "This levels out temperature differences over time.",
            "Equilibrium is reached across the metal rod."
        ]
        self.setup_layout("Core Anatomy of a PDE", lecture_lines)
        
        # Asset Loading
        rod = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rod.svg")
        self.place_at_grid(rod, "A4", scale_factor=0.5)
        
        # Create MathTex PDE: u_t = \alpha u_{xx}
        pde = MathTex(r"u_t", "=", r"\alpha", r"u_{xx}", font_size=48)
        pde.set_color(WHITE)
        # Apply layout fix from Critic (Issue 29)
        self.place_in_area(pde, "B3", "C5", scale_factor=1.0)

        # Define terms
        u_t = pde[0]
        equal_sign = pde[1]
        alpha = pde[2]
        u_xx = pde[3]

        # Labels (Layout fixes from Critic Issue 28)
        label_u = Text("Unknown function", font_size=20, color=WHITE)
        self.place_at_grid(label_u, "C3", scale_factor=0.7)
        arrow1 = Arrow(label_u.get_bottom(), u_t.get_top(), buff=0.1, color=WHITE, tip_length=0.1)
        
        label_deriv = Text("Curvature term", font_size=20, color="#FF8000")
        self.place_at_grid(label_deriv, "E5", scale_factor=0.7)
        arrow2 = Arrow(label_deriv.get_top(), u_xx.get_bottom(), buff=0.1, color="#FF8000", tip_length=0.1)

        label_alpha = Text("Diffusivity", font_size=20, color="#00FF80")
        self.place_at_grid(label_alpha, "D5", scale_factor=0.7)
        arrow3 = Arrow(label_alpha.get_left(), alpha.get_right(), buff=0.1, color="#00FF80", tip_length=0.1)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(rod), self.lecture[0].animate.set_color(WHITE), Write(pde[0:2]))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF8000"), 
                  FadeIn(label_deriv), Create(arrow2), 
                  pde[3].animate.set_color("#FF8000"))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF80"), 
                  FadeIn(label_alpha), Create(arrow3), 
                  pde[2].animate.set_color("#00FF80"))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(WHITE), 
                  FadeIn(label_u), Create(arrow1))

        # === Animation for Lecture Line 5 ===
        # Flash rod and PDE structure
        self.play(self.lecture[4].animate.set_color(WHITE), 
                  Flash(rod, color=YELLOW),
                  pde.animate.set_opacity(0.5).set_opacity(1.0))
        self.wait(1)
