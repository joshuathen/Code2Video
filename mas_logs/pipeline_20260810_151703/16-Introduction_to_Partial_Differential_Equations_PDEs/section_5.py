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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary and Real-World Impact", ["PDEs model our complex world.", "They drive weather, finance, and physics.", "Mastering them unlocks deep understanding."])
        
        # Define mobjects
        pde_elements = VGroup(
            MathTex(r"\frac{\partial u}{\partial t}", color="#FF0000"),
            MathTex(r"\nabla^2 u", color="#00FF00"),
            MathTex(r"f(x, t)", color="#0000FF")
        ).arrange(RIGHT, buff=0.5)
        
        final_formula = MathTex(r"\frac{\partial u}{\partial t} = \alpha \nabla^2 u + f", color=WHITE)
        final_text = Text("PDEs drive modern science", color="#FFD700")
        physics_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/physics.svg")
        
        final_group = VGroup(final_text, physics_icon).arrange(RIGHT, buff=0.3)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.place_in_area(pde_elements, 'B1', 'C6', scale_factor=1.2)
        self.play(FadeIn(pde_elements))
        self.play(self.lecture[0].animate.set_color("#FF0000"))

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.place_at_grid(final_formula, 'D4', scale_factor=1.0)
        self.play(Flash(final_formula, color=WHITE, line_length=0.2, num_lines=10))
        self.play(self.lecture[1].animate.set_color("#00FF00"))

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        self.place_at_grid(final_group, 'E4', scale_factor=0.9)
        self.play(Write(final_group))
        self.play(self.lecture[2].animate.set_color("#0000FF"))
        self.wait(2)
