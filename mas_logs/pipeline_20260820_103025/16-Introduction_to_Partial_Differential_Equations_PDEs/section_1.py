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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Defining the Problem: ODE vs. PDE", [
            "ODEs track change in one variable, like time.",
            "PDEs model complex phenomena across multiple variables.",
            "Think of a single pendulum versus pond ripples."
        ])
        
        # === Animation for Lecture Line 1 ===
        # ODE: One Variable
        ode_text = Text("ODE: One Variable", color=WHITE)
        self.place_in_area(ode_text, 'B2', 'B3', scale_factor=0.6)
        self.play(Write(ode_text))
        self.play(self.lecture[0].animate.set_color("#FF9900"))

        # === Animation for Lecture Line 2 ===
        # PDE: Many Variables
        pde_text = Text("PDE: Many Variables", color=WHITE)
        self.place_in_area(pde_text, 'B5', 'B6', scale_factor=0.6)
        self.play(Write(pde_text))
        self.play(self.lecture[1].animate.set_color("#00CCFF"))

        # === Animation for Lecture Line 3 ===
        # Pendulum (ODE)
        pendulum = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pendulum.svg", color="#FF9900")
        self.place_at_grid(pendulum, 'C4', scale_factor=0.5)
        pendulum_label = Text("ODE", color="#FF9900", font_size=20)
        self.place_at_grid(pendulum_label, 'C4')
        pendulum_label.next_to(pendulum, DOWN, buff=0.1).scale(0.7)
        
        # Ripple (PDE)
        ripple = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pond.svg", color="#00CCFF")
        self.place_at_grid(ripple, 'C5', scale_factor=0.5)
        ripple_label = Text("PDE", color="#00CCFF", font_size=20)
        self.place_at_grid(ripple_label, 'C5')
        ripple_label.next_to(ripple, DOWN, buff=0.1).scale(0.7)

        self.play(Create(pendulum), Write(pendulum_label))
        self.play(Create(ripple), Write(ripple_label))
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
