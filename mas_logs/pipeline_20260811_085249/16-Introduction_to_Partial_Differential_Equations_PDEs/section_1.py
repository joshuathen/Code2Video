from manim import *
import numpy as np

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
        self.setup_layout("The Concept: From ODEs to PDEs", [
            "ODEs relate one variable, like a moving point.",
            "PDEs involve multiple variables, like time and space.",
            "Think of a single bee versus a swarm."
        ])
        
        # --- Visual Objects ---
        # ODE graph
        axes_ode = Axes(x_range=[0, 4, 1], y_range=[-1, 1, 0.5], x_length=2.5, y_length=1.5).set_color(WHITE)
        func_ode = axes_ode.plot(lambda x: 0.5 * np.sin(x * PI), color="#FF0000")
        bee_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bee.svg").scale(0.3)
        ode_group = VGroup(axes_ode, func_ode, bee_icon)
        
        # PDE surface (simplified)
        axes_pde = ThreeDAxes(x_range=[-2, 2, 1], y_range=[-2, 2, 1], z_range=[-1, 1, 1], x_length=2.5, y_length=2.5, z_length=1.5).set_color(WHITE)
        surface_pde = axes_pde.plot_surface(lambda u, v: 0.5 * np.sin(u) * np.cos(v), u_range=[-2, 2], v_range=[-2, 2], color="#00FF00")
        swarm_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/swarm.svg").scale(0.3)
        pde_group = VGroup(axes_pde, surface_pde, swarm_icon)
        
        # Text for derivative
        diff_text = Text("Total vs Partial Derivatives", color="#FFFF00", font_size=24)

        # === Animation for Lecture Line 1 ===
        self.place_in_area(ode_group, 'C1', 'C2', scale_factor=0.8)
        self.play(Create(ode_group), self.lecture[0].animate.set_color("#FF0000"))
        
        # Move bee along curve
        bee_icon.move_to(axes_ode.c2p(0, 0))
        self.play(MoveAlongPath(bee_icon, func_ode), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeOut(ode_group), self.lecture[0].animate.set_color(WHITE))
        self.place_in_area(pde_group, 'D4', 'E5', scale_factor=1.0)
        self.play(Create(pde_group), self.lecture[1].animate.set_color("#00FF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(diff_text, 'A4', scale_factor=0.9)
        self.play(Write(diff_text), self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#FFFF00"))
        self.wait(2)
        self.play(FadeOut(pde_group), FadeOut(diff_text), self.lecture[2].animate.set_color(WHITE))
