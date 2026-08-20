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
        lecture_lines = [
            "ODEs depend on a single variable, like time.",
            "PDEs involve multiple variables, like space and time.",
            "Variables interact across a multi-dimensional field."
        ]
        self.setup_layout("From ODEs to PDEs: The Leap in Complexity", lecture_lines)
        
        # Define Mobjects
        eq_ode = MathTex(r"\frac{dy}{dt} = f(t, y)")
        clock = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/clock.svg")
        
        system_pde = MathTex(r"\frac{\partial u}{\partial t} = \nabla^2 u")
        
        globe = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/globe.svg")

        # === Animation for Lecture Line 1 ===
        self.place_in_area(eq_ode, 'B4', 'B5', scale_factor=0.9)
        self.play(FadeIn(eq_ode))
        self.place_at_grid(clock, 'B6', scale_factor=0.4)
        self.play(FadeIn(clock))
        self.play(self.lecture[0].animate.set_color("#FF5733"), eq_ode.animate.set_color("#FF5733"))

        # === Animation for Lecture Line 2 ===
        self.place_in_area(system_pde, 'D4', 'D5', scale_factor=0.9)
        self.play(FadeIn(system_pde))
        self.play(self.lecture[1].animate.set_color("#33FF57"), system_pde.animate.set_color("#33FF57"))

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(globe, 'E4', scale_factor=0.5)
        arrow = Arrow(start=system_pde.get_bottom(), end=globe.get_top(), color="#3357FF")
        self.play(FadeIn(globe), Create(arrow))
        self.play(self.lecture[2].animate.set_color("#3357FF"))
        self.wait(2)
