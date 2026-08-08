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
            "Kolmogorov constant C is approximately 1.5.",
            "Statistical structure is universal in inertial range.",
            "Navier-Stokes complexity yields simple statistical patterns.",
            "Different scales share the same mathematical signature.",
            "Universal constants define turbulent energy distribution."
        ]
        self.setup_layout("Universal Constants and Scaling", lecture_lines)
        
        # Assets
        turbine = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/turbine.svg")
        engine = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/engine.svg")
        nozzle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/nozzle.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF8C00")
        c_text = MathTex("C", "\\approx", "1.5", font_size=48, color="#FF8C00")
        self.place_at_grid(c_text, 'A2', scale_factor=0.9)
        self.place_at_grid(turbine, 'A4', scale_factor=0.5)
        self.play(Write(c_text), FadeIn(turbine))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00CED1")
        l_param = MathTex("L", font_size=48, color="#00CED1")
        self.place_at_grid(l_param, 'B2', scale_factor=1.0)
        self.place_at_grid(engine, 'B4', scale_factor=0.5)
        self.play(Write(l_param), FadeIn(engine))
        self.play(l_param.animate.scale(0.5).set_opacity(0.5), engine.animate.scale(0.8), run_time=1.5)
        self.play(l_param.animate.scale(2.0).set_opacity(1.0), engine.animate.scale(1.25), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        pattern = VGroup(*[Dot(color="#FFD700") for _ in range(20)]).arrange_in_grid(4, 5)
        self.place_in_area(pattern, 'B3', 'D5', scale_factor=0.4)
        self.place_at_grid(nozzle, 'E4', scale_factor=0.5)
        self.play(FadeIn(pattern), FadeIn(nozzle))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#7FFF00")
        axes = Axes(x_range=[0, 3], y_range=[0, 3], tips=False).scale(0.3)
        slope = axes.plot(lambda x: x**(-0.5), color="#7FFF00") # Slope placeholder
        self.place_at_grid(VGroup(axes, slope), 'C4', scale_factor=0.6)
        self.play(Create(axes), Create(slope))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF69B4")
        energy_dot = Dot(color="#FF69B4", radius=0.2)
        self.place_at_grid(energy_dot, 'F4', scale_factor=0.7)
        self.play(Flash(energy_dot, color="#FF69B4"))
        self.wait(1)
