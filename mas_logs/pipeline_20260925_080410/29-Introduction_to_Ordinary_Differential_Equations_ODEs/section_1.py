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
        self.setup_layout("The Core Concept: Change Over Time", [
            "ODEs model how functions change over time.", 
            "Algebra solves values; ODEs solve behaviors.", 
            "Acceleration defines a path's curvature."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Draw a function graph changing over time, label it #FFFFFF f(t).
        func_graph = FunctionGraph(lambda t: 0.5 * np.sin(2 * t), x_range=[-2, 2], color=BLUE)
        self.place_in_area(func_graph, 'A1', 'C3', scale_factor=0.8)
        self.play(Create(func_graph))
        label1 = Text("f(t)", color="#FFFFFF", font_size=20)
        self.place_at_grid(label1, 'B4')
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Highlight 'Algebra' as fixed values and 'ODE' as behavior, label them #FF8080 Algebra and #80FF80 ODE.
        alg_text = Text("Algebra", color="#FF8080", font_size=24)
        ode_text = Text("ODE", color="#80FF80", font_size=24)
        self.place_at_grid(alg_text, 'D1')
        self.place_at_grid(ode_text, 'D6')
        self.play(Write(alg_text), Write(ode_text))
        self.lecture[1].set_color("#FFFF00") # Highlighting the lecture line

        # === Animation for Lecture Line 3 ===
        # Show a [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/particle.svg] tracing a path, change color to #FFFF80 as it curves.
        particle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/particle.svg")
        self.place_at_grid(particle, 'E2')
        path = CurvedArrow(self.grid['E2'], self.grid['F5'], angle=-TAU/4)
        self.play(FadeIn(particle))
        particle.add_updater(lambda m: m.set_color("#FFFF80"))
        self.play(MoveAlongPath(particle, path), run_time=2)
        particle.remove_updater(lambda m: m.set_color("#FFFF80"))
        self.lecture[2].set_color("#FFFF80")
        self.wait(1)
