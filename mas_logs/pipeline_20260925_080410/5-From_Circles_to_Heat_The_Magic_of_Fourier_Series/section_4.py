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
        self.setup_layout("Synthesis: Thermal Equilibrium", [
            "Frequency dictates thermal equilibrium speed.",
            "Circular phase maps to space.",
            "Heat diffuses to average temperature."
        ])
        
        # 1. Assets
        thermometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/thermometer.svg")
        heater = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/heater.svg")
        
        # 1. Epicycles representing temperature distribution
        epicycle_group = VGroup(
            Circle(radius=1.0, color='#FFFFFF'),
            Dot(color='#FF8000').shift(RIGHT * 1.0),
            thermometer
        )
        self.place_at_grid(epicycle_group, 'B2', scale_factor=0.7)
        
        # 2. Heat distribution plot
        heat_graph = FunctionGraph(lambda x: 0.5 * np.cos(2 * PI * x), x_range=[-1, 1], color='#FFFF00')
        self.place_at_grid(heat_graph, 'D3', scale_factor=0.6)
        
        # 3. Equilibrium state
        equil_line = VGroup(
            Line(LEFT, RIGHT, color='#00FF00'),
            heater
        )
        self.place_at_grid(equil_line, 'E3', scale_factor=0.6)
        equil_line.set_opacity(0)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color('#FF8000')
        self.play(Create(epicycle_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color('#FFFF00')
        self.play(Transform(epicycle_group, heat_graph))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color('#00FF00')
        self.play(FadeIn(equil_line), FadeOut(epicycle_group))
        self.wait(1)
