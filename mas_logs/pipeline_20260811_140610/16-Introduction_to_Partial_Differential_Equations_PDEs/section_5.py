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
        self.setup_layout("Summary & Real-World Impact", [
            "PDEs are the mathematical language of nature.", 
            "They govern everything from weather to finance.", 
            "We move from tracking points to fields."
        ])
        
        # Colors for lecture lines
        c1, c2, c3 = "#FFD700", "#00BFFF", "#FF4500"
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(c1)
        eqn = MathTex(r"\frac{\partial u}{\partial t} = \alpha \nabla^2 u", color=c1)
        self.place_in_area(eqn, 'B1', 'B3', scale_factor=1.2)
        self.play(Write(eqn))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(c2)
        # Using SVG asset
        weather_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/weather.svg", color=c2)
        icons = VGroup(
            weather_icon,
            Text("Weather", color=c2, font_size=24),
            Text("Finance", color=c2, font_size=24),
            Text("Aerodynamics", color=c2, font_size=24)
        ).arrange(RIGHT, buff=0.3)
        self.place_in_area(icons, 'C1', 'C6', scale_factor=0.7)
        self.play(FadeIn(icons))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(c3)
        dot = Dot(color=WHITE, radius=0.1)
        field = VGroup(*[Dot(color=c3, radius=0.05).move_to(np.array([i*0.2-0.5, j*0.2-0.5, 0])) for i in range(5) for j in range(5)])
        self.place_at_grid(dot, 'E3', scale_factor=1.0)
        self.place_at_grid(field, 'E5', scale_factor=1.0)
        self.play(Create(dot))
        self.play(Transform(dot, field))
        self.wait(2)
