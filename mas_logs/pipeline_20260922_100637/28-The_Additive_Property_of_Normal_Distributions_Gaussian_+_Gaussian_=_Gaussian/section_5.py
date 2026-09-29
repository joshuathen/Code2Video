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
        self.setup_layout("Application and Conclusion", [
            "This property simplifies modeling complex systems.",
            "Just add parameters instead of using complex calculus.",
            "Gaussian noise is universal in nature and engineering."
        ])
        
        engine = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/engine.svg")
        sensor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg")
        
        # === Animation for Lecture Line 1 ===
        sys_group = VGroup(engine, sensor).arrange(RIGHT, buff=0.5)
        self.place_in_area(sys_group, "B2", "D5", scale_factor=0.5)
        self.play(FadeIn(sys_group), self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        plus_sign = Tex("+", font_size=72, color="#FFCC00")
        self.place_at_grid(plus_sign, "C3")
        self.play(FadeOut(sys_group), FadeIn(plus_sign), self.lecture[1].animate.set_color("#FFCC00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        gaussian = FunctionGraph(lambda x: np.exp(-x**2), x_range=[-2, 2], color="#00FF00")
        self.place_at_grid(gaussian, "D3", scale_factor=1.5)
        self.place_at_grid(sensor.copy().set_color("#00FF00"), "F6", scale_factor=0.3)
        self.play(FadeOut(plus_sign), Create(gaussian), FadeIn(sensor), self.lecture[2].animate.set_color("#00FF00"))
        self.play(Indicate(gaussian), Indicate(sensor))
        self.wait(2)
