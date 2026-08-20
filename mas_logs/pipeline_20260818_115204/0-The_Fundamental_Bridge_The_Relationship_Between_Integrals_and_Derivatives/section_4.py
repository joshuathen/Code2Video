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
        self.setup_layout("Synthesis and Summary", [
            "Derivative is the slope.", 
            "Integral is the area.", 
            "The theorem bridges these geometric concepts."
        ])
        
        # Colors for the lecture lines
        c1 = "#FF9999"
        c2 = "#99FF99"
        c3 = "#9999FF"

        # Load Assets
        car_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png")
        odometer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/odometer.svg")
        
        # Objects
        f_label = MathTex("f(x)", color=WHITE)
        deriv_label = MathTex("f'(x) = \\text{slope}", color=c1)
        int_label = MathTex("\\int f(x)dx = \\text{area}", color=c2)
        
        self.place_at_grid(f_label, 'B1', scale_factor=0.7)
        self.place_in_area(deriv_label, 'B3', 'B5', scale_factor=0.6)
        self.place_in_area(int_label, 'D3', 'D5', scale_factor=0.6)

        arrow1 = Arrow(start=self.grid["B2"], end=self.grid["B4"], color=WHITE)
        arrow2 = Arrow(start=self.grid["B2"], end=self.grid["D2"], color=WHITE)
        
        # Place assets
        self.place_at_grid(car_icon, "A5", scale_factor=0.2)
        self.place_at_grid(odometer_icon, "E5", scale_factor=0.2)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(c1))
        self.play(FadeIn(car_icon), Create(arrow1), Write(deriv_label))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(c2))
        self.play(Create(arrow2), Write(int_label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(c3))
        self.play(FadeIn(odometer_icon))
        bridge_rect = SurroundingRectangle(VGroup(arrow1, arrow2, f_label, deriv_label, int_label), color=c3, buff=0.3)
        self.play(Create(bridge_rect))
        self.wait(2)
