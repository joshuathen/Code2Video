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
            "Turbulence is chaotic, non-linear, and multi-scale motion.",
            "Cream swirling in coffee demonstrates this complex behavior.",
            "The Reynolds number dictates transition to turbulent flow."
        ]
        self.setup_layout("Introduction: The Chaos of the Coffee Cup", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        turb_circle = Circle(radius=1.5, color=BLUE, stroke_width=2)
        self.place_at_grid(turb_circle, 'C3', scale_factor=0.6)
        self.play(Create(turb_circle), run_time=2)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        coffee = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coffee.svg", color=WHITE)
        cream = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cream.svg", color=WHITE)
        
        self.place_at_grid(coffee, 'C3', scale_factor=0.8)
        self.place_at_grid(cream, 'B2', scale_factor=0.4)
        
        self.play(FadeIn(coffee), FadeIn(cream))
        self.play(cream.animate.move_to(coffee.get_center()), run_time=2)
        self.play(Rotating(cream, radians=PI*2, about_point=coffee.get_center(), run_time=2))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(RED)
        re_label = Text("Re", font_size=36, color=RED)
        arrow = Arrow(start=LEFT, end=RIGHT, color=RED)
        
        self.place_at_grid(re_label, 'E2')
        self.place_at_grid(arrow, 'E4')
        
        self.play(FadeIn(re_label), GrowArrow(arrow))
        self.wait(2)
