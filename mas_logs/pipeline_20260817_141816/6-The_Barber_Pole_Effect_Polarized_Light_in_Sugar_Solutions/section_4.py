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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Biot's Law models this rotation.", "Rotation is proportional to concentration.", "Increased concentration compresses the Barber Pole stripes."]
        self.setup_layout("Mathematical Model: Biot's Law", lecture_lines)
        
        # Math objects
        biot_eq = MathTex(r"\\theta = [\\alpha] \\cdot c \\cdot L", font_size=40)
        self.place_in_area(biot_eq, 'B2', 'B5', scale_factor=0.7)

        # Barber Pole simulation
        beaker = Rectangle(height=3, width=1, color=WHITE)
        self.place_in_area(beaker, 'B3', 'D5', scale_factor=0.9)
        
        # Barber Pole Asset
        barber_pole = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/barberpole.svg")
        self.place_at_grid(barber_pole, 'E4', scale_factor=0.5)
        
        # Stripes
        stripes = VGroup(*[Line(start=[-0.5, y, 0], end=[0.5, y, 0], color=BLUE) for y in np.linspace(-1.4, 1.4, 10)])
        stripes.rotate(PI/4)
        stripes.move_to(beaker.get_center())
        self.add(stripes)

        # Concentration variable
        c = ValueTracker(1.0)
        
        def update_stripes(s):
            val = c.get_value()
            new_stripes = VGroup(*[Line(start=[-0.5, y, 0], end=[0.5, y, 0], color=BLUE) for y in np.linspace(-1.4/val, 1.4/val, 10)])
            new_stripes.rotate(PI/4)
            new_stripes.move_to(beaker.get_center())
            s.become(new_stripes)

        stripes.add_updater(update_stripes)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(biot_eq))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.play(Indicate(biot_eq), run_time=1)
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        self.play(c.animate.set_value(3.0), FadeIn(barber_pole), run_time=3)
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        self.wait(1)
