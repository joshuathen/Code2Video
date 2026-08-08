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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Inverse Operation: Accumulation", [
            "Integrals act as the inverse of derivatives.",
            "They accumulate tiny changes over time.",
            "This calculates the total displacement traveled."
        ])
        
        # Define objects
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 3, 1], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 0.25 * x**2 + 1, x_range=[0.5, 3.5], color=WHITE)
        area = axes.get_area(curve, x_range=[0.5, 3.5], color="#3385FF", opacity=0.5)
        label = Text("Area = Accumulation", font_size=20, color=WHITE)
        
        car = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png")
        
        # Grid setup
        self.place_in_area(axes, 'B3', 'E6', scale_factor=0.5)
        axes.add(curve, area)
        self.place_at_grid(label, 'E4', scale_factor=0.6)
        
        sweeper = Line(start=axes.c2p(0.5, 0), end=axes.c2p(0.5, 3), color="#FF33A1", stroke_width=4)
        self.place_at_grid(sweeper, 'D4', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"), Write(axes), Create(curve))
        # Place car at the base of the curve
        car.scale(0.1)
        car.move_to(axes.c2p(0.5, 0) + DOWN * 0.2)
        self.add(car)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#FFFF00"))
        self.play(Create(area), Write(label))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#FFFF00"))
        self.add(sweeper)
        # Move car and sweeper
        self.play(
            sweeper.animate.shift(RIGHT * 2), 
            car.animate.move_to(axes.c2p(3.5, 0) + DOWN * 0.2), 
            run_time=3
        )
        self.play(FadeOut(sweeper))
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
