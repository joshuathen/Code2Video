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
            "Derivative represents an instantaneous rate of change.",
            "Imagine a speedometer monitoring function changes.",
            "Distance depends on speed and time."
        ]
        self.setup_layout("Prerequisite Warm-up: The Concept of Rates", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Create rate representation using asset
        rate_obj = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg", color="#FFCC00")
        rate_label = Text("Rate", font_size=24, color="#FFCC00")
        rate_group = VGroup(rate_obj, rate_label).arrange(DOWN)
        self.place_at_grid(rate_group, 'B3', scale_factor=0.8)
        self.play(FadeIn(rate_group))
        self.play(self.lecture[0].animate.set_color("#FFCC00"))

        # === Animation for Lecture Line 2 ===
        # Show rate change visually
        speedometer = VGroup(
            Arc(radius=0.6, start_angle=PI/4, angle=3*PI/2, color="#00FFCC"),
            Line(ORIGIN, UP*0.5, color="#00FFCC")
        )
        self.place_at_grid(speedometer, 'B5', scale_factor=0.7)
        self.play(Create(speedometer))
        self.play(Rotate(speedometer[1], angle=PI/2, about_point=speedometer.get_center()))
        self.play(self.lecture[1].animate.set_color("#00FFCC"))

        # === Animation for Lecture Line 3 ===
        # Compare two rates side-by-side using asset
        rate_comp = VGroup(
            SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg", color="#FF66FF"),
            SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg", color="#FF66FF")
        ).arrange(RIGHT, buff=0.5)
        self.place_at_grid(rate_comp, 'D3', scale_factor=0.7)
        self.play(FadeIn(rate_comp))
        self.play(self.lecture[2].animate.set_color("#FF66FF"))
        self.wait(2)
