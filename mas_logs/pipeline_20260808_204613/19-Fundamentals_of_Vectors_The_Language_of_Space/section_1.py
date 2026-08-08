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
        self.setup_layout("What is a Vector? (The Intuition)", [
            "Vectors have both magnitude and direction.",
            "Scalars are just magnitude values.",
            "Think of a drone's flight path.",
            "We represent vectors as arrows.",
            "Arrows show direction and length."
        ])

        # Assets
        drone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/drone.svg")

        # Animation elements
        dot = Dot(color=WHITE)
        arrow = Arrow(start=ORIGIN, end=RIGHT*2 + UP*1, color="#FFFF00")
        tip_highlight = Circle(radius=0.1, color="#FF0000")
        label = MathTex("v", color=WHITE)
        dashed_line = DashedLine(start=ORIGIN, end=RIGHT*2 + UP*1, color=WHITE)

        # Positioning
        self.place_at_grid(dot, 'C3')
        self.place_at_grid(drone, 'C3', scale_factor=0.3)
        self.place_at_grid(arrow, 'D3', scale_factor=1.2)
        self.place_at_grid(tip_highlight, 'D4', scale_factor=0.8)
        self.place_at_grid(label, 'B4', scale_factor=1.0)
        self.place_at_grid(dashed_line, 'D3', scale_factor=1.2)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"), Create(dot), FadeIn(drone))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#8888FF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"), Create(arrow))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF88FF"), Create(tip_highlight))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF8800"), Write(label), Create(dashed_line))
        self.wait(2)
