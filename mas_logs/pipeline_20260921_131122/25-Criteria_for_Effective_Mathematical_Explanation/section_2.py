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
        self.setup_layout("Pillar 1: Precision and Representational Accuracy", [
            "Precision is the first priority.",
            "Definitions must be unambiguous.",
            "Visuals must accurately represent mathematics."
        ])
        
        # Load asset
        graph_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/graph.svg")
        
        # 1. Axes layout
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-1, 3, 1],
            axis_config={"include_tip": False},
        )
        self.place_in_area(axes, 'D2', 'F6', scale_factor=0.6)
        
        # 2. Circle object
        circle = graph_svg.copy()
        self.place_at_grid(circle, 'B4', scale_factor=0.7)
        circle.set_color(RED)
        
        # 3. Vertical line
        vertical_line = Line(axes.get_bottom(), axes.get_top(), color=WHITE)
        self.place_in_area(vertical_line, 'C3', 'D5', scale_factor=0.8)
        
        # Combine everything
        group = VGroup(axes, circle, vertical_line)
        self.add(group)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(RED))
        self.play(FadeIn(circle))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(WHITE))
        self.play(Create(vertical_line))
        self.play(vertical_line.animate.shift(RIGHT * 1.5), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(RED))
        dot1 = Dot(color=RED).move_to(circle.get_center() + UP*0.2)
        dot2 = Dot(color=RED).move_to(circle.get_center() + DOWN*0.2)
        self.play(FadeIn(dot1), FadeIn(dot2))
        self.wait(2)
