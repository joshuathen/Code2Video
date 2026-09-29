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
        lines = [
            "We want to find roots where f(x) equals zero.",
            "Newton's Method approximates the root using linear tangent lines.",
            "The tangent hits the x-axis, refining our guess.",
            "Watch the rover iterate toward the root.",
            "Iterative refinement turns guesses into accurate solutions."
        ]
        self.setup_layout("Introduction: The Root-Finding Challenge", lines)
        
        axes = Axes(x_range=[-2, 5], y_range=[-2, 3], axis_config={"include_tip": True}).scale(0.5)
        graph = axes.plot(lambda x: 0.25 * (x-1)**3 + 0.5, color=WHITE)
        axes_and_graph = VGroup(axes, graph)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        prob_text = Text("f(x) = 0", font_size=36, color="#00FFFF")
        self.place_at_grid(prob_text, 'B4', scale_factor=0.7)
        self.play(Write(prob_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        self.place_in_area(axes_and_graph, 'D3', 'F6', scale_factor=0.6)
        self.play(FadeIn(axes), Create(graph))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        root_dot = Dot(axes.c2p(1, 0), color="#FF0000")
        self.play(Create(root_dot))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        rover = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rover.svg")
        self.place_at_grid(rover, 'D2', scale_factor=0.5)
        self.play(FadeIn(rover))
        self.play(rover.animate.move_to(axes.c2p(1, 0)))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#0000FF")
        flash = Flash(axes.c2p(1, 0), color="#FFFF00", line_length=0.2)
        self.play(flash)
        self.wait(2)
