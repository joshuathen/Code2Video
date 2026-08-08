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
        lecture_lines = [
            "Zoom in to see the interval shrink.", 
            "As dx approaches zero, it transforms.", 
            "Secant line becomes a tangent line.", 
            "Average speed becomes instantaneous speed.", 
            "Limit identifies the exact rate of change."
        ]
        self.setup_layout("The Transformation: The Limit of Shrinkage", lecture_lines)
        
        axes = Axes(x_range=[-1, 3], y_range=[-1, 3], x_length=4, y_length=4)
        graph = axes.plot(lambda x: 0.5 * x**2 + 0.5, color=BLUE)
        
        # Visuals
        p1 = axes.c2p(0.5, 0.625)
        p2 = axes.c2p(2.0, 2.5)
        dot1 = Dot(p1, color=YELLOW)
        dot2 = Dot(p2, color=YELLOW)
        secant = Line(p1, p2, color=RED)
        
        graph_group = VGroup(axes, graph, dot1, dot2, secant)
        
        # === Animation for Lecture Line 1 ===
        # Fix for Issue 24: Use scale_factor=0.7 and area B2-E5
        self.place_in_area(graph_group, "B2", "E5", scale_factor=0.7)
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(graph_group.animate.scale(1.1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Fix for Issue 25: Use scale_factor=0.75 and area C2-F6
        self.place_in_area(graph_group, "C2", "F6", scale_factor=0.75)
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(dot2.animate.move_to(axes.c2p(1.0, 1.0)))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Fix for Issue 26: Use scale_factor=0.65 and area B2-E5
        self.place_in_area(graph_group, "B2", "E5", scale_factor=0.65)
        self.play(self.lecture[2].animate.set_color(YELLOW))
        tangent = TangentLine(graph, alpha=0.5, length=3, color=GREEN)
        self.play(FadeIn(tangent))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(YELLOW))
        self.play(Indicate(tangent))
        self.wait(2)
