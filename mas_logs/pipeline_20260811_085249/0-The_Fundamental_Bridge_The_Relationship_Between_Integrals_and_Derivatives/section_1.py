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
        lecture_lines = ["Calculus links slope and area together.", "Derivative measures the instantaneous slope.", "Integral represents the accumulated area."]
        self.setup_layout("Prerequisite Review: Slope vs. Area", lecture_lines)
        
        # Mobjects
        axes = Axes(x_range=[-2, 2], y_range=[0, 2], axis_config={"include_tip": False}, x_length=4, y_length=3)
        func = axes.plot(lambda x: 0.2*x**2 + 0.5, x_range=[-2, 2], color=WHITE)
        graph_group = VGroup(axes, func)
        self.place_in_area(graph_group, 'B1', 'E5', scale_factor=0.55)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(Create(graph_group))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        # Derivative/Slope
        dot = Dot(color="#FFFF00")
        dot.move_to(func.point_from_proportion(0.5))
        tangent = TangentLine(func, alpha=0.5, length=1, color="#FFFF00")
        self.play(FadeIn(dot), Create(tangent))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        # Area
        area = axes.get_area(func, x_range=[-1, 1], color="#00FFFF", opacity=0.3)
        self.play(Create(area))
