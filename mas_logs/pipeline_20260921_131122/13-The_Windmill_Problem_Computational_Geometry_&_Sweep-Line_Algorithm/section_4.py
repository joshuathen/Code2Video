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
        lecture_lines = ["Brute force checks all pairs.", "Sweep-line remembers point order.", "This significantly reduces operations."]
        self.setup_layout("Complexity & Efficiency", lecture_lines)
        
        # Colors for the lecture lines (must only change colors)
        l1, l2, l3 = self.lecture[0], self.lecture[1], self.lecture[2]
        
        # Animation Elements
        comp_formula = MathTex(r"O(N \log N)", color=WHITE)
        self.place_in_area(comp_formula, 'A4', 'B6', scale_factor=0.9)
        
        axes = Axes(x_range=[0, 5, 1], y_range=[0, 5, 1], x_length=2.5, y_length=2.5)
        graph_n2 = axes.plot(lambda x: x**2/5, color=YELLOW)
        graph_nlogn = axes.plot(lambda x: x*np.log(x+1)/2, color=BLUE)
        graph_group = VGroup(axes, graph_n2, graph_nlogn)
        self.place_in_area(graph_group, 'C3', 'E4', scale_factor=0.75)
        
        bar_chart = BarChart([1, 0.2], bar_names=["Brute", "Sweep"], y_range=[0, 1.2, 0.2], x_length=2.5, y_length=2)
        self.place_at_grid(bar_chart, 'D6', scale_factor=0.85)

        # === Animation for Lecture Line 1 ===
        self.play(l1.animate.set_color("#FFFFFF"), Write(comp_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(l2.animate.set_color("#FFFF00"), Create(graph_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(l3.animate.set_color("#00FF00"), Create(bar_chart))
        self.wait(2)
