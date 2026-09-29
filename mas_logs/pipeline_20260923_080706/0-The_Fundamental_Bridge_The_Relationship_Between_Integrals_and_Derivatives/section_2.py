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
        self.setup_layout("Prerequisite Review: Slope vs. Area", [
            "Derivative is the slope of tangent.",
            "Integral is the area under curve.",
            "Observe them side by side."
        ])
        
        # Coordinate Plane
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": True})
        axes.set_color(WHITE)
        curve = axes.plot(lambda x: 0.25 * x**2, x_range=[0, 4], color="#32CD32")
        
        graph_group = VGroup(axes, curve)
        # Apply fix from issue 35 (C2, F5)
        self.place_in_area(graph_group, 'C2', 'F5', scale_factor=0.5)
        self.add(graph_group)
        
        # Add labels
        graph_title = Text("f(x) = 0.25x²", font_size=20)
        # Apply fix from issue 35 (C5)
        self.place_at_grid(graph_title, 'C5', scale_factor=0.8)
        self.add(graph_title)
        
        slope_label = Text("Rate of Change", color="#FF4500", font_size=18)
        area_label = Text("Total Accumulation", color="#1E90FF", font_size=18)
        # Apply fix from issue 35 (B4, E4)
        self.place_at_grid(slope_label, 'B4', scale_factor=0.7)
        self.place_at_grid(area_label, 'E4', scale_factor=0.7)
        self.add(slope_label, area_label)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#32CD32")
        
        # Tangent line (Rate of Change)
        x_val = ValueTracker(2.0)
        tangent = always_redraw(lambda: TangentLine(curve, alpha=x_val.get_value()/4, length=1, color="#FF4500"))
        self.add(tangent)
        self.play(x_val.animate.set_value(3.5), run_time=2)
        self.play(x_val.animate.set_value(0.5), run_time=2)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#1E90FF")
        
        # Area under curve (Total Accumulation)
        area = always_redraw(lambda: axes.get_area(curve, x_range=[0, x_val.get_value()], color="#1E90FF", opacity=0.4))
        self.add(area)
        self.play(x_val.animate.set_value(3.0), run_time=2)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
